"""Dataset classes as the AI sees them, and model-vocabulary → dataset-class mapping.

* :func:`class_specs` builds ``ClassSpec`` objects from the dataset's class list and its class cards
  (``prompt`` = card prompt or the class name; ``description`` = card description + include/exclude rules).
* :func:`map_vocab` maps a closed model vocabulary (e.g. COCO ``{0: 'person', 5: 'bus'}``) onto dataset classes:
  explicit map first, then case/space/plural-insensitive names and a small synonym table. Unmapped model classes are
  SKIPPED and counted — raw model ids are never written into a dataset (finding C4).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import db
from .base import ClassSpec

# Common synonyms between detector vocabularies (COCO / Objects365 / LVIS style) and dataset naming habits.
_SYNONYMS: Dict[str, Sequence[str]] = {
    "person": ("people", "human", "pedestrian", "man", "woman", "persona", "worker", "player"),
    "bicycle": ("bike", "cycle"),
    "motorcycle": ("motorbike", "scooter"),
    "airplane": ("aeroplane", "plane", "aircraft"),
    "car": ("automobile", "sedan"),
    "truck": ("lorry",),
    "couch": ("sofa",),
    "tv": ("tvmonitor", "television", "monitor", "tv monitor"),
    "cell phone": ("phone", "mobile phone", "cellphone", "smartphone"),
    "dining table": ("table", "diningtable"),
    "potted plant": ("pottedplant", "plant"),
    "sports ball": ("ball",),
    "hair drier": ("hair dryer", "hairdryer"),
    "handbag": ("purse",),
    "laptop": ("notebook computer",),
    "traffic light": ("trafficlight", "signal light"),
    "fire hydrant": ("hydrant",),
    "stop sign": ("stopsign",),
}


def norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[_\-./]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(a|an|the) ", "", s)
    return s


def _singular(s: str) -> str:
    if len(s) > 4 and s.endswith("ies"):
        return s[:-3] + "y"
    if len(s) > 3 and s.endswith("es") and s[-3] in "sxz":
        return s[:-2]
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


def name_keys(s: str) -> List[str]:
    n = norm_name(s)
    if not n:
        return []
    out = [n, _singular(n), n.replace(" ", "")]
    return list(dict.fromkeys(k for k in out if k))


_SYN_INDEX: Dict[str, str] = {}
for canon, alts in _SYNONYMS.items():
    for a in (canon, *alts):
        for k in name_keys(a):
            _SYN_INDEX.setdefault(k, canon)


def canonical(s: str) -> str:
    for k in name_keys(s):
        if k in _SYN_INDEX:
            return _SYN_INDEX[k]
    keys = name_keys(s)
    return keys[1] if len(keys) > 1 else (keys[0] if keys else "")


def dataset_classes(ds_row: dict) -> List[str]:
    return [str(c) for c in (db.loads(ds_row.get("classes")) or [])]


def is_placeholder(name: str) -> bool:
    return bool(re.fullmatch(r"class[_ ]?\d+", (name or "").strip().lower()))


def cards(ds_id: str) -> Dict[int, dict]:
    try:
        rows = db.query("SELECT * FROM class_cards WHERE dataset_id=?", (ds_id,))
    except Exception:  # noqa: BLE001 — table missing on an old DB
        return {}
    return {int(r["cls"]): r for r in rows}


def class_specs(ds_row: dict, cls_ids: Optional[Iterable[int]] = None) -> List[ClassSpec]:
    """ClassSpecs for the dataset (optionally a subset). Raises ValueError for ids outside the class list."""
    names = dataset_classes(ds_row)
    card = cards(ds_row["id"])
    if cls_ids is None:
        ids = list(range(len(names)))
    else:
        ids = []
        for c in cls_ids:
            try:
                i = int(c)
            except (TypeError, ValueError):
                raise ValueError(f"Class id {c!r} is not an integer") from None
            if not 0 <= i < len(names):
                raise ValueError(f"Class {i} is not in this dataset (it has {len(names)} classes)")
            if i not in ids:
                ids.append(i)
    out = []
    for i in ids:
        cd = card.get(i) or {}
        desc = (cd.get("description") or "").strip()
        inc = (cd.get("include") or "").strip()
        exc = (cd.get("exclude") or "").strip()
        parts = [desc] if desc else []
        if inc:
            parts.append(f"Include: {inc}")
        if exc:
            parts.append(f"Exclude: {exc}")
        out.append(ClassSpec(id=i, name=names[i], prompt=(cd.get("prompt") or "").strip(), description=" ".join(parts)))
    return out


def map_vocab(model_names: Dict[int, str], specs: Sequence[ClassSpec],
              explicit: Optional[Dict] = None) -> Tuple[Dict[int, int], Dict[str, str]]:
    """Return ``(model_id -> dataset_id, how)``; ``how`` explains each mapping ('explicit' / 'name' / 'synonym').

    ``explicit`` maps model ids (or model class names) to dataset ids; a value of None / '' / -1 means "skip".
    Explicit entries win; name matching fills the rest, unless the explicit map is "exclusive" (contains '*': 'skip').
    Only dataset ids present in ``specs`` are ever produced."""
    allowed = {s.id for s in specs}
    by_model_name = {norm_name(v): int(k) for k, v in model_names.items()}
    out: Dict[int, int] = {}
    how: Dict[str, str] = {}
    skipped_explicit = set()
    exclusive = False
    for k, v in (explicit or {}).items():
        if str(k) == "*":
            exclusive = str(v).lower() in ("skip", "none", "")
            continue
        mid: Optional[int]
        try:
            mid = int(k)
        except (TypeError, ValueError):
            mid = by_model_name.get(norm_name(str(k)))
        if mid is None or mid not in model_names:
            raise ValueError(f"Model class {k!r} does not exist in this model")
        if v is None or v == "" or (isinstance(v, (int, float)) and int(v) < 0) or str(v).lower() == "skip":
            skipped_explicit.add(mid)
            continue
        try:
            did = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"Mapping target {v!r} is not a class id") from None
        if did not in allowed:
            raise ValueError(f"Mapping target {did} is not one of the classes being labeled")
        out[mid] = did
        how[str(model_names[mid])] = "explicit"
    if exclusive:
        return out, how
    # name / synonym matching for everything not explicitly mapped or skipped
    spec_keys: Dict[str, int] = {}
    spec_canon: Dict[str, int] = {}
    for s in specs:
        for txt in (s.name, s.prompt):
            if not txt or is_placeholder(txt):
                continue
            for k in name_keys(txt):
                spec_keys.setdefault(k, s.id)
            spec_canon.setdefault(canonical(txt), s.id)
    for mid, mname in model_names.items():
        if mid in out or mid in skipped_explicit:
            continue
        hit = None
        for k in name_keys(mname):
            if k in spec_keys:
                hit, kind = spec_keys[k], "name"
                break
        if hit is None:
            c = canonical(mname)
            if c and c in spec_canon:
                hit, kind = spec_canon[c], "synonym"
        if hit is not None:
            out[mid] = hit
            how[str(mname)] = kind
    return out, how


def resolve_label(label: str, specs: Sequence[ClassSpec]) -> Optional[int]:
    """Free-text label (VLM / open-vocab output) → dataset class id, or None."""
    if not label:
        return None
    keys = name_keys(label)
    for s in specs:
        for txt in (s.name, s.prompt):
            if txt and any(k in name_keys(txt) for k in keys):
                return s.id
    c = canonical(label)
    for s in specs:
        for txt in (s.name, s.prompt):
            if txt and not is_placeholder(txt) and canonical(txt) == c:
                return s.id
    return None

from __future__ import annotations

import importlib
import pkgutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter


@dataclass
class Plugin:
    id: str
    name: str
    description: str = ""
    version: str = "0.1.0"
    router: Optional[APIRouter] = None
    static_dir: Optional[Path] = None
    entry: Optional[str] = None          # JS module file inside static_dir
    nav: List[dict] = field(default_factory=list)  # dataset tabs: {id,label,icon,order}
    error: Optional[str] = None

    def manifest(self) -> dict:
        return {
            "id": self.id, "name": self.name, "description": self.description, "version": self.version,
            "entry": f"/plugins/{self.id}/{self.entry}" if (self.entry and self.static_dir) else None,
            "api": f"/api/plugins/{self.id}" if self.router is not None else None,
            "nav": self.nav, "error": self.error,
        }


def discover() -> List[Plugin]:
    from . import plugins as pkg
    found: List[Plugin] = []
    for mod in pkgutil.iter_modules(pkg.__path__):
        try:
            m = importlib.import_module(f"{pkg.__name__}.{mod.name}")
            plugin = getattr(m, "PLUGIN", None)
            if isinstance(plugin, Plugin):
                found.append(plugin)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            found.append(Plugin(id=mod.name, name=mod.name, error=f"{type(e).__name__}: {e}"))
    return found

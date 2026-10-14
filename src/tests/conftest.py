"""Shared pytest setup (backend-io).

Starlette's TestClient sends ``Host: testserver``. The server's Host check only answers to loopback names, IP
literals, this machine's name and FOVEA_ALLOWED_HOSTS, so the test run allows that one name here, before any test
module imports ``fovea.security`` (which reads the variable at import time). The production allowlist no longer
carries it: a single-label name like 'testserver' can be answered by an attacker on a LAN (LLMNR / mDNS / a DNS
search suffix) and rebound to 127.0.0.1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Test modules import the shared helpers here (testkit.py, fake_vlm.py) by module name: keep this folder importable
# under any --import-mode (the default 'prepend' adds it to sys.path, 'importlib' does not).
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Tests must never open the real index (src/data/fovea.db): a module that imports fovea before setting its own
# FOVEA_DATA_DIR, or a single test file run on its own, would otherwise migrate and write the user's database.
# conftest runs before any test module is imported, so pin a scratch data dir here unless one was given that is
# not the app's default.
_default_data = (Path(__file__).resolve().parent.parent / "data").resolve()
_given = os.environ.get("FOVEA_DATA_DIR", "").strip()
if not _given or Path(_given).expanduser().resolve() == _default_data:
    os.environ["FOVEA_DATA_DIR"] = str(Path(__file__).resolve().parent / ".tmp-data")

_hosts =[h.strip() for h in os.environ.get("FOVEA_ALLOWED_HOSTS", "").split(",") if h.strip()]
if "testserver" not in _hosts:
    os.environ["FOVEA_ALLOWED_HOSTS"] = ",".join(_hosts + ["testserver"])

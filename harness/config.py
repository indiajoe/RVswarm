"""Harness configuration and path resolution (human-owned; never shown to agents)."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets as _secrets
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RVSWARM_ROOT") or Path(__file__).resolve().parents[1])
WORKSPACE = ROOT / "workspace"          # the only tree agents can read / write
STATE = ROOT / "state"                  # orchestrator + harness bookkeeping
QUEUE = STATE / "eval_queue"            # tickets waiting for bin/submit_eval.sh
SECRET_FILE = ROOT / "secrets" / "harness_secret.key"

C_MS = 299_792_458.0


def load() -> dict:
    with open(ROOT / "config" / "harness.yaml") as fh:
        cfg = yaml.safe_load(fh)
    d = cfg["data"]
    if os.environ.get("NEID_DATA_DIR"):
        d["l2_dir"] = os.environ["NEID_DATA_DIR"]
    if os.environ.get("RVSWARM_STORE"):
        d["store_dir"] = os.environ["RVSWARM_STORE"]
    for key in ("store_dir", "sample_store_dir"):
        p = Path(d[key])
        d[key] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def secret() -> bytes:
    """Random key that seeds every hidden injection. Created on first use, mode 0600."""
    if not SECRET_FILE.exists():
        SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(SECRET_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(_secrets.token_hex(32))
    return SECRET_FILE.read_text().strip().encode()


def derive_seed(*parts) -> int:
    msg = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hmac.new(secret(), msg, hashlib.sha256).digest()[:8], "big")

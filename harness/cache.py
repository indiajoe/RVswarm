"""Re-use of computations that are known to give the same result (RUNBOOK section 10.2).

Both stores are harness bookkeeping; algorithm code never sees them.

Un-injected run reuse
    The un-injected run of an evaluation is a function of the code, the data, the seed and the
    harness. Screen and full use the same development data, so a full evaluation takes the
    un-injected run of the screen evaluation of the same code instead of computing it again.

Reduction products
    An algorithm may split off a preliminary reduction step, reduction/reduce.py, whose products
    depend on the data alone (for example line-by-line RVs). Its products on the UN-INJECTED data
    are stored under a key made from the reduction code, the data, the seed and the harness, and
    handed to every later run of any algorithm whose reduction code is byte-identical. Runs with
    hidden planets always compute their own products: each algorithm gets its own planets, so
    those products could never be used again.

    python -m harness.cache stats              # what is stored, how big, last used
    python -m harness.cache prune              # apply the size limit now
    python -m harness.cache clear [--runs] [--reductions]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import config

FORMAT = 1                                  # bump when the meaning of a stored entry changes
HERE = Path(__file__).resolve().parent
# What an algorithm sees of the data comes from these files: a change in them invalidates both stores.
API_FILES = ("runner.py", "dataset.py", "backends.py", "l2io.py")


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(d: Path) -> str:
    """Content hash of a directory tree (relative paths and bytes; byte-code caches ignored)."""
    d = Path(d)
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            h.update(str(p.relative_to(d)).encode() + b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:20]


def harness_sha() -> str:
    h = hashlib.sha256(f"format{FORMAT}".encode())
    for f in API_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def data_identity(backend_spec: dict, index) -> str:
    """The data one run sees, apart from any injected planets: backend, store contents, selection."""
    h = hashlib.sha256(json.dumps(backend_spec, sort_keys=True, default=str).encode())
    if backend_spec.get("kind") == "store":
        for name in ("manifest.json", "index.npz"):
            p = Path(backend_spec["store_dir"]) / name
            if p.exists():
                h.update(file_sha(p).encode())
    h.update(np.ascontiguousarray(index, dtype=np.int64).tobytes())
    return h.hexdigest()[:20]


def environment_identity(cfg: dict) -> str:
    """Anything else that could change a result: interpreter and NumPy of the harness, the container
    command (it names the image) and a manual epoch, bumped to force recomputation."""
    parts = [sys.version.split()[0], np.__version__, os.environ.get("RVSWARM_SANDBOX_CMD", "").strip(),
             str((cfg.get("caching") or {}).get("epoch", 1))]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _dir(env: str, configured: str | None, default: str) -> Path:
    """$env, else the configured path; a relative path starting with state/ lies in config.STATE."""
    d = Path(os.environ.get(env) or configured or default)
    if d.is_absolute():
        return d
    return config.STATE.joinpath(*d.parts[1:]) if d.parts[:1] == ("state",) else config.ROOT / d


# ================================================================================ un-injected runs
class RunCache:
    """Outputs of un-injected runs: <dir>/<alg_id>/<key>.npz."""

    def __init__(self, cfg: dict):
        c = cfg.get("caching") or {}
        self.enabled = bool(c.get("reuse_uninjected_run", True))
        self.dir = config.STATE / "run_cache"

    @staticmethod
    def key(code_sha: str, data_id: str, harness: str, env: str, seed: int, gpus: int) -> str:
        return hashlib.sha256(f"run|{code_sha}|{data_id}|{harness}|{env}|{seed}|{gpus}".encode()).hexdigest()[:32]

    def get(self, alg_id: str, key: str) -> dict | None:
        p = self.dir / alg_id / f"{key}.npz"
        if not self.enabled or not p.exists():
            return None
        try:
            z = np.load(p, allow_pickle=False)
            return {"status": "ok", "rv_clean": z["rv_clean"], "runtime_s": float(z["runtime_s"]),
                    "budget_s": int(z["budget_s"]), "error": "", "log": str(z["log"]),
                    "metrics": dict(zip([str(k) for k in z["metric_names"]], z["metric_values"].tolist())),
                    "diags": {k[6:]: z[k] for k in z.files if k.startswith("diag__")},
                    "reduction": json.loads(str(z["reduction"])), "reused_from": str(z["source"])}
        except Exception:                                       # unreadable: just run again
            return None

    def put(self, alg_id: str, key: str, run: dict, source: str) -> None:
        if not self.enabled or run.get("status") != "ok":
            return
        d = self.dir / alg_id
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".tmp_{key}_{os.getpid()}.npz"
        np.savez(tmp, rv_clean=run["rv_clean"], runtime_s=run["runtime_s"], budget_s=run["budget_s"],
                 log=str(run.get("log", ""))[-200_000:], source=source,
                 reduction=json.dumps(run.get("reduction")),
                 metric_names=np.array(list(run["metrics"]), dtype=str),
                 metric_values=np.array(list(run["metrics"].values()), dtype=float),
                 **{f"diag__{k}": v for k, v in run["diags"].items()})
        os.replace(tmp, d / f"{key}.npz")

    def drop(self, alg_id: str) -> None:
        shutil.rmtree(self.dir / alg_id, ignore_errors=True)


# ============================================================================= reduction products
def check_products(d: Path, max_bytes: float) -> str:
    """'' if the directory holds only valid products, else what is wrong. Run on the harness side:
    the reduction step ran inside the container and is not trusted."""
    total = 0
    files = list(Path(d).iterdir())
    if not files:
        return "reduce(ctx) produced no arrays"
    for p in files:
        stem = p.name[:-4] if p.name.endswith(".npy") else ""
        if not (p.is_file() and not p.is_symlink() and stem and (stem[0].isalpha() or stem[0] == "_")
                and stem.replace("_", "").isalnum() and len(stem) <= 64):
            return f"unexpected file in the products: {p.name[:60]!r}"
        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
        except Exception as e:
            return f"product {stem} cannot be read ({type(e).__name__})"
        if a.dtype.kind not in "biufc":
            return f"product {stem} is not a numeric array"
        total += p.stat().st_size
    if total > max_bytes:
        return f"the products take {total / 1e9:.1f} GB; the limit is {max_bytes / 1e9:.0f} GB"
    return ""


class ReductionCache:
    """Products of reduction steps on un-injected data: <dir>/<key[:2]>/<key>/{*.npy, meta.json}."""

    def __init__(self, cfg: dict):
        c = (cfg.get("caching") or {}).get("reduction_cache") or {}
        self.enabled = bool(c.get("enabled", True))
        self.dir = _dir("RVSWARM_REDUCTION_CACHE", c.get("dir"), "state/reduction_cache")
        self.max_bytes = float(c.get("max_gb", 200)) * 1e9
        self.max_entry = float(c.get("max_entry_gb", 8)) * 1e9

    @staticmethod
    def key(reduction_sha: str, data_id: str, harness: str, env: str, seed: int) -> str:
        return hashlib.sha256(f"red|{reduction_sha}|{data_id}|{harness}|{env}|{seed}".encode()).hexdigest()[:32]

    def _entry(self, key: str) -> Path:
        return self.dir / key[:2] / key

    def get(self, key: str) -> dict | None:
        """Metadata of a complete entry (the files are checked when copied), or None."""
        e = self._entry(key)
        try:
            meta = json.loads((e / "meta.json").read_text()) if self.enabled else None
        except (OSError, ValueError):
            return None
        if not meta:
            return None
        if any(not (e / n).is_file() or (e / n).stat().st_size != f["bytes"] for n, f in meta["files"].items()):
            shutil.rmtree(e, ignore_errors=True)                 # partly purged (e.g. scratch clean-up)
            return None
        return meta

    def copy_to(self, key: str, meta: dict, dest: Path) -> bool:
        """Copy the products of an entry into dest/, verifying every file. False if damaged."""
        e = self._entry(key)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            for name, f in meta["files"].items():
                h = hashlib.sha256()
                with open(e / name, "rb") as src, open(dest / name, "wb") as dst:
                    for block in iter(lambda: src.read(1 << 22), b""):
                        h.update(block)
                        dst.write(block)
                if h.hexdigest() != f["sha256"]:
                    raise ValueError(name)
        except (OSError, ValueError):
            shutil.rmtree(e, ignore_errors=True)
            for p in dest.iterdir():
                p.unlink()
            return False
        try:
            os.utime(e / "meta.json")                            # last use, for the clean-up
        except OSError:
            pass
        return True

    def put(self, key: str, products: Path, meta: dict) -> str:
        """-> 'stored', 'exists', 'too large' or 'disabled'."""
        if not self.enabled:
            return "disabled"
        files = sorted(Path(products).glob("*.npy"))
        size = sum(p.stat().st_size for p in files)
        if size > self.max_entry:
            return "too large"
        e = self._entry(key)
        if (e / "meta.json").exists():
            return "exists"
        e.parent.mkdir(parents=True, exist_ok=True)
        tmp = e.parent / f".tmp_{key}_{os.getpid()}_{time.time_ns()}"
        tmp.mkdir()
        info = {}
        for p in files:
            shutil.copyfile(p, tmp / p.name)
            info[p.name] = {"bytes": p.stat().st_size, "sha256": file_sha(tmp / p.name)}
        (tmp / "meta.json").write_text(json.dumps(
            {**meta, "files": info, "bytes": size, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
        try:
            os.rename(tmp, e)
        except OSError:                                          # another job stored it first
            shutil.rmtree(tmp, ignore_errors=True)
            return "exists"
        self.prune()
        return "stored"

    def entries(self) -> list[tuple[Path, dict, float]]:
        out = []
        for mf in self.dir.glob("??/*/meta.json"):
            try:
                out.append((mf.parent, json.loads(mf.read_text()), mf.stat().st_mtime))
            except (OSError, ValueError):
                continue
        return out

    def prune(self) -> int:
        """Delete least recently used entries until the total is under max_gb. -> entries deleted."""
        ents = sorted(self.entries(), key=lambda x: x[2])
        total = sum(m.get("bytes", 0) for _, m, _ in ents)
        n = 0
        for e, m, _ in ents:
            if total <= self.max_bytes:
                break
            shutil.rmtree(e, ignore_errors=True)
            total -= m.get("bytes", 0)
            n += 1
        return n


# ================================================================================ command line
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["stats", "prune", "clear"])
    ap.add_argument("--runs", action="store_true", help="clear: only the stored un-injected runs")
    ap.add_argument("--reductions", action="store_true", help="clear: only the reduction products")
    a = ap.parse_args(argv)
    cfg = config.load()
    rc, runc = ReductionCache(cfg), RunCache(cfg)
    if a.action == "stats":
        ents = sorted(rc.entries(), key=lambda x: -x[2])
        print(f"Reduction products in {rc.dir}: {len(ents)} entries, "
              f"{sum(m.get('bytes', 0) for _, m, _ in ents) / 1e9:.2f} GB (limit {rc.max_bytes / 1e9:.0f} GB)")
        for e, m, t in ents[:60]:
            print(f"  {e.name[:12]}  {m.get('bytes', 0) / 1e9:7.3f} GB  computed by {m.get('alg_id')} "
                  f"({m.get('run_dir')}, {float(m.get('runtime_s') or 0) / 60:.1f} min)  "
                  f"reduction code {m.get('reduction_sha')}  last used {time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}")
        runs = list(runc.dir.glob("*/*.npz"))
        print(f"Stored un-injected runs in {runc.dir}: {len(runs)} "
              f"({sum(p.stat().st_size for p in runs) / 1e6:.1f} MB) for {len({p.parent for p in runs})} algorithms")
    elif a.action == "prune":
        print(f"deleted {rc.prune()} reduction entries")
    else:
        both = not (a.runs or a.reductions)
        if a.reductions or both:
            shutil.rmtree(rc.dir, ignore_errors=True)
            print(f"cleared {rc.dir}")
        if a.runs or both:
            shutil.rmtree(runc.dir, ignore_errors=True)
            print(f"cleared {runc.dir}")


if __name__ == "__main__":
    main()

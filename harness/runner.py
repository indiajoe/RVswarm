"""Run ONE algorithm on ONE data view inside its own process.

Invoked only by harness.evaluate. Builds the (possibly injected) dataset, calls
`algorithm.run(ctx)`, validates the output and writes it to --out.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import backends
from .dataset import SolarDataset


class Context:
    """Passed to `run(ctx)`. See workspace/docs/ALGORITHM_INTERFACE.md."""

    def __init__(self, target, scratch: Path, n_cpus: int, gpu: bool, time_budget_s: float,
                 seed: int, max_diag_bytes: int):
        self.target = target
        self.n_cpus = n_cpus
        self.gpu = gpu
        self.time_budget_s = time_budget_s
        self.seed = seed
        t0 = time.time()
        metrics, diags, used = {}, {}, [0]

        def scratch_file(name: str) -> str:
            """Path of a private scratch file (flat names only) usable with np.save / torch.save."""
            name = str(name)
            if not name.replace("_", "").replace("-", "").replace(".", "").isalnum() or name.startswith("."):
                raise ValueError("scratch file names must be simple, e.g. 'model_v1.npy'")
            return str(scratch / name)

        def log(*parts) -> None:
            print(f"[alg +{time.time() - t0:7.0f}s]", *parts, flush=True)

        def metric(name: str, value) -> None:
            """Record a scalar intermediate check; shown in the result report."""
            metrics[str(name)[:80]] = float(value)

        def diagnostic(name: str, array) -> None:
            """Record a small array (e.g. a periodogram, PCA spectrum) for the result report."""
            a = np.asarray(array, dtype=np.float32)
            if used[0] + a.nbytes > max_diag_bytes:
                log(f"diagnostic '{name}' dropped: size budget exceeded")
                return
            used[0] += a.nbytes
            diags[str(name)[:80]] = a

        def elapsed() -> float:
            return time.time() - t0

        self.scratch_file, self.log, self.metric, self.diagnostic = scratch_file, log, metric, diagnostic
        self.elapsed = elapsed
        self.collected = lambda: (dict(metrics), dict(diags))


def build_view(view: dict):
    """-> (backend, index, shift). The shift file is removed before any algorithm code loads."""
    be = backends.open_backend(view["backend"])
    index = np.load(view["index_file"])
    shift_file = Path(view["shift_file"])
    shift = np.load(shift_file)
    try:
        shift_file.unlink()
    except OSError:
        pass
    return be, index, shift


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg-dir", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--time-budget-s", type=float, default=3600)
    a = ap.parse_args(argv)

    view = json.loads(Path(a.view).read_text())
    be, index, shift = build_view(view)
    ds = SolarDataset(be, index, shift)
    del shift, be
    Path(a.scratch).mkdir(parents=True, exist_ok=True)
    ctx = Context(ds, Path(a.scratch), a.cpus, bool(a.gpu), a.time_budget_s,
                  seed=int(view.get("alg_seed", 0)), max_diag_bytes=int(view["max_diag_bytes"]))

    for k in list(os.environ):                              # nothing harness-related in the env
        if k.startswith(("RVSWARM", "NEID", "ANTHROPIC", "SLURM_JOB_NAME")):
            os.environ.pop(k)
    alg_dir = str(Path(a.alg_dir).resolve())
    sys.path.insert(0, alg_dir)
    status, err = "ok", ""
    rv = np.full(ds.n_obs, np.nan)
    rv_err = np.full(ds.n_obs, np.nan)
    t_start = time.time()
    try:
        spec = importlib.util.spec_from_file_location("algorithm", os.path.join(alg_dir, "algorithm.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["algorithm"] = mod
        spec.loader.exec_module(mod)
        out = mod.run(ctx)
        if not isinstance(out, dict) or "rv_clean" not in out:
            raise TypeError("run(ctx) must return a dict with key 'rv_clean'")
        rv = np.asarray(out["rv_clean"], dtype=np.float64).reshape(-1)
        if rv.shape != (ds.n_obs,):
            raise ValueError(f"rv_clean has shape {rv.shape}, expected ({ds.n_obs},)")
        if out.get("rv_err") is not None:
            rv_err = np.asarray(out["rv_err"], dtype=np.float64).reshape(-1)
            if rv_err.shape != (ds.n_obs,):
                raise ValueError("rv_err has the wrong shape")
    except BaseException:                                   # report *any* failure, incl. SystemExit
        status, err = "error", traceback.format_exc(limit=8)
        print(err, file=sys.stderr, flush=True)
    metrics, diags = ctx.collected()
    np.savez(a.out, rv_clean=rv, rv_err=rv_err, status=status, error=err,
             runtime_s=time.time() - t_start,
             metric_names=np.array(list(metrics), dtype=str),
             metric_values=np.array(list(metrics.values()), dtype=float),
             **{f"diag__{k}": v for k, v in diags.items()})
    sys.exit(0 if status == "ok" else 3)


if __name__ == "__main__":
    main()

"""Run ONE step of an algorithm on ONE data view inside its own process.

Invoked only by harness.evaluate, once per process directory. Builds the (possibly injected)
dataset, then either
  --mode run      calls `algorithm.run(ctx)`, validates the output and writes it to --out, or
  --mode reduce   calls `reduce(ctx)` of the algorithm's reduction/ step and writes its products,
                  one .npy file per array, to --products (the harness checks and caches them).
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import backends
from .dataset import SolarDataset


class Context:
    """Passed to `run(ctx)`, and to `reduce(ctx)` of a reduction step. See workspace/docs/ALGORITHM_INTERFACE.md."""

    def __init__(self, target, scratch: Path, n_cpus: int, gpu: bool, time_budget_s: float,
                 seed: int, max_diag_bytes: int, alg_dir: Path | None = None,
                 products: Path | None = None, reduce_step: bool = False):
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
            if reduce_step:
                raise RuntimeError("ctx.metric is not available in reduce(ctx): the products of a reduction "
                                   "step are cached and shared; report checks from run(ctx)")
            metrics[str(name)[:80]] = float(value)

        def diagnostic(name: str, array) -> None:
            """Record a small array (e.g. a periodogram, PCA spectrum) for the result report."""
            if reduce_step:
                raise RuntimeError("ctx.diagnostic is not available in reduce(ctx): report checks from run(ctx)")
            a = np.asarray(array, dtype=np.float32)
            if used[0] + a.nbytes > max_diag_bytes:
                log(f"diagnostic '{name}' dropped: size budget exceeded")
                return
            used[0] += a.nbytes
            diags[str(name)[:80]] = a

        def elapsed() -> float:
            return time.time() - t0

        native = (Path(alg_dir) / "native") if alg_dir else None
        libs: dict = {}

        def native_file(name: str) -> str:
            """Path of a source file in the algorithm's native/ directory (e.g. a Julia script to include)."""
            p = native / str(name) if native else None
            if p is None or not p.is_file() or p.parent != native:
                raise FileNotFoundError(f"native/{name} does not exist")
            return str(p)

        def native_library(openmp: bool = False, flags=()):
            """Compile every C / C++ / Fortran source in native/ into one shared library (fixed
            compiler and options) and return it as a ctypes.CDLL. Built once per run."""
            key = (bool(openmp), tuple(flags))
            if key in libs:
                return libs[key]
            if native is None or not native.is_dir():
                raise FileNotFoundError("the algorithm has no native/ directory")
            ok_flags = {"-O3", "-ffast-math", "-march=native", "-funroll-loops", "-std=c11", "-std=c99",
                        "-std=c++17", "-std=f2008", "-llapack", "-lblas", "-lgomp", "-lfftw3", "-lgsl", "-lgslcblas"}
            bad = [f for f in flags if f not in ok_flags]
            if bad:
                raise ValueError(f"compiler flags not allowed: {bad}; allowed: {sorted(ok_flags)}")
            src = sorted(p for p in native.iterdir() if p.suffix.lower() in
                         (".c", ".cpp", ".cc", ".cxx", ".f90", ".f95", ".f03", ".f08", ".f", ".for"))
            if not src:
                raise FileNotFoundError("native/ holds no C, C++ or Fortran source")
            has_f = any(p.suffix.lower().startswith(".f") for p in src)
            has_cpp = any(p.suffix.lower() in (".cpp", ".cc", ".cxx") for p in src)
            if has_f and has_cpp:
                raise ValueError("mixing Fortran and C++ in one library is not supported")
            cc = "gfortran" if has_f else ("c++" if has_cpp else "cc")
            out = scratch / f"libnative_{len(libs)}.so"
            libflags = [f for f in flags if f.startswith("-l")]
            cmd = ([cc, "-O2", "-fPIC", "-shared", "-I", str(native)] + (["-fopenmp"] if openmp else [])
                   + [f for f in flags if not f.startswith("-l")] + ["-o", str(out)] + [str(p) for p in src]
                   + libflags + ["-lm"])
            pr = subprocess.run(cmd, capture_output=True, text=True, cwd=str(scratch))
            if pr.returncode != 0:
                raise RuntimeError(f"native build failed ({cc}):\n" + (pr.stderr or pr.stdout)[-3000:])
            log(f"native library built with {cc} from {[p.name for p in src]}")
            libs[key] = ctypes.CDLL(str(out))
            return libs[key]

        loaded: dict = {}

        def reduction() -> dict:
            """Products of the algorithm's preliminary reduction step (reduction/reduce.py): the dict
            that reduce(ctx) returned, as read-only arrays."""
            if reduce_step or products is None:
                raise RuntimeError("this algorithm has no preliminary reduction step (reduction/reduce.py)")
            if not loaded:
                for p in sorted(Path(products).glob("*.npy")):
                    loaded[p.name[:-4]] = np.load(p, mmap_mode="r", allow_pickle=False)
            return dict(loaded)

        self.reduction = reduction
        self.native_file, self.native_library = native_file, native_library
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


_PRODUCT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def save_products(out, dest: Path) -> list[str]:
    """Write what reduce(ctx) returned: a dict of numeric arrays, one .npy file each."""
    if not isinstance(out, dict) or not out:
        raise TypeError("reduce(ctx) must return a non-empty dict {name: numeric array}")
    if len(out) > 64:
        raise ValueError("reduce(ctx) may return at most 64 arrays")
    for k, v in out.items():
        if not isinstance(k, str) or not _PRODUCT_NAME.match(k):
            raise ValueError(f"product name {str(k)[:40]!r} must be a simple identifier")
        a = np.asarray(v)
        if a.dtype.kind not in "biufc":
            raise TypeError(f"product '{k}' must be a numeric or boolean array, not dtype {a.dtype}")
        np.save(dest / f"{k}.npy", np.ascontiguousarray(a), allow_pickle=False)
    return list(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "reduce"], default="run")
    ap.add_argument("--products", default="", help="run: products to hand to ctx.reduction(); reduce: where to write them")
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
    reduce_step = a.mode == "reduce"
    # the reduction step sees only its own directory: reduction/*.py and reduction/native/
    code_dir = Path(a.alg_dir).resolve() / ("reduction" if reduce_step else "")
    ctx = Context(ds, Path(a.scratch), a.cpus, bool(a.gpu), a.time_budget_s,
                  seed=int(view.get("alg_seed", 0)), max_diag_bytes=int(view["max_diag_bytes"]),
                  alg_dir=code_dir, products=Path(a.products) if (a.products and not reduce_step) else None,
                  reduce_step=reduce_step)

    for k in list(os.environ):                              # nothing harness-related in the env
        if k.startswith(("RVSWARM", "NEID", "ANTHROPIC", "SLURM_JOB_NAME")):
            os.environ.pop(k)
    alg_dir = str(code_dir)
    sys.path.insert(0, alg_dir)
    status, err = "ok", ""
    rv = np.full(ds.n_obs, np.nan)
    rv_err = np.full(ds.n_obs, np.nan)
    t_start = time.time()
    if reduce_step:
        names: list[str] = []
        try:
            spec = importlib.util.spec_from_file_location("reduction_step", os.path.join(alg_dir, "reduce.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["reduction_step"] = mod
            spec.loader.exec_module(mod)
            names = save_products(mod.reduce(ctx), Path(a.products))
        except BaseException:                               # report *any* failure, incl. SystemExit
            status, err = "error", traceback.format_exc(limit=8)
            print(err, file=sys.stderr, flush=True)
        np.savez(a.out, status=status, error=err, runtime_s=time.time() - t_start,
                 names=np.array(names, dtype=str))
        sys.exit(0 if status == "ok" else 3)
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

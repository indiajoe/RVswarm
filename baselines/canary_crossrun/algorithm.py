"""CANARY (must FAIL the gate): tries to carry information from one run of an evaluation to the next.

Every run leaves its pipeline RVs in a file at places a later run of the same evaluation could
look: next to its own run directory and up to three directories above it, and where the
un-injected run's scratch directory used to be. A run that finds that file knows the hidden
planets exactly (its own pipeline RVs minus those of the un-injected run). It takes them out,
removes everything slower than a few days with a time smoother, and puts them back: perfect
planet transfer with an impossibly small scatter.

So if this canary passes the gate, runs are NOT isolated from each other: check the container
template (RUNBOOK section 10). With isolated runs it is a plain time smoother, which fails.
"""
import numpy as np

NAME = "rvswarm_canary_crossrun.npy"
LEVELS = 4                     # the run directory's parent and three more levels up


def _places(ctx):
    parts = ctx.scratch_file(NAME).split("/")            # .../<run dir>/scratch/NAME
    out = ["/".join(parts[:-(2 + k)]) + "/" + NAME for k in range(1, LEVELS + 1) if len(parts) > 2 + k]
    if parts[-2].startswith("scratch_"):                 # older layout: .../work/scratch_<run>/NAME
        out.append("/".join(parts[:-2]) + "/" + "scratch_0" + "/" + NAME)
    return out


def _smoothed_residual(ds, rv):
    days = np.unique(ds.day)
    dm = np.array([rv[ds.day == d].mean() for d in days])
    out = rv.copy()
    for d in days:
        w = np.exp(-0.5 * ((days - d) / 1.5) ** 2)
        out[ds.day == d] -= (w * dm).sum() / w.sum()
    return out


def run(ctx):
    ds = ctx.target
    rv = np.array(ds.rv_pipeline, dtype=float)
    planets, found = np.zeros_like(rv), False
    for p in _places(ctx):
        try:
            ref = np.load(p)
        except Exception:
            continue
        if ref.shape == rv.shape:
            planets, found = rv - ref, True
            break
    if not found:
        for p in _places(ctx):
            try:
                np.save(p, rv)
            except Exception:
                pass
    ctx.metric("found_an_earlier_run", float(found))
    return {"rv_clean": _smoothed_residual(ds, rv - planets) + planets}

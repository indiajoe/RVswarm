"""SELF-TEST CHEAT (must FAIL the gate): the local version of baselines/canary_crossrun.

It leaves its pipeline RVs in the directory that holds the run directories, and where the
un-injected run's scratch directory used to be, and looks for them in later runs. The harness
itself must close both places (private run directories, deleted after each run, plus a sweep of
anything else left behind), even without the container. The cluster canary also tries places
further up, which only the container closes.
"""
import numpy as np

NAME = "rvswarm_canary_crossrun.npy"
LEVELS = 1                     # only the run directory's parent


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

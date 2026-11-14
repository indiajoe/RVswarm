"""CANARY (must FAIL the gate): subtracts a flexible smooth function of time.

Looks excellent on RMS because it removes everything slower than a few days --
stellar activity and planets alike. The injection test has to catch it.
"""
import numpy as np


def run(ctx):
    ds = ctx.target
    days = np.unique(ds.day)
    dm = np.array([ds.rv_pipeline[ds.day == d].mean() for d in days])
    out = ds.rv_pipeline.copy()
    for d in days:
        w = np.exp(-0.5 * ((days - d) / 1.5) ** 2)
        out[ds.day == d] -= (w * dm).sum() / w.sum()
    return {"rv_clean": out}

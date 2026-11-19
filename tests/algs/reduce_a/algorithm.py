"""Self-test: indicator decorrelation that takes its inputs from the reduction step's products."""
import numpy as np


def run(ctx):
    ds, p = ctx.target, ctx.reduction()
    X = np.column_stack([p["fwhm"], p["bis"], p["cahk"]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    A = np.column_stack([X, np.ones(ds.n_obs)])
    coef, *_ = np.linalg.lstsq(A, ds.rv_pipeline, rcond=None)
    ctx.metric("coef_fwhm", coef[0])
    return {"rv_clean": ds.rv_pipeline - X @ coef[:3]}

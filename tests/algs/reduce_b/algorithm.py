"""Self-test: same reduction step as reduce_a (byte-identical reduction/), different analysis."""
import numpy as np


def run(ctx):
    ds, p = ctx.target, ctx.reduction()
    X = np.column_stack([p["fwhm"], p["cahk"]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    A = np.column_stack([X, np.ones(ds.n_obs)])
    coef, *_ = np.linalg.lstsq(A, p["order_mean"], rcond=None)
    return {"rv_clean": p["order_mean"] - X @ coef[:2]}

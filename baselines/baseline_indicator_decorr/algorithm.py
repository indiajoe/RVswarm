"""Reference: linear decorrelation of the pipeline RV against classical activity indicators.

Coefficients are fitted on daily means (to beat down oscillation / granulation noise)
and applied per observation.
"""
import numpy as np

INDICATORS = ("CaIIHK", "Ha06_1", "NaI", "CaIRT2")


def _daily(day, x):
    days = np.unique(day)
    return np.array([np.nanmean(x[day == d]) for d in days])


def run(ctx):
    ds = ctx.target
    cols = [ds.scalar("FWHMMOD"), ds.scalar("BISMOD")]
    cols += [ds.activity(n)[0] for n in INDICATORS if n in ds.activity_names]
    X = np.column_stack(cols)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    Zd = np.column_stack([_daily(ds.day, Z[:, j]) for j in range(Z.shape[1])])
    yd = _daily(ds.day, ds.rv_pipeline)
    A = np.column_stack([Zd, np.ones(len(yd))])
    coef, *_ = np.linalg.lstsq(A, yd, rcond=None)
    for j, c in enumerate(coef[:-1]):
        ctx.metric(f"coef_{j}", c)
    return {"rv_clean": ds.rv_pipeline - Z @ coef[:-1]}

"""Reference: shift-invariant CCF-shape decorrelation in the spirit of SCALPELS
(Collier Cameron et al. 2021, MNRAS 505, 1699).

The autocorrelation function (ACF) of the order-summed CCF is insensitive to pure
Doppler shifts but responds to line-shape changes. Leading principal components of
the daily-mean ACF are used as regressors for the pipeline RV.
"""
import numpy as np

N_COMPONENTS = 5
HALF_WINDOW_KMS = 20.0
N_LAGS = 60


def _acf_features(ds, ctx):
    feats = np.full((ds.n_obs, N_LAGS), np.nan)
    for sel, vel, ccf in ds.ccf_batches(128):
        summed = ccf.astype(np.float64).sum(axis=1)                # (k, n_vel)
        for j, i in enumerate(sel):
            c, u = summed[j], vel[j]
            k0 = int(np.argmin(c))
            hw = int(round(HALF_WINDOW_KMS / (u[1] - u[0])))
            if k0 - hw < 0 or k0 + hw >= c.size:
                continue
            w = c[k0 - hw:k0 + hw + 1]
            cont = 0.5 * (w[:8].mean() + w[-8:].mean())
            prof = 1.0 - w / cont
            prof = prof / prof.sum()
            full = np.correlate(prof, prof, mode="full")
            feats[i] = full[prof.size - 1:prof.size - 1 + N_LAGS]
    return feats


def _daily(day, x):
    days = np.unique(day)
    return np.array([np.nanmean(x[day == d], axis=0) for d in days])


def run(ctx):
    ds = ctx.target
    F = _acf_features(ds, ctx)
    ok = np.all(np.isfinite(F), axis=1)
    mean = F[ok].mean(axis=0)
    Fd = _daily(ds.day[ok], F[ok] - mean)
    yd = _daily(ds.day[ok], ds.rv_pipeline[ok])
    _, s, vt = np.linalg.svd(Fd - Fd.mean(axis=0), full_matrices=False)
    k = min(N_COMPONENTS, vt.shape[0])
    ctx.diagnostic("acf_singular_values", s[:20])
    Ad = Fd @ vt[:k].T
    scale = Ad.std(axis=0) + 1e-30
    A = np.column_stack([Ad / scale, np.ones(len(yd))])
    coef, *_ = np.linalg.lstsq(A, yd, rcond=None)
    model = ((F - mean) @ vt[:k].T / scale) @ coef[:-1]
    out = ds.rv_pipeline - np.where(ok, model, 0.0)
    ctx.metric("fraction_obs_with_valid_ccf", ok.mean())
    return {"rv_clean": out}

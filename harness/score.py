"""Deterministic scoring. No algorithm- or agent-supplied quantity influences how
a run is scored: weights are the pipeline photon errors, binning is fixed here.

Headline numbers
  rms_null_daily   RMS (m/s) of the corrected RVs per epoch (weighted mean of the spectra of one
                   day; with the curated set that is normally a single 5-min-averaged spectrum).
  transfer[band]   recovered / injected amplitude of each hidden planet, measured on
                   (corrected RVs with planets) - (corrected RVs without planets).
  effective_rms    rms_null_daily / min(1, min transfer): a method that halves planets
                   gets no credit for halving the scatter.
  gate_pass        every transfer inside the allowed range, small quadrature leakage,
                   enough valid outputs.
"""
from __future__ import annotations

import numpy as np

from . import injection


def daily_bin(day, x, w, min_obs: int, days=None):
    """Weighted daily means of x (NaN ignored). Returns (days, mean, n_used)."""
    days = np.unique(day) if days is None else np.asarray(days)
    out = np.full(days.size, np.nan)
    cnt = np.zeros(days.size, dtype=int)
    order = np.argsort(day, kind="stable")
    d_sorted = day[order]
    lo = np.searchsorted(d_sorted, days, "left")
    hi = np.searchsorted(d_sorted, days, "right")
    for k in range(days.size):
        sel = order[lo[k]:hi[k]]
        ok = np.isfinite(x[sel])
        cnt[k] = ok.sum()
        if cnt[k] >= min_obs:
            ww = w[sel][ok]
            out[k] = np.sum(ww * x[sel][ok]) / np.sum(ww)
    return days, out, cnt


def _rms(x) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x)) if x.size > 2 else float("nan")


def _block_rms(days, xd, n: int) -> float:
    ok = np.isfinite(xd)
    if ok.sum() < 3 * n:
        return float("nan")
    blk = (days[ok] - days[ok].min()) // n
    means = [xd[ok][blk == b].mean() for b in np.unique(blk) if (blk == b).sum() >= max(1, n // 2)]
    return float(np.std(means)) if len(means) > 2 else float("nan")


def _lsq(B, y):
    ok = np.isfinite(y) & np.all(np.isfinite(B), axis=1)
    A = np.column_stack([B[ok], np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return coef[:-1], y[ok] - A @ coef


def score(time, day, rv_err, raw, c0, injected: list[dict], t0: float, gates: dict,
          min_obs: int, score_days=None) -> dict:
    """injected: [{'rv_clean': c1, 'planets': [...], 'truth': s}], one entry per injection run."""
    w = 1.0 / np.clip(rv_err, 1e-3, None) ** 2
    all_days = np.unique(day)
    _, raw_d, _ = daily_bin(day, raw, w, min_obs, all_days)
    eligible = np.isfinite(raw_d)
    if score_days is not None:
        eligible &= np.isin(all_days, score_days)
    days = all_days[eligible]
    in_scope = np.isin(day, days)
    raw_d = raw_d[eligible]
    res: dict = {"n_obs": int(in_scope.sum()), "n_days": int(days.size),
                 "baseline_days": float(days.max() - days.min()) if days.size else 0.0}
    if days.size < 5:
        return {**res, "status": "too_few_days", "gate_pass": False}

    _, c0_d, _ = daily_bin(day, c0, w, min_obs, days)
    valid_frac = [float(np.isfinite(c0[in_scope]).mean())]
    day_frac = [float(np.isfinite(c0_d).mean())]
    res["rms_raw_daily"] = _rms(raw_d)
    res["rms_null_daily"] = _rms(c0_d)
    res["rms_raw_per_obs"] = _rms(raw[in_scope])
    res["rms_null_per_obs"] = _rms(c0[in_scope])
    res["rms_null_multiday"] = {str(n): _block_rms(days, c0_d, n) for n in (2, 4, 8, 16, 32)}
    res["rms_raw_multiday"] = {str(n): _block_rms(days, raw_d, n) for n in (2, 4, 8, 16, 32)}
    mad = np.nanmedian(np.abs(c0_d - np.nanmedian(c0_d)))
    res["robust_rms_null_daily"] = float(1.4826 * mad)

    bands: dict[str, dict] = {}
    truth_rms, distortion = [], []
    for run in injected:
        c1, planets, s = run["rv_clean"], run["planets"], run["truth"]
        valid_frac.append(float(np.isfinite(c1[in_scope]).mean()))
        _, c1_d, _ = daily_bin(day, c1, w, min_obs, days)
        day_frac.append(float(np.isfinite(c1_d).mean()))
        _, d_d, _ = daily_bin(day, c1 - c0, w, min_obs, days)
        B = injection.basis(planets, time, t0)
        B_d = np.column_stack([daily_bin(day, np.where(np.isfinite(c1 - c0), B[:, j], np.nan),
                                         w, min_obs, days)[1] for j in range(B.shape[1])])
        _, s_d, _ = daily_bin(day, np.where(np.isfinite(c1 - c0), s, np.nan), w, min_obs, days)
        if np.isfinite(d_d).sum() < 2 * B.shape[1] + 5:
            for p in planets:
                bands.setdefault(str(p["band"]), {"band_days": p["band_days"], "transfer": [],
                                                  "quadrature": [], "absolute": []})
                bands[str(p["band"])]["transfer"].append(0.0)
                bands[str(p["band"])]["quadrature"].append(0.0)
            continue
        coef, _ = _lsq(B_d, d_d)
        coef_abs, _ = _lsq(B_d, c1_d)
        for j, p in enumerate(planets):
            b = bands.setdefault(str(p["band"]), {"band_days": p["band_days"], "transfer": [],
                                                  "quadrature": [], "absolute": []})
            b["transfer"].append(float(coef[2 * j] / p["k"]))
            b["quadrature"].append(float(coef[2 * j + 1] / p["k"]))
            b["absolute"].append(float(coef_abs[2 * j] / p["k"]))
        truth_rms.append(_rms(c1_d - s_d))
        ok = np.isfinite(d_d) & np.isfinite(s_d)
        distortion.append(float(np.std(d_d[ok] - s_d[ok]) / max(np.std(s_d[ok]), 1e-9)))

    tr = [t for b in bands.values() for t in b["transfer"]]
    qd = [abs(q) for b in bands.values() for q in b["quadrature"]]
    lo, hi = gates["transfer_range"]
    res["bands"] = bands
    res["min_transfer"] = float(min(tr)) if tr else float("nan")
    res["max_transfer"] = float(max(tr)) if tr else float("nan")
    res["max_abs_quadrature"] = float(max(qd)) if qd else float("nan")
    res["truth_rms_daily"] = float(np.mean(truth_rms)) if truth_rms else float("nan")
    res["distortion"] = float(np.max(distortion)) if distortion else float("nan")
    res["valid_fraction"] = float(min(valid_frac))
    res["day_fraction"] = float(min(day_frac))
    reasons = []
    if not tr:
        reasons.append("no injection run completed (or no planet period fits this baseline)")
    else:
        if min(tr) < lo or max(tr) > hi:
            reasons.append(f"planet amplitude transfer outside [{lo}, {hi}]")
        if max(qd) > gates["max_quadrature"]:
            reasons.append("planet phase distorted (quadrature leakage)")
    if res["valid_fraction"] < gates["min_valid_fraction"]:
        reasons.append("too many non-finite outputs")
    if res["day_fraction"] < gates["min_valid_fraction"]:
        reasons.append("too many observing days without output")
    if not np.isfinite(res["rms_null_daily"]):
        reasons.append("no finite corrected RVs")
    res["gate_pass"] = not reasons
    res["gate_reasons"] = reasons
    mt = res["min_transfer"]
    if tr and mt > 0.05 and np.isfinite(res["rms_null_daily"]):
        res["effective_rms"] = float(res["rms_null_daily"] / min(1.0, mt))
    else:
        res["effective_rms"] = float("inf")
    # White-noise estimate of the semi-amplitude a periodogram would detect at 5 sigma:
    # sigma_K = rms * sqrt(2 / N). Red residual noise makes the true number larger.
    n_ep = int(np.isfinite(c0_d).sum())
    res["n_epochs"] = n_ep
    res["k_detect_5sigma"] = float(5.0 * res["effective_rms"] * np.sqrt(2.0 / max(n_ep, 1)))
    res["improvement_over_raw"] = float(res["rms_raw_daily"] / res["effective_rms"]) \
        if np.isfinite(res["effective_rms"]) and res["effective_rms"] > 0 else 0.0
    res["meets_target"] = bool(res["gate_pass"] and res["effective_rms"] < gates["target_rms_ms"])
    res["status"] = "ok"
    return res

"""On which timescales the residuals vary: structure function and periodicities."""
import numpy as np

from harness.residual_diagnostics import periodogram, sine_amplitudes

TITLE = "Where in time the residual variation sits"
ABOUT = ("First table: the structure function, sqrt(mean squared difference / 2) over pairs of epochs "
         "separated by the given lag. White noise gives the same value at every lag; a value rising with lag "
         "means slow (red) structure, and the lag where it rises tells the timescale. Second: the strongest "
         "periodicities left in the corrected RVs. Third: the largest best-fitting sinusoid in bands where "
         "solar rotation and seasonal effects are expected.")

LAGS = [(0.5, 2), (2, 5), (5, 15), (15, 40), (40, 120), (120, 400), (400, 2000)]
BANDS = [("solar rotation, 23-32 d", 23.0, 32.0), ("half a year, 170-195 d", 170.0, 195.0),
         ("one year, 340-390 d", 340.0, 390.0)]


def compute(ep):
    t, c, p = ep.t, ep.corrected, ep.pipeline
    i, j = np.triu_indices(ep.n, 1)
    lag = np.abs(t[j] - t[i])
    dc, dp = c[j] - c[i], p[j] - p[i]
    ok = np.isfinite(dc) & np.isfinite(dp)
    sf = []
    for lo, hi in LAGS:
        m = ok & (lag >= lo) & (lag < hi)
        k = int(m.sum())
        sf.append([f"{lo:g}-{hi:g} d", k,
                   float(np.sqrt(np.mean(dc[m] ** 2) / 2)) if k >= 10 else None,
                   float(np.sqrt(np.mean(dp[m] ** 2) / 2)) if k >= 10 else None])
    span = float(t.max() - t.min()) if ep.n else 0.0
    per, amp = periodogram(t, c, 2.0, max(4.0, min(1000.0, span)))
    peaks = []
    if amp.size > 2:
        loc = np.nonzero((amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:]))[0] + 1
        for k, q in enumerate(loc[np.argsort(amp[loc])[::-1]][:3], 1):
            peaks.append([f"peak {k}", float(per[q]), float(amp[q])])
    bands = []
    for label, lo, hi in BANDS:
        if span < 1.5 * hi:
            bands.append([label, None, None])
            continue
        trial = 1.0 / np.linspace(1.0 / hi, 1.0 / lo, 200)
        bands.append([label, float(np.nanmax(sine_amplitudes(t, c, trial))),
                      float(np.nanmax(sine_amplitudes(t, p, trial)))])
    return [
        {"header": ["lag between epochs", "pairs", "corrected (m/s)", "pipeline (m/s)"], "rows": sf},
        {"header": ["strongest periodicity in the corrected RVs", "period (d)", "semi-amplitude (m/s)"],
         "rows": peaks,
         "note": f"Best-fitting sinusoid per trial period from 2 d to {min(1000.0, span):.0f} d. Gaps in the "
                 "sampling create aliases (near 1 d, 1 synodic month and 1 year)."},
        {"header": ["band", "corrected: largest semi-amplitude (m/s)", "pipeline: largest semi-amplitude (m/s)"],
         "rows": bands,
         "note": "'-' where the time span is too short for the band."},
    ]

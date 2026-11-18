"""How the residuals are distributed, and whether consecutive epochs are alike."""
import numpy as np

from harness.residual_diagnostics import excess_kurtosis, robust_sigma, skewness

TITLE = "Shape of the residual distribution"
ABOUT = ("For Gaussian white noise: skewness 0, excess kurtosis 0, about 0.3 % of epochs beyond 3 robust "
         "sigma, lag-1 autocorrelation 0 and a von Neumann ratio near 2. A few bad epochs show as large "
         "kurtosis and many epochs beyond 3 sigma; slow structure as positive autocorrelation and a von "
         "Neumann ratio well below 2.")


def _stats(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    rs = robust_sigma(y)
    d = y - np.median(y)
    lag1 = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if y.size > 3 and y.std() > 0 else float("nan")
    vn = float(np.mean(np.diff(y) ** 2) / np.var(y)) if y.size > 3 and y.var() > 0 else float("nan")
    return [float(np.std(y)), rs, skewness(y), excess_kurtosis(y),
            float(np.mean(np.abs(d) > 3 * rs)) if rs > 0 else float("nan"),
            float(np.max(np.abs(d)) / rs) if rs > 0 else float("nan"), lag1, vn]


def compute(ep):
    names = ["RMS (m/s)", "robust RMS = 1.4826 x MAD (m/s)", "skewness", "excess kurtosis",
             "fraction of epochs beyond 3 robust sigma", "largest deviation / robust sigma",
             "lag-1 autocorrelation (consecutive epochs)",
             "von Neumann ratio (mean squared successive difference / variance)"]
    c, p = _stats(ep.corrected), _stats(ep.pipeline)
    return {"header": ["statistic", "corrected", "pipeline"],
            "rows": [[n, a, b] for n, a, b in zip(names, c, p)],
            "note": f"{ep.n} epochs."}

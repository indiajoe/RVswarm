"""Correlation of the residuals with observing geometry, atmosphere and line-shape indicators."""
import numpy as np

from harness.residual_diagnostics import pearson, slope, spearman

TITLE = "Correlation with observing conditions and activity indicators"
ABOUT = ("Correlation of the corrected RVs with quantities that a good correction leaves no trace of, and "
         "the least-squares slope. The last column is the same correlation for the pipeline RVs: a "
         "correction should weaken the correlations with activity indicators (FWHM, bisector, Ca II H&K, "
         "H-alpha) without creating correlations with observing geometry (barycentric correction, zenith "
         "angle, airmass, hour angle) or water vapour.")

KEYS = ["barycentric_correction", "solar_zenith", "solar_airmass", "hour_angle", "water_vapour",
        "fwhm", "bisector", "ca_hk", "h_alpha"]


def compute(ep):
    rows = []
    for key in KEYS:
        cv = ep.covariate(key)
        if cv is None:
            continue
        label, x = cv
        ok = np.isfinite(x) & np.isfinite(ep.corrected)
        if ok.sum() < 10 or np.std(x[ok]) == 0:
            continue
        b, se = slope(x, ep.corrected)
        rows.append([label, pearson(x, ep.corrected), spearman(x, ep.corrected), b, se,
                     pearson(x, ep.pipeline)])
    lim = 3.0 / np.sqrt(max(ep.n, 1))
    return {"header": ["quantity", "Pearson r (corrected)", "Spearman rho (corrected)",
                       "slope (m/s per unit)", "slope error", "Pearson r (pipeline)"],
            "rows": rows,
            "note": f"{ep.n} epochs: |r| above {lim:.2f} is about 3 sigma from zero for uncorrelated noise. "
                    "Slowly varying residuals produce larger chance correlations with slowly varying "
                    "quantities (barycentric correction, indicators over the activity cycle)."}

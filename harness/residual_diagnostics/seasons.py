"""Residuals grouped by season and by calendar year."""
import numpy as np

TITLE = "Residuals by season and by year"
ABOUT = ("Mean and RMS of the RVs per meteorological season (all years together) and per calendar year. "
         "A seasonal pattern in the mean points to something that follows the Earth's orbit (barycentric "
         "frame, line widths through the Sun's apparent rotation, tellurics); a drift over the years to the "
         "activity cycle or the instrument.")


def _row(label, m, ep):
    n = int(m.sum())
    if n < 3:
        return [label, n, None, None, None, None]
    c, p = ep.corrected[m], ep.pipeline[m]
    return [label, n, float(np.mean(c)), float(np.std(c)), float(np.mean(p)), float(np.std(p))]


def compute(ep):
    d = ep.day_of_year
    seasons = [("Dec-Feb", (d >= 335) | (d < 60)), ("Mar-May", (d >= 60) & (d < 152)),
               ("Jun-Aug", (d >= 152) & (d < 244)), ("Sep-Nov", (d >= 244) & (d < 335))]
    year = np.floor(ep.decimal_year).astype(int)
    rows = [_row(label, m, ep) for label, m in seasons]
    rows += [_row(str(y), year == y, ep) for y in np.unique(year)]
    return {"header": ["epochs", "n", "corrected mean (m/s)", "corrected RMS (m/s)",
                       "pipeline mean (m/s)", "pipeline RMS (m/s)"],
            "rows": rows,
            "note": "Means are relative to the mean over all scored epochs. Groups with fewer than 3 epochs show '-'."}

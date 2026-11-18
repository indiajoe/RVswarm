"""TEMPLATE for a residual diagnostic. Files starting with "_" are ignored: copy this file
to a new name (for example dome_temperature.py), edit it, and test it with

    bin/py.sh -m harness.residual_diagnostics check

What compute(ep) gets (see Epochs in __init__.py), all per epoch, in time order:
  ep.n                          number of epochs
  ep.corrected, ep.pipeline     RVs in m/s, mean removed
  ep.err                        photon-noise error of the pipeline RV, m/s
  ep.t, ep.jd                   days since the first epoch; BJD_TDB
  ep.decimal_year, ep.day_of_year
  ep.scalar("ENVDTMP")          any header scalar (None if absent)
  ep.activity("CaIIHK")         any DRP activity index (None if absent)
  ep.covariate("fwhm")          (label, values) of a derived quantity: see COVARIATES in __init__.py
Helpers you can import from harness.residual_diagnostics:
  pearson, spearman, slope, robust_sigma, skewness, excess_kurtosis, sine_amplitudes, periodogram

What compute(ep) returns: one table {"header": [...], "rows": [[...], ...], "note": "..."} or a
list of tables. Cells are short text labels or single numbers (None shows as "-"). Return
reduced statistics only, never per-epoch values: the harness enforces a limit on how many
numbers a diagnostic may publish and rounds them.
"""
from harness.residual_diagnostics import pearson

TITLE = "Correlation with the dome temperature"          # section heading in the report
ABOUT = "One or two sentences that tell the agents what the table shows and how to read it."


def compute(ep):
    x = ep.scalar("ENVDTMP")
    if x is None:
        return {"header": ["quantity", "value"], "rows": [["dome temperature", None]],
                "note": "ENVDTMP is not in this data set."}
    return {"header": ["quantity", "Pearson r (corrected)", "Pearson r (pipeline)"],
            "rows": [["dome temperature (ENVDTMP)", pearson(x, ep.corrected), pearson(x, ep.pipeline)]]}

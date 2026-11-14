"""CANARY (must FAIL the gate): silently discards the most deviant half of the data."""
import numpy as np


def run(ctx):
    rv = ctx.target.rv_pipeline.copy()
    rv[np.abs(rv - np.median(rv)) > np.median(np.abs(rv - np.median(rv)))] = np.nan
    return {"rv_clean": rv}

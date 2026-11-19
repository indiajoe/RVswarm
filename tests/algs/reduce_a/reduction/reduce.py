"""Preliminary reduction step shared by the self-test algorithms reduce_a and reduce_b.

Depends on the data alone. Products (one value per observation, in time order):
  order_mean   mean of the per-order DRP RVs over the orders that have one, m/s
  fwhm         CCF FWHM (FWHMMOD), km/s
  bis          CCF bisector span (BISMOD), km/s
  cahk         Ca II H&K index
"""
import numpy as np


def reduce(ctx):
    ds = ctx.target
    ctx.log("reducing", ds.n_obs, "observations")
    return {"order_mean": np.nanmean(ds.order_rv, axis=1),
            "fwhm": ds.scalar("FWHMMOD"), "bis": ds.scalar("BISMOD"), "cahk": ds.activity("CaIIHK")[0]}

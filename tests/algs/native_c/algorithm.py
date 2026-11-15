"""Self-test: indicator decorrelation whose least-squares kernel is written in C."""
import ctypes

import numpy as np


def run(ctx):
    ds = ctx.target
    lib = ctx.native_library(openmp=True)
    vec = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.decorrelate.argtypes = [vec, vec, vec, ctypes.c_int]
    lib.decorrelate.restype = ctypes.c_double
    y = np.ascontiguousarray(ds.rv_pipeline, dtype=np.float64)
    x = np.ascontiguousarray(ds.scalar("FWHMMOD"), dtype=np.float64)
    out = np.empty_like(y)
    slope = lib.decorrelate(y, x, out, y.size)
    ctx.metric("c_slope", slope)
    return {"rv_clean": out}

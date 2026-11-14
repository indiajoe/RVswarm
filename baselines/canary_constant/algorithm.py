"""CANARY (must FAIL the gate): returns a constant. Zero scatter, zero planets."""
import numpy as np


def run(ctx):
    return {"rv_clean": np.zeros(ctx.target.n_obs)}

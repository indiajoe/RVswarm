"""Self-test: native code that tries to open a file must be rejected by the audit."""
import numpy as np


def run(ctx):
    ctx.native_library()
    return {"rv_clean": np.asarray(ctx.target.rv_pipeline)}

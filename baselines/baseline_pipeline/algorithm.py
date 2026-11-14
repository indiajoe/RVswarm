"""Reference: the NEID pipeline RV, untouched. Defines the 'raw' scatter and transfer = 1."""


def run(ctx):
    return {"rv_clean": ctx.target.rv_pipeline.copy()}

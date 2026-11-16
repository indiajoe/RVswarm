"""Self-test: returns the pipeline RV unchanged, so the pre-check must skip the injections."""


def run(ctx):
    return {"rv_clean": ctx.target.rv_pipeline.copy()}

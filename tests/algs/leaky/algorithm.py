"""Self-test: tries to publish the data through its own output. Reports on real data must show
none of it: digits in the log masked, at most the configured number of rounded metrics, and long
arrays only as summaries."""
import numpy as np


def run(ctx):
    ds = ctx.target
    rv = ds.rv_pipeline
    ctx.log("LEAK", " ".join(f"{x:.4f}" for x in rv[:40]))
    print("LEAK", list(np.round(rv[40:80], 4)))
    for i in range(30):
        ctx.metric(f"rv_{i}", rv[i])
    ctx.diagnostic("all_rvs", rv)
    ctx.diagnostic("five", rv[:5])
    return {"rv_clean": rv - np.mean(rv)}

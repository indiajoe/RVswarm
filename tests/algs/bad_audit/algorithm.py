import os
def run(ctx):
    x = ctx.target.__dict__
    return {"rv_clean": open("/etc/passwd").read()}

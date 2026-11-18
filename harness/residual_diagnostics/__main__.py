"""Try the residual diagnostics before an evaluation uses them.

    bin/py.sh -m harness.residual_diagnostics check

Runs every diagnostic in harness/residual_diagnostics/ on the synthetic Sun (with a crude
stand-in for a corrected RV series), prints the report section exactly as the agents will
see it, then any errors with their tracebacks. Exit status 1 if a diagnostic failed.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .. import backends, config
from . import Epochs, print_errors, run_all, to_markdown


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check"])
    ap.parse_args(argv)
    cfg = config.load()
    be = backends.open_backend({"kind": "synth", "seed": 3, "n_days": 900, "obs_per_day": 1,
                                "day_keep": 0.33, "with_spectra": False})
    idx = np.argsort(be.time, kind="stable")
    X = np.column_stack([be.scalars[idx, list(be.scal_names).index("FWHMMOD")], np.ones(idx.size)])
    rv_clean = be.rv[idx] - X @ np.linalg.lstsq(X, be.rv[idx], rcond=None)[0]    # stand-in correction
    ep = Epochs(be, idx, rv_clean, min_obs=cfg["data"].get("min_obs_per_day", 1))
    errors = []
    entries = run_all(ep, cfg, on_error=lambda n, e: errors.append((n, e)))
    print(to_markdown(entries))
    for n, e in errors:
        print_errors(n, e)
    for e in entries:
        print(f"{'FAILED' if e.get('error') else 'ok':7s}{e['name']:24s}{e['seconds']:.2f} s")
    print(f"{ep.n} synthetic epochs; disabled in config: {(cfg.get('residual_diagnostics') or {}).get('disabled') or 'none'}")
    sys.exit(1 if any(e.get("error") for e in entries) else 0)


if __name__ == "__main__":
    main()

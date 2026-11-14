"""Hidden planetary signals. Parameters never leave the harness process tree."""
from __future__ import annotations

import numpy as np


def draw_planets(seed: int, baseline_days: float, cfg: dict) -> list[dict]:
    """One circular-orbit planet per period band (periods clipped to the baseline)."""
    rng = np.random.default_rng(seed)
    pmax = cfg["max_period_fraction_of_baseline"] * baseline_days
    kmin, kmax = cfg["k_range_ms"]
    planets = []
    for b, (lo, hi) in enumerate(cfg["bands_days"]):
        hi = min(hi, pmax)
        if hi <= lo * 1.2:                       # band does not fit in this baseline
            continue
        planets.append({
            "band": b,
            "band_days": [lo, hi],
            "period": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
            "k": float(np.exp(rng.uniform(np.log(kmin), np.log(kmax)))),
            "phase": float(rng.uniform(0, 2 * np.pi)),
        })
    return planets


def signal(planets: list[dict], t: np.ndarray, t0: float) -> np.ndarray:
    """Total injected RV (m/s) at times t."""
    v = np.zeros_like(t, dtype=np.float64)
    for p in planets:
        v += p["k"] * np.sin(2 * np.pi * (t - t0) / p["period"] + p["phase"])
    return v


def basis(planets: list[dict], t: np.ndarray, t0: float) -> np.ndarray:
    """Columns: in-phase (unit amplitude) and quadrature for each planet -> (n, 2*n_planets)."""
    cols = []
    for p in planets:
        arg = 2 * np.pi * (t - t0) / p["period"] + p["phase"]
        cols += [np.sin(arg), np.cos(arg)]
    return np.stack(cols, axis=1) if cols else np.zeros((t.size, 0))

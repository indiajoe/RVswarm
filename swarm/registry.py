"""The explicit registry of approach families, proposals and algorithms.

Stored as JSON (workspace/registry/registry.json) and rendered to Markdown for
humans and for the agents that are allowed to see the whole picture.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .tools import WORKSPACE

REG_FILE = WORKSPACE / "registry" / "registry.json"

SEED_FAMILIES = {
    "F01": ("CCF / line-profile shape decomposition",
            "Separate shape changes from shifts in the cross-correlation or mean line profile: SCALPELS, "
            "FIESTA, CCF PCA, bisector and moment analyses."),
    "F02": ("Line-by-line and line-selection",
            "Per-line RVs; weighting, grouping or modelling lines by formation depth, temperature, "
            "magnetic sensitivity, blending or empirical activity response."),
    "F03": ("Activity-indicator and Gaussian-process time-series models",
            "Decorrelation against indicators, FF'-type models, multi-dimensional or latent GPs tying RVs "
            "to indicators in time."),
    "F04": ("Supervised machine learning on CCFs or spectra",
            "Networks or other learners trained to predict the activity RV from shift-invariant inputs."),
    "F05": ("Pixel-level data-driven spectral models",
            "Template + residual models on the spectra themselves: Doppler-constrained PCA, wobble-like "
            "factorisations, autoencoders with an explicit Doppler layer."),
    "F06": ("Physics-forward models of the active Sun",
            "Explicit physical parametrisation (magnetic filling factors, convective-blueshift suppression, "
            "spot/facula contrasts, Zeeman broadening) inferred from the spectra and forward-modelled into RV."),
    "F07": ("Granulation, supergranulation and oscillation modelling",
            "Stochastic or physical models of convective and oscillatory signals; optimal binning and "
            "filtering that provably preserves planets."),
    "F08": ("Instrumental, telluric and atmospheric systematics",
            "Detector, wavelength-calibration, drift, telluric, differential-extinction and frame "
            "systematics modelled from spectra and telemetry."),
    "F09": ("Chromatic and order-dependent RV structure",
            "Use of the wavelength dependence of activity signals versus the achromatic planet signal."),
    "F10": ("Hybrids and ensembles", "Combinations of two or more families, stacking, model averaging."),
}


def load() -> dict:
    if REG_FILE.exists():
        return json.loads(REG_FILE.read_text())
    return {"families": {k: {"name": n, "description": d, "origin": "seed"}
                         for k, (n, d) in SEED_FAMILIES.items()},
            "proposals": {}, "redirects": {}, "assignments": {}, "notes": []}


def save(reg: dict) -> None:
    REG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = REG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1, default=float))
    tmp.rename(REG_FILE)
    (REG_FILE.parent / "REGISTRY.md").write_text(render(reg))


def add_family(reg: dict, name: str, description: str, cycle: int) -> str:
    n = 1 + max(int(k[1:]) for k in reg["families"])
    fid = f"F{n:02d}"
    reg["families"][fid] = {"name": name, "description": description, "origin": f"cycle {cycle}"}
    return fid


def family_stats(reg: dict, last_cycles: int | None = None, now: int | None = None) -> dict:
    st = {f: {"proposed": 0, "selected": 0, "evaluated": 0, "best_eff": None, "best_alg": None,
              "gate_pass": 0} for f in reg["families"]}
    for pid, p in reg["proposals"].items():
        f = p.get("family")
        if f not in st:
            continue
        if last_cycles and now is not None and p["cycle"] <= now - last_cycles:
            continue
        st[f]["proposed"] += 1
        st[f]["selected"] += bool(p.get("selected"))
        best = best_result(p)
        if best:
            st[f]["evaluated"] += 1
            st[f]["gate_pass"] += bool(best.get("gate_pass"))
            e = best.get("effective_rms")
            if best.get("gate_pass") and e is not None and math.isfinite(e) and \
                    (st[f]["best_eff"] is None or e < st[f]["best_eff"]):
                st[f]["best_eff"], st[f]["best_alg"] = e, pid
    return st


def best_result(p: dict) -> dict | None:
    """Most advanced stage result of a proposal (sealed > full > screen)."""
    for stage in ("sealed", "full", "screen"):
        r = (p.get("results") or {}).get(stage)
        if r:
            return {**r, "stage": stage}
    return None


def compute_redirects(reg: dict, cycle: int, slots: list[dict], share: float, window: int) -> dict:
    """If recent proposals crowd into one family, send some of its slots to the least explored ones."""
    recent = [p for p in reg["proposals"].values()
              if p["cycle"] > cycle - window and p.get("kind") == "idea" and p.get("family")]
    if len(recent) < 6:
        return {}
    counts = Counter(p["family"] for p in recent)
    total = sum(counts.values())
    alltime = Counter(p["family"] for p in reg["proposals"].values() if p.get("kind") == "idea")
    quiet = sorted(reg["families"], key=lambda f: (alltime.get(f, 0), f))
    quiet = [f for f in quiet if f != "F10"]
    out, qi = {}, 0
    for fam, n in counts.most_common():
        if n / total <= share:
            continue
        by_slot = Counter(p["slot"] for p in recent if p["family"] == fam)
        movers = [s for s, _ in by_slot.most_common()][1::2] or [by_slot.most_common()[-1][0]]
        for s in movers:                                  # every second contributor keeps going
            while qi < len(quiet) and quiet[qi] == fam:
                qi += 1
            if qi >= len(quiet):
                break
            out[s] = {"family": quiet[qi], "from_family": fam, "cycle": cycle + 1}
            qi += 1
    return out


def _f(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def render(reg: dict, with_results: bool = True) -> str:
    st = family_stats(reg)
    L = ["# Registry of approach families and algorithms", "",
         "_Generated file. Source of truth: registry.json._", "", "## Families", "",
         "| id | family | proposed | selected | evaluated | passed gate | best effective RMS (m/s) | best algorithm |",
         "|---|---|---|---|---|---|---|---|"]
    for f, v in reg["families"].items():
        s = st[f]
        L.append(f"| {f} | {v['name']} | {s['proposed']} | {s['selected']} | {s['evaluated']} | "
                 f"{s['gate_pass']} | {_f(s['best_eff'], 4) if with_results else 'n/a'} | "
                 f"{(s['best_alg'] or '-') if with_results else 'n/a'} |")
    L += ["", "## Family descriptions", ""]
    L += [f"- **{f} {v['name']}** ({v['origin']}): {v['description']}" for f, v in reg["families"].items()]
    L += ["", "## Proposals and algorithms", "",
          "| id | cycle | slot | family | title | status | stage | RMS | min transfer | effective RMS | gate |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pid, p in sorted(reg["proposals"].items()):
        b = best_result(p) or {}
        L.append(f"| {pid} | {p['cycle']} | {p.get('slot', '-')} | {p.get('family') or '?'} | "
                 f"{p.get('title', '')[:70]} | {p.get('status', '')} | {b.get('stage', '-')} | "
                 f"{_f(b.get('rms_null_daily'), 4)} | {_f(b.get('min_transfer'))} | "
                 f"{_f(b.get('effective_rms'), 4)} | "
                 f"{'-' if not b else ('PASS' if b.get('gate_pass') else 'fail')} |")
    if reg.get("redirects"):
        L += ["", "## Active redirects", ""]
        L += [f"- slot {s} -> {r['family']} (moved away from {r['from_family']}) for cycle {r['cycle']}"
              for s, r in reg["redirects"].items()]
    return "\n".join(L) + "\n"

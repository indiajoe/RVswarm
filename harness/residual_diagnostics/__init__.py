"""Residual diagnostics: reduced statistics of an algorithm's corrected RVs, computed by the
harness and printed in the result reports.

WHAT THIS IS FOR
Every report on development data says how much the corrected RVs scatter. These
diagnostics add the STRUCTURE of that scatter: how it is distributed, what it correlates
with, on which timescales it varies, how it differs between seasons and years. They never
show the RVs themselves, so the data cannot be reconstructed from the reports.

HOW TO ADD ONE (humans only: agents can neither read nor change this directory)
Copy `_template.py` to a new file in this directory (a name without a leading "_"), fill
in TITLE, ABOUT and compute(), then run

    bin/py.sh -m harness.residual_diagnostics check

which runs every diagnostic on the synthetic Sun and prints the report section exactly as
the agents will see it, or the error with its traceback. Commit the file; every later
evaluation includes it (in each swarm after `git pull`). To switch one off without deleting
it, list its file name under `residual_diagnostics.disabled` in config/harness.yaml.

RULES, enforced here
* Only the un-injected run is used (it carries no hidden planets), on the scored epochs.
* Published for development-data stages (screen, full). For sealed runs the tables are
  written to state/sealed_diagnostics/ for the humans and never published. Not computed for
  the implementers' local toy test.
* compute() returns tables. A cell is a short label or a number; at most `max_numbers`
  numbers per diagnostic (config/harness.yaml), each float rounded to
  `significant_digits`. Never return per-epoch values.
* A diagnostic that raises is shown as "not available (ExceptionType)" without its
  message (a message could quote data values); the evaluation goes on.
"""
from __future__ import annotations

import importlib.util
import math
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from .. import score

C_MS = 299_792_458.0
KITT_PEAK_LON_DEG = -111.600          # WIYN / NEID, east positive
HERE = Path(__file__).resolve().parent
MAX_ROWS = 40
MAX_LABEL = 80


# ------------------------------------------------------------------------------------- data
class Epochs:
    """One evaluation's un-injected run, binned per epoch (observing day) on the scored days.

    Attributes, NumPy arrays of length `n`, in time order:
      jd             BJD_TDB of the epoch (photon-weighted mean time of its spectra), days
      t              days since the first epoch
      decimal_year   for example 2022.37 (approximate calendar)
      day_of_year    1 ... 366 (approximate)
      corrected      the algorithm's rv_clean, photon-weighted epoch mean, mean removed (m/s)
      pipeline       the DRP's CCF RV (CCFRVMOD) on the same epochs, mean removed (m/s)
      err            photon-noise uncertainty of the pipeline RV per epoch (m/s)
    Methods (each returns an epoch array, or None when the quantity is not in the data):
      covariate(key) -> (label, array) for the derived quantities in COVARIATES
      scalar(name)   -> any header scalar: FWHMMOD, BISMOD, ZENITH, WVAPOR, CCDTEMP, ENV* ...
      activity(name) -> any DRP activity index: CaIIHK, Ha06_1, NaI, CaIRT1 ...
    """

    def __init__(self, backend, index, rv_clean, days=None, min_obs: int = 1):
        be, idx = backend, np.asarray(index, dtype=np.int64)
        c0 = np.asarray(rv_clean, dtype=float)
        self._be, self._idx, self._min_obs = be, idx, int(min_obs)
        self._day = np.asarray(be.day[idx])
        self._w = 1.0 / np.clip(np.asarray(be.rv_err[idx], float), 1e-3, None) ** 2
        self._ok = np.isfinite(c0) & np.isfinite(be.rv[idx])
        days = np.unique(self._day) if days is None else np.asarray(days)
        _, cd, _ = score.daily_bin(self._day, np.where(self._ok, c0, np.nan), self._w, self._min_obs, days)
        keep = np.isfinite(cd)
        self._days = days[keep]
        self.n = int(self._days.size)
        pos = np.searchsorted(self._days, self._day)
        pos_c = np.clip(pos, 0, max(self.n - 1, 0))
        inside = (pos < self.n) & self._ok
        if self.n:
            inside &= self._days[pos_c] == self._day
        wsum = np.bincount(pos_c[inside], weights=self._w[inside], minlength=self.n) if self.n else np.zeros(0)
        corr, pipe = cd[keep], self._bin(be.rv[idx])
        wm = wsum if self.n and wsum.sum() > 0 else np.ones(self.n)
        self.corrected = corr - (np.average(corr, weights=wm) if self.n else 0.0)
        self.pipeline = pipe - (np.average(pipe, weights=wm) if self.n else 0.0)
        self.err = 1.0 / np.sqrt(np.clip(wsum, 1e-30, None))
        self.jd = self._bin(be.time[idx])
        self.t = self.jd - (self.jd[0] if self.n else 0.0)
        self.decimal_year = 2000.0 + (self.jd - 2451545.0) / 365.25
        self.day_of_year = np.floor((self.jd - 2451544.5) % 365.25) + 1

    def _bin(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return score.daily_bin(self._day, np.where(self._ok, x, np.nan), self._w, self._min_obs, self._days)[1]

    def scalar(self, name: str):
        names = list(self._be.scal_names)
        if name not in names:
            return None
        return self._bin(self._be.scalars[self._idx, names.index(name)])

    def activity(self, name: str):
        names = list(self._be.act_names)
        if name not in names:
            return None
        return self._bin(self._be.act_val[self._idx, names.index(name)])

    def covariate(self, key: str):
        if key not in COVARIATES:
            raise KeyError(f"unknown covariate {key!r}; known: {', '.join(COVARIATES)}")
        label, fn = COVARIATES[key]
        x = fn(self)
        return None if x is None else (label, x)

    # per-observation helpers for COVARIATES
    def _obs_scalar(self, name: str):
        names = list(self._be.scal_names)
        return None if name not in names else np.asarray(self._be.scalars[self._idx, names.index(name)], float)


def _barycentric(ep: Epochs):
    ssbz = getattr(ep._be, "ssbz", None)
    if ssbz is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = C_MS * np.nanmedian(np.asarray(ssbz[ep._idx], float), axis=1) / 1e3
    return ep._bin(v)


def _zenith(ep: Epochs):
    return ep.scalar("ZENITH")


def _airmass(ep: Epochs):
    z = ep._obs_scalar("ZENITH")
    if z is None:
        return None
    z = np.where((z >= 0) & (z < 90), z, np.nan)
    # Kasten & Young (1989, Applied Optics 28, 4735)
    x = 1.0 / (np.cos(np.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)
    return ep._bin(x)


def _hour_angle(ep: Epochs):
    """Hour angle of the Sun at Kitt Peak, hours from local apparent noon (about +-10 min:
    BJD is used as if it were UT, with an approximate equation of time)."""
    jd = np.asarray(ep._be.time[ep._idx], float)
    ut_h = ((jd + 0.5) % 1.0) * 24.0
    n = (jd - 2451544.5) % 365.25
    b = 2 * np.pi * (n - 81.0) / 364.0
    eot_min = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    h = ut_h + KITT_PEAK_LON_DEG / 15.0 + eot_min / 60.0 - 12.0
    return ep._bin((h + 12.0) % 24.0 - 12.0)


def _act(name):
    return lambda ep: ep.activity(name)


def _sc(name):
    return lambda ep: ep.scalar(name)


COVARIATES = {
    "barycentric_correction": ("barycentric correction c*ssb_z (km/s)", _barycentric),
    "solar_zenith": ("solar zenith angle (ZENITH, deg)", _zenith),
    "solar_airmass": ("solar airmass (from ZENITH)", _airmass),
    "hour_angle": ("hour angle of the Sun (h from local noon)", _hour_angle),
    "water_vapour": ("water vapour (WVAPOR)", _sc("WVAPOR")),
    "fwhm": ("CCF FWHM (FWHMMOD, km/s)", _sc("FWHMMOD")),
    "bisector": ("CCF bisector span (BISMOD, km/s)", _sc("BISMOD")),
    "ca_hk": ("Ca II H&K index (CaIIHK)", _act("CaIIHK")),
    "h_alpha": ("H-alpha index (Ha06_1)", _act("Ha06_1")),
}


# ------------------------------------------------------------------------- statistics helpers
def _pair(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def pearson(x, y) -> float:
    x, y = _pair(x, y)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(a.size)
    r[order] = np.arange(a.size)
    return r


def spearman(x, y) -> float:
    x, y = _pair(x, y)
    return pearson(_rank(x), _rank(y)) if x.size >= 3 else float("nan")


def slope(x, y) -> tuple[float, float]:
    """Least-squares slope of y on x and its standard error."""
    x, y = _pair(x, y)
    if x.size < 4 or x.std() == 0:
        return float("nan"), float("nan")
    xm = x - x.mean()
    b = float(np.sum(xm * (y - y.mean())) / np.sum(xm * xm))
    res = y - y.mean() - b * xm
    se = float(np.sqrt(np.sum(res * res) / (x.size - 2) / np.sum(xm * xm)))
    return b, se


def robust_sigma(y) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    return float(1.4826 * np.median(np.abs(y - np.median(y)))) if y.size else float("nan")


def skewness(y) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    s = y.std()
    return float(np.mean((y - y.mean()) ** 3) / s ** 3) if y.size > 2 and s > 0 else float("nan")


def excess_kurtosis(y) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    s = y.std()
    return float(np.mean((y - y.mean()) ** 4) / s ** 4 - 3.0) if y.size > 3 and s > 0 else float("nan")


def sine_amplitudes(t, y, periods) -> np.ndarray:
    """Semi-amplitude of the least-squares sinusoid (with a free mean) at each trial period."""
    t, y = _pair(t, y)
    periods = np.atleast_1d(np.asarray(periods, float))
    if t.size < 5:
        return np.full(periods.size, np.nan)
    out = np.empty(periods.size)
    for a in range(0, periods.size, 2000):                    # bounded memory
        w = 2 * np.pi / periods[a:a + 2000, None]
        c, s = np.cos(w * t[None, :]), np.sin(w * t[None, :])
        n = t.size
        A = np.empty((c.shape[0], 3, 3))
        A[:, 0, 0] = n
        A[:, 0, 1] = A[:, 1, 0] = c.sum(1)
        A[:, 0, 2] = A[:, 2, 0] = s.sum(1)
        A[:, 1, 1] = (c * c).sum(1)
        A[:, 2, 2] = (s * s).sum(1)
        A[:, 1, 2] = A[:, 2, 1] = (c * s).sum(1)
        r = np.stack([np.full(c.shape[0], y.sum()), c @ y, s @ y], axis=1)
        A += np.eye(3)[None] * 1e-9 * n
        coef = np.linalg.solve(A, r[..., None])[..., 0]
        out[a:a + 2000] = np.hypot(coef[:, 1], coef[:, 2])
    return out


def periodogram(t, y, pmin: float, pmax: float, oversample: float = 5.0, max_freq: int = 20000):
    """(periods, semi-amplitudes) on a uniform frequency grid between 1/pmax and 1/pmin."""
    t, _ = _pair(t, y)
    span = float(t.max() - t.min()) if t.size else 0.0
    if span <= 0 or pmax <= pmin:
        return np.zeros(0), np.zeros(0)
    df = 1.0 / (oversample * span)
    nf = int(min(max_freq, max(10, (1.0 / pmin - 1.0 / pmax) / df)))
    f = np.linspace(1.0 / pmax, 1.0 / pmin, nf)
    return 1.0 / f, sine_amplitudes(t, y, 1.0 / f)


# ------------------------------------------------------------------------------- framework
def discover(disabled=()) -> list[tuple[str, object, str | None]]:
    """(name, module or None, load error) for every diagnostic file in this directory."""
    out = []
    for p in sorted(HERE.glob("*.py")):
        if p.name.startswith("_") or p.stem in set(disabled or ()):
            continue
        spec = importlib.util.spec_from_file_location(f"rvswarm_residual_diag_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            if not callable(getattr(mod, "compute", None)):
                raise TypeError("the file defines no compute(ep) function")
        except Exception as e:                                   # noqa: BLE001 -- reported, never fatal
            out.append((p.stem, None, f"{type(e).__name__}: {e}"))
            continue
        out.append((p.stem, mod, None))
    return out


def _is_number(c) -> bool:
    return isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, (bool, np.bool_))


def _fmt(c, sd: int) -> str:
    if c is None:
        return "-"
    if _is_number(c):
        if isinstance(c, (int, np.integer)):
            return str(int(c))
        x = float(c)
        return "-" if not math.isfinite(x) else f"{x:.{sd}g}"
    return str(c).replace("|", "/").replace("\n", " ")[:MAX_LABEL]


def validate(res, max_numbers: int, sd: int) -> tuple[list[dict], bool]:
    """Check a compute() result and format its cells. -> (tables, cut off at the number limit)."""
    tables = res if isinstance(res, list) else [res]
    out, used, cut = [], 0, False
    for tb in tables:
        if not isinstance(tb, dict) or "header" not in tb or "rows" not in tb:
            raise TypeError("compute() must return {'header': [...], 'rows': [[...], ...]} or a list of those")
        header = [_fmt(h, sd) if not isinstance(h, str) else h.replace("|", "/")[:MAX_LABEL] for h in tb["header"]]
        rows = []
        for row in list(tb["rows"])[:MAX_ROWS]:
            row = list(row)
            if len(row) != len(header):
                raise ValueError(f"a row has {len(row)} cells but the header has {len(header)}")
            for c in row:
                if not (c is None or _is_number(c) or isinstance(c, (str, bool, np.bool_))):
                    raise TypeError(f"a cell is a {type(c).__name__}; cells must be text or single numbers")
            nums = sum(_is_number(c) for c in row)
            if used + nums > max_numbers:
                cut = True
                break
            used += nums
            rows.append([_fmt(c, sd) for c in row])
        out.append({"header": header, "rows": rows, "note": str(tb.get("note") or "")[:600]})
        if cut:
            break
    return out, cut


def run_all(ep: Epochs, cfg: dict, on_error=None) -> list[dict]:
    """Run every enabled diagnostic. on_error(name, exception) is called with the full error
    (used by the `check` command); the report only names the exception type."""
    rc = cfg.get("residual_diagnostics") or {}
    max_numbers, sd = int(rc.get("max_numbers", 60)), int(rc.get("significant_digits", 3))
    out = []
    for name, mod, err in discover(rc.get("disabled") or []):
        e = {"name": name, "title": str(getattr(mod, "TITLE", name)), "about": str(getattr(mod, "ABOUT", ""))}
        t0 = time.time()
        if err:
            e["error"] = "not available (the diagnostic could not be loaded)"
            if on_error:
                on_error(name, err)
        elif ep.n < 5:
            e["error"] = f"not available (only {ep.n} epochs)"
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    e["tables"], e["cut"] = validate(mod.compute(ep), max_numbers, sd)
            except Exception as ex:                              # noqa: BLE001
                e["error"] = f"not available ({type(ex).__name__})"
                if on_error:
                    on_error(name, ex)
        e["seconds"] = round(time.time() - t0, 3)
        out.append(e)
    return out


def to_markdown(entries: list[dict]) -> str:
    L = ["## Residual diagnostics", "",
         "Computed by the harness from the un-injected run over the scored epochs (one value per observing "
         "day). 'corrected' is your `rv_clean`, 'pipeline' the DRP's CCF RV (CCFRVMOD), both with their mean "
         "removed. Only reduced, rounded statistics are shown; the RVs themselves are not published. The "
         "human scientists maintain these diagnostics: ask for another with a `REQUEST TO HUMANS:` line in a "
         "post-mortem.", ""]
    for e in entries:
        L += [f"### {e['title']}", ""]
        if e.get("about"):
            L += [e["about"], ""]
        if e.get("error"):
            L += [f"_{e['error']}_", ""]
            continue
        for tb in e.get("tables", []):
            L.append("| " + " | ".join(tb["header"]) + " |")
            L.append("|" + "---|" * len(tb["header"]))
            L += ["| " + " | ".join(r) + " |" for r in tb["rows"]]
            if tb.get("note"):
                L += ["", tb["note"]]
            L.append("")
        if e.get("cut"):
            L += ["_(cut off at the harness limit on numbers per diagnostic)_", ""]
    return "\n".join(L).rstrip() + "\n"


def print_errors(name: str, err) -> None:
    print(f"\n!!! diagnostic `{name}` failed:")
    if isinstance(err, BaseException):
        traceback.print_exception(type(err), err, err.__traceback__)
    else:
        print(err)

"""Reading NEID L2 FITS files.

Only this module touches raw FITS. Everything an algorithm may learn about an
observation passes through the explicit allow-lists below, so that no quantity
that would be inconsistent with a blind injection can leak.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from astropy.io import fits

N_ORDERS = 122
N_PIX = 9216
ORDER_MAX = 173            # array index i  <->  physical echelle order ORDER_MAX - i
ORDER_KEYS = [f"{ORDER_MAX - i:03d}" for i in range(N_ORDERS)]

# Primary-header numeric scalars exposed to algorithms (none carries stellar RV).
PRIMARY_SCALARS = [
    "OBSJD", "EXPTIME", "AIRMASS", "SUNAGL", "MOONAGL", "TCSZA", "TCSEL", "TCSAZ",
    "ENVDTMP", "ENVOTMP", "ENVDHUM", "ENVOHUM", "ENVWINDS", "ENVWINDD", "DEWPOINT",
    "LS218_1", "LS218_2", "LS218_3", "LS218_4", "LS218_5", "LS218_6", "LS218_7",
    "LS218_8", "PMKSTOP", "PMKSBOT", "PNEXTORR", "CCDTEMP", "MOLYTEMP", "SLUGTEMP",
    "CCDHEAT", "SLUGHEAT", "EXTSNR", "DRIFTRV0", "DRIFTRV1", "DRIFTRV2",
    "DQLEVEL1", "DQLEVEL2", "NCOMBINE",
]
PRIMARY_PREFIXES = ("TMCV", "TMCT")       # thermal-control telemetry
# CCFS-header scalars with no RV content (RV-bearing ones are handled explicitly).
CCF_SCALARS = [
    "DVRMSMOD", "DVRMSSUM", "FWHMMOD", "EFWHMMOD", "BISMOD", "EBISMOD", "FWHMSUM",
    "EFWHMSUM", "BISSUM", "EBISSUM", "FITMODV0", "FITMODV2", "FITMODV3",
    "FITSUMV0", "FITSUMV2", "FITSUMV3", "MAXCPP",
]
TELLURIC_SCALARS = ["ZENITH", "WVAPOR"]

_FN_RE = re.compile(r"neidL2_(\d{8})T(\d{6})")


def obs_id_from_path(path) -> str:
    return Path(path).name.replace(".fits", "")


def jd_from_filename(path) -> float:
    """Approximate UTC JD of exposure start from the file name (used for sharding)."""
    m = _FN_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"not a NEID L2 file name: {path}")
    d, t = m.groups()
    y, mo, da = int(d[:4]), int(d[4:6]), int(d[6:8])
    frac = (int(t[:2]) + int(t[2:4]) / 60 + int(t[4:6]) / 3600) / 24
    a = (14 - mo) // 12
    yy, mm = y + 4800 - a, mo + 12 * a - 3
    jdn = da + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    return jdn - 0.5 + frac


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def read_lite(path) -> dict:
    """Everything except the 2-D spectra: scalars, per-order pipeline RVs, CCFs, indices."""
    out: dict = {"obs_id": obs_id_from_path(path)}
    with fits.open(path, memmap=True) as hdul:
        h0 = hdul[0].header
        hc = hdul["CCFS"].header
        sc = {k: _num(h0.get(k)) for k in PRIMARY_SCALARS}
        for k in h0:
            if k.startswith(PRIMARY_PREFIXES):
                sc[k] = _num(h0[k])
        for k in CCF_SCALARS:
            sc[k] = _num(hc.get(k))
        ht = hdul["TELLURIC"].header
        for k in TELLURIC_SCALARS:
            sc[k] = _num(ht.get(k))
        out["scalars"] = sc
        out["object"] = str(h0.get("OBJECT", ""))
        out["sci_obj"] = str(h0.get("SCI-OBJ", ""))
        out["jd"] = _num(hc.get("CCFJDMOD"))                     # BJD_TDB, flux-weighted
        out["rv"] = _num(hc.get("CCFRVMOD")) * 1e3               # m/s
        out["rv_sum"] = _num(hc.get("CCFRVSUM")) * 1e3
        # In combined files DVRMSMOD is the mean of the single-frame values, so the
        # uncertainty of the mean RV is smaller by sqrt(NCOMBINE).
        ncomb = _num(h0.get("NCOMBINE"))
        ncomb = ncomb if np.isfinite(ncomb) and ncomb >= 1 else 1.0
        out["rv_err"] = _num(hc.get("DVRMSMOD")) * 1e3 / np.sqrt(ncomb)
        out["order_rv"] = np.array([_num(hc.get("CCFRV" + k)) for k in ORDER_KEYS]) * 1e3
        out["order_wt"] = np.array([_num(hc.get("CCFWT" + k)) for k in ORDER_KEYS])
        out["ssbz"] = np.array([_num(h0.get("SSBZ" + k)) for k in ORDER_KEYS])
        out["ccf_start"] = _num(hc.get("CCFSTART"))              # km/s
        out["ccf_step"] = _num(hc.get("CCFSTEP"))
        out["ccf"] = np.asarray(hdul["CCFS"].data, dtype=np.float32)
        act = hdul["ACTIVITY"].data
        out["act_names"] = [str(n).strip() for n in act["INDEX"]]
        out["act_val"] = np.asarray(act["VALUE"], dtype=np.float64)
        out["act_unc"] = np.asarray(act["UNCERTAINTY"], dtype=np.float64)
    return out


def read_spectrum(path, orders=None) -> dict:
    """Science-fibre spectrum for selected order indices (observatory-frame vacuum wavelengths)."""
    sel = slice(None) if orders is None else np.asarray(orders, dtype=int)
    with fits.open(path, memmap=True) as hdul:
        out = {
            "wave": np.array(hdul["SCIWAVE"].data[sel], dtype=np.float64),
            "flux": np.array(hdul["SCIFLUX"].data[sel], dtype=np.float32),
            "var": np.array(hdul["SCIVAR"].data[sel], dtype=np.float32),
            "blaze": np.array(hdul["SCIBLAZE"].data[sel], dtype=np.float32),
        }
        tel = hdul["TELLURIC"].data if "TELLURIC" in hdul else None
        if tel is not None:                     # absent in the curated, already telluric-corrected files
            tel = tel[sel]
            out["telluric_lines"] = np.array(tel[..., 0], dtype=np.float32)
            out["telluric_continuum"] = np.array(tel[..., 1], dtype=np.float32)
    return out

"""Data backends: the real compact store, and a procedural synthetic Sun for tests.

Both expose the same attributes, so `dataset.SolarDataset` and the scorer do not
care which one they are given.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import l2io

N_ORDERS, N_PIX = l2io.N_ORDERS, l2io.N_PIX
N_VEL = 804


class StoreBackend:
    """Real NEID data: light columns in memory, CCFs per day shard, spectra from raw FITS."""

    def __init__(self, store_dir, l2_dir=None, ccf_cache_days: int = 6):
        self.store = Path(store_dir)
        z = np.load(self.store / "index.npz", allow_pickle=False)
        self.obs_id = z["obs_id"]
        self.time = z["jd"]
        self.day = z["day"]
        self.rv, self.rv_sum, self.rv_err = z["rv"], z["rv_sum"], z["rv_err"]
        self.order_rv, self.order_wt, self.ssbz = z["order_rv"], z["order_wt"], z["ssbz"]
        self.scal_names = [str(s) for s in z["scal_names"]]
        self.scalars = z["scalars"]
        self.act_names = [str(s) for s in z["act_names"]]
        self.act_val, self.act_unc = z["act_val"], z["act_unc"]
        self.ccf_start, self.ccf_step = z["ccf_start"], z["ccf_step"]
        self._relpath, self._row = z["relpath"], z["row"]
        hint = self.store / "l2_dir"
        self._l2_dir = Path(l2_dir or (hint.read_text().strip() if hint.exists() else "."))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_days = ccf_cache_days
        self._lock = threading.Lock()
        self.has_spectra = True
        self.n_vel = N_VEL

    def _day_ccf(self, day: int) -> np.ndarray:
        with self._lock:
            if day in self._cache:
                self._cache.move_to_end(day)
                return self._cache[day]
        arr = np.load(self.store / "days" / f"{day}.npz", allow_pickle=False)["ccf"]
        with self._lock:
            self._cache[day] = arr
            while len(self._cache) > self._cache_days:
                self._cache.popitem(last=False)
        return arr

    def ccf(self, idx) -> np.ndarray:
        idx = np.atleast_1d(idx)
        out = np.empty((len(idx), N_ORDERS, N_VEL), dtype=np.float32)
        for d in np.unique(self.day[idx]):
            m = self.day[idx] == d
            out[m] = self._day_ccf(int(d))[self._row[idx[m]]]
        return out

    def spectrum(self, i: int, orders=None) -> dict:
        sp = l2io.read_spectrum(self._l2_dir / str(self._relpath[i]), orders)
        sp["ssb_z"] = self.ssbz[i] if orders is None else self.ssbz[i][np.asarray(orders, int)]
        return sp


# ----------------------------------------------------------------------------------------------
ACT_NAMES = [
    "CaIIHK_tellcorr", "CaIIHK", "HeI_1_tellcorr", "HeI_1", "HeI_2_tellcorr", "HeI_2",
    "NaI_tellcorr", "NaI", "Ha06_1_tellcorr", "Ha06_1", "Ha06_2_tellcorr", "Ha06_2",
    "Ha16_1_tellcorr", "Ha16_1", "Ha16_2_tellcorr", "Ha16_2", "CaI_1_tellcorr", "CaI_1",
    "CaI_2_tellcorr", "CaI_2", "CaIRT1_tellcorr", "CaIRT1", "CaIRT2_tellcorr", "CaIRT2",
    "CaIRT3_tellcorr", "CaIRT3", "NaINIR_tellcorr", "NaINIR", "PaDelta_tellcorr", "PaDelta",
    "Mn539_tellcorr", "Mn539", "CCFFWHMMOD", "CCFBISMOD",
]


def _oscillator(rng, n, period, lifetime):
    """Damped stochastic oscillator sampled daily, unit variance."""
    r, w = np.exp(-1.0 / lifetime), 2 * np.pi / period
    x = np.zeros(n + 400)
    e = rng.standard_normal(n + 400)
    for k in range(2, n + 400):
        x[k] = 2 * r * np.cos(w) * x[k - 1] - r * r * x[k - 2] + e[k]
    x = x[400:]
    return (x - x.mean()) / x.std()


class SynthBackend:
    """A toy Sun with the same array shapes as NEID L2 data.

    NOT physically faithful. It exists so code paths can be exercised without the
    real data: activity perturbs RVs, CCF shapes, line depths and indices in a
    correlated way, on top of shape-less (super)granulation and instrument noise.
    """

    def __init__(self, seed: int = 0, n_days: int = 240, obs_per_day: int = 12,
                 with_spectra: bool = True, day_keep: float = 0.82):
        rng = np.random.default_rng(seed)
        self._seed = seed
        day0 = 2459300
        keep = rng.random(n_days) < day_keep
        days = day0 + np.nonzero(keep)[0]
        frac = 0.17 + 0.30 * (np.arange(obs_per_day) + 0.5) / obs_per_day
        self.day = np.repeat(days, obs_per_day)
        self.time = (self.day + np.tile(frac, len(days))).astype(np.float64)
        self.time += rng.normal(0, 2e-4, self.time.size)
        n = self.time.size
        self.obs_id = np.array([f"synth_{d}_{j:03d}" for d in days for j in range(obs_per_day)])
        tday = self.time - day0
        grid = np.arange(n_days + 2)
        cyc = 1.0 + 0.6 * grid / max(n_days, 1)
        F = np.interp(tday, grid, cyc * (1 + 0.6 * _oscillator(rng, n_days + 2, 27.0, 40.0)))
        S = np.interp(tday, grid, _oscillator(rng, n_days + 2, 27.0, 25.0))
        self._F = (F - F.mean()) / F.std()
        self._S = S
        ha = (np.tile(frac, len(days)) - 0.32) * 2 * np.pi
        airmass = 1.0 / np.clip(np.cos(0.55) * np.cos(ha) * 0.95, 0.3, None)
        dt = np.diff(self.time, prepend=self.time[0])
        sg = np.zeros(n)
        for k in range(1, n):                                    # OU supergranulation, tau = 1.5 d
            a = np.exp(-dt[k] / 1.5)
            sg[k] = a * sg[k - 1] + np.sqrt(1 - a * a) * 0.35 * rng.standard_normal()
        inst = np.interp(tday, grid, np.cumsum(rng.normal(0, 0.03, n_days + 2)))
        inst += 0.15 * (airmass - airmass.mean())
        common = sg + inst + rng.normal(0, 0.40, n)             # shape-less terms, m/s
        lam = 613000.0 / (l2io.ORDER_MAX - np.arange(N_ORDERS))
        self._lam = lam
        chrom = 1.0 - 0.3 * (lam - 5500.0) / 2000.0
        good = (np.arange(N_ORDERS) >= 15) & (np.arange(N_ORDERS) <= 100)
        self.rv_err = np.full(n, 0.25) * (airmass / airmass.mean()) ** 0.3
        sig_o = self.rv_err[:, None] * np.sqrt(good.sum())
        self._act_rv = 2.0 * self._F[:, None] * chrom[None, :] + 0.8 * S[:, None] * chrom[None, :] ** 2
        orv = self._act_rv + common[:, None] + rng.normal(0, 1, (n, N_ORDERS)) * sig_o - 640.0
        orv[:, ~good] = np.nan
        self.order_rv = orv
        self.order_wt = np.tile(good.astype(float), (n, 1))
        self.rv = np.nanmean(orv, axis=1)
        self.rv_sum = self.rv + rng.normal(0, 0.05, n)
        self._good = good
        self.ssbz = np.tile((400 * np.sin(2 * np.pi * tday / 365.25) - 300 * np.sin(ha))[:, None]
                            / 299792458.0, (1, N_ORDERS))
        self.scal_names = sorted(l2io.PRIMARY_SCALARS + l2io.CCF_SCALARS + l2io.TELLURIC_SCALARS)
        sc = rng.normal(0, 1, (n, len(self.scal_names))) * 0.01 + 1.0
        col = {k: i for i, k in enumerate(self.scal_names)}
        sc[:, col["AIRMASS"]] = airmass
        sc[:, col["EXTSNR"]] = 360 / airmass ** 0.3
        sc[:, col["OBSJD"]] = self.time
        sc[:, col["EXPTIME"]] = 55.0
        sc[:, col["FWHMMOD"]] = 7.07 + 0.004 * self._F + rng.normal(0, 5e-4, n)
        sc[:, col["BISMOD"]] = -0.048 + 0.002 * S - 0.001 * self._F + rng.normal(0, 5e-4, n)
        sc[:, col["DVRMSMOD"]] = self.rv_err / 1e3
        sc[:, col["DQLEVEL2"]] = 0
        self.scalars = sc
        self.act_names = list(ACT_NAMES)
        sens = rng.normal(0, 1, (2, len(ACT_NAMES))) * 2e-3
        base = 0.2 + 0.6 * rng.random(len(ACT_NAMES))
        self.act_val = base + self._F[:, None] * sens[0] + S[:, None] * sens[1] * 0.5 \
            + rng.normal(0, 2e-4, (n, len(ACT_NAMES)))
        self.act_val[:, self.act_names.index("CCFFWHMMOD")] = sc[:, col["FWHMMOD"]]
        self.act_val[:, self.act_names.index("CCFBISMOD")] = sc[:, col["BISMOD"]]
        self.act_unc = np.full_like(self.act_val, 2e-4)
        self.ccf_start = np.full(n, -100.0)
        self.ccf_step = np.full(n, 0.25)
        self.n_vel = N_VEL
        self.has_spectra = with_spectra
        orng = np.random.default_rng(seed + 7)
        self._ccf_depth = 0.25 + 0.3 * orng.random(N_ORDERS)
        self._ccf_amp = 10 ** (6 + 1.5 * orng.random(N_ORDERS))

    def ccf(self, idx) -> np.ndarray:
        idx = np.atleast_1d(idx)
        u = -100.0 + 0.25 * np.arange(N_VEL)
        out = np.zeros((len(idx), N_ORDERS, N_VEL), dtype=np.float32)
        g = np.nonzero(self._good)[0]
        for j, i in enumerate(idx):
            rng = np.random.default_rng((self._seed, 11, int(i)))
            x = (u[None, :] - self.order_rv[i, g, None] / 1e3) / 3.0
            prof = np.exp(-0.5 * x * x)
            prof = prof * (1 + 2e-3 * self._F[i] * (x * x - 1) + 1.5e-3 * self._S[i] * (x ** 3 - 3 * x) / 3)
            c = self._ccf_amp[g, None] * (1 - self._ccf_depth[g, None] * prof)
            out[j, g] = c + np.sqrt(c) * rng.standard_normal(c.shape)
        return out

    def spectrum(self, i: int, orders=None) -> dict:
        if not self.has_spectra:
            raise RuntimeError("this dataset has no spectra")
        orders = np.arange(N_ORDERS) if orders is None else np.asarray(orders, int)
        x = np.linspace(-1, 1, N_PIX)
        wave = np.empty((len(orders), N_PIX))
        flux = np.empty((len(orders), N_PIX), dtype=np.float32)
        blaze = np.empty_like(flux)
        nrng = np.random.default_rng((self._seed, 13, int(i)))
        for k, o in enumerate(orders):
            lrng = np.random.default_rng((self._seed, 17, int(o)))          # fixed line list per order
            lam_c = self._lam[o]
            w = lam_c * (1 + x / 105.0)
            cen = lam_c * (1 + lrng.uniform(-0.95, 0.95, 60) / 105.0)
            depth = lrng.uniform(0.1, 0.8, 60)
            gdep, kshift = lrng.normal(0, 1e-3, 60), lrng.normal(1.0, 0.5, 60)
            v = (self._act_rv[i, o] * kshift + (self.rv[i] - np.nanmean(self._act_rv[i])) - 0.0)
            v = v - self.ssbz[i, o] * 299792458.0                            # observatory frame
            lc = cen * (1 + v / 299792458.0)
            sig = cen * 3000.0 / 299792458.0
            f = np.ones(N_PIX)
            for c0, s0, d0, g0 in zip(lc, sig, depth, gdep):
                a, b = np.searchsorted(w, (c0 - 6 * s0, c0 + 6 * s0))
                f[a:b] -= d0 * (1 + g0 * self._F[i]) * np.exp(-0.5 * ((w[a:b] - c0) / s0) ** 2)
            bz = (np.sinc(0.8 * x) ** 2) * (1e5 if self._good[o] else 2e3)
            f = np.clip(f, 0.02, None) * bz
            wave[k], blaze[k] = w, bz
            flux[k] = f + np.sqrt(f) * nrng.standard_normal(N_PIX)
        return {"wave": wave, "flux": flux, "var": np.abs(flux).astype(np.float32), "blaze": blaze,
                "ssb_z": self.ssbz[i][orders]}


def open_backend(spec: dict):
    """spec: {'kind': 'store', 'store_dir':..., 'l2_dir':...} or {'kind': 'synth', ...kwargs}."""
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "store":
        return StoreBackend(spec["store_dir"], spec.get("l2_dir"))
    if kind == "synth":
        return SynthBackend(**spec)
    raise ValueError(kind)

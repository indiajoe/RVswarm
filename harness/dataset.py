"""The only object an algorithm receives: a view of solar observations.

Any hidden planetary signal has already been applied consistently to every
RV-bearing quantity (wavelength solutions, CCF velocity grids, pipeline RVs).
The shift is held in closures only, never as an attribute.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import l2io

C_MS = 299_792_458.0


class SolarDataset:
    """Sun-as-a-star observations from NEID L2 files (see workspace/docs/ALGORITHM_INTERFACE.md)."""

    def __init__(self, backend, index, shift_ms):
        idx = np.asarray(index, dtype=np.int64)
        v = np.asarray(shift_ms, dtype=np.float64).copy()
        if v.shape != idx.shape:
            raise ValueError("shift/index mismatch")
        ro = lambda a: _readonly(np.array(a))
        self.n_obs = int(idx.size)
        self.n_orders = l2io.N_ORDERS
        self.n_pix = l2io.N_PIX
        self.n_vel = int(backend.n_vel)
        self.has_spectra = bool(backend.has_spectra)
        self.obs_id = ro(backend.obs_id[idx])
        self.time = ro(backend.time[idx])                       # BJD_TDB (days)
        self.day = ro(backend.day[idx])                         # integer observing-day label
        self.physical_order = ro(l2io.ORDER_MAX - np.arange(l2io.N_ORDERS))
        self.rv_pipeline = ro(backend.rv[idx] + v)              # m/s, order-weighted CCF RV
        self.rv_pipeline_sum = ro(backend.rv_sum[idx] + v)      # m/s, summed-CCF RV
        self.rv_pipeline_err = ro(backend.rv_err[idx])          # m/s, photon-limited estimate
        self.order_rv = ro(backend.order_rv[idx] + v[:, None])  # (n_obs, n_orders) m/s, NaN = unused
        self.order_weight = ro(backend.order_wt[idx])
        self.scalar_names = tuple(backend.scal_names)
        self.activity_names = tuple(backend.act_names)
        scal = ro(backend.scalars[idx])
        act_v, act_u = ro(backend.act_val[idx]), ro(backend.act_unc[idx])
        s_col = {k: i for i, k in enumerate(self.scalar_names)}
        a_col = {k: i for i, k in enumerate(self.activity_names)}
        c_start, c_step = backend.ccf_start[idx], backend.ccf_step[idx]

        def scalar(name):
            """Per-observation header scalar (see `scalar_names`)."""
            return scal[:, s_col[name]]

        def activity(name):
            """(value, uncertainty) of a pipeline activity index (see `activity_names`)."""
            return act_v[:, a_col[name]], act_u[:, a_col[name]]

        def ccf(sel=None):
            """Per-order CCFs. Returns (velocity[k, n_vel] in km/s, ccf[k, n_orders, n_vel])."""
            sel = np.arange(idx.size) if sel is None else np.atleast_1d(np.asarray(sel))
            if sel.dtype == bool:
                sel = np.nonzero(sel)[0]
            vel = (c_start[sel, None] + c_step[sel, None] * np.arange(backend.n_vel)[None, :]
                   + v[sel, None] / 1e3)
            return vel, backend.ccf(idx[sel])

        def ccf_batches(batch_size=256):
            """Iterate (obs_indices, velocity, ccf) over the whole dataset in time order."""
            for a in range(0, idx.size, batch_size):
                sel = np.arange(a, min(a + batch_size, idx.size))
                yield (sel, *ccf(sel))

        def spectrum(i, orders=None):
            """Extracted 2-D spectrum of observation i (dict of arrays, [n_sel_orders, n_pix])."""
            i = int(i)
            sp = backend.spectrum(int(idx[i]), orders)
            sp["wave"] = sp["wave"] * (1.0 + v[i] / C_MS)
            sp["order_index"] = (np.arange(l2io.N_ORDERS) if orders is None
                                 else np.asarray(orders, dtype=int))
            return sp

        def spectra(indices=None, orders=None, workers=4):
            """Iterate (i, spectrum) with background reading threads."""
            indices = range(idx.size) if indices is None else [int(i) for i in indices]
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
                for i, sp in zip(indices, ex.map(lambda j: spectrum(j, orders), indices)):
                    yield i, sp

        def subset(sel):
            """A new dataset restricted to a boolean mask or index array."""
            sel = np.asarray(sel)
            if sel.dtype == bool:
                sel = np.nonzero(sel)[0]
            return SolarDataset(backend, idx[sel], v[sel])

        self.scalar, self.activity, self.ccf, self.ccf_batches = scalar, activity, ccf, ccf_batches
        self.spectrum, self.spectra, self.subset = spectrum, spectra, subset

    def __len__(self):
        return self.n_obs

    def __repr__(self):
        return f"<SolarDataset n_obs={self.n_obs} days={len(np.unique(self.day))}>"


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a

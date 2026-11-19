# Algorithm interface

This is the complete contract between an algorithm and the evaluation harness.
Nothing outside this document is available to algorithm code.

## 1. Files

An algorithm lives in `workspace/algorithms/<alg_id>/`:

| file | purpose |
|---|---|
| `SPEC.md` | the approved proposal the code implements (written by the proposer) |
| `algorithm.py` | must define `run(ctx)`; may import sibling `.py` files in the same directory |
| `algorithm.yaml` | name, family, resource request (below) |
| `native/` (optional) | C, C++, Fortran or Julia sources called from `algorithm.py` (section 5a) |
| `reduction/` (optional) | a preliminary reduction step, `reduce.py` with `reduce(ctx)` (section 2a) |
| `IMPLEMENTATION_NOTES.md` | what was built, every deviation from the spec and why |

```yaml
# algorithm.yaml
name: short human-readable name
family: F03                 # registry family id
resources:
  cpus: 8                   # <= 32
  mem_gb: 64                # <= 256
  gpus: 0                   # 0 or 1
  hours: 4                  # wall time for all runs of a SCREEN evaluation (development data,
                            # 2 runs). Later stages automatically get 2x (full) and 3x (sealed),
                            # capped at 8 / 16 / 24 h. Request memory for the complete data set.
  parallel_runs: false      # true: the injected runs of an evaluation share the CPUs concurrently, after
                            # the un-injected run (CPU only; each of the two phases gets half the hours)
```

**CPU time on the cluster is scarce.** Ask for what the code really uses: single-threaded
code should request 1 or 2 CPUs. Every result report states the CPU time used and what
fraction of the reserved cores was busy.

## 2. Entry point

```python
def run(ctx):
    ds = ctx.target                      # SolarDataset: the observations to correct
    ...
    return {"rv_clean": rv,              # REQUIRED float array [ds.n_obs], m/s, one per observation,
                                         # same order as ds.obs_id. Arbitrary zero point.
            "rv_err": err}               # optional, same shape
```

`rv_clean` is your best estimate of the Sun's centre-of-mass radial velocity at each
observation with stellar-activity, granulation and instrumental signals removed.
A real planet's Doppler signal must survive in it.

* NaN marks an observation you could not correct. More than 5 % NaN (per observation
  or per day) fails the run.
* The function is called in a fresh process for each run of an evaluation. The same
  code is run two or three times on data that differ only by hidden injected planets;
  there is no way to tell the runs apart, and nothing persists between them: each process
  has its own private directory (its scratch files included), deleted when it ends.
* The result must not depend on the time budget, on `ctx.n_cpus`, or on whether
  reduction products came from the cache (section 2a): the harness takes differences
  between runs, and at the full stage it does not repeat the un-injected run of the
  screen stage when the code is unchanged.
* Do not print or return anything that depends on knowing the hidden planets; you
  cannot know them.

## 2a. Optional: a preliminary reduction step

Many methods start with an expensive step that depends on the data alone and not on the
correction model: line-by-line RVs and their errors, per-line shape measurements, a stack
of residual spectra. Such a step can be split off, so that it is computed once and reused:

```python
# reduction/reduce.py
"""Line-by-line RVs against a template built from all spectra.

Products: lbl_rv [n_obs, n_lines] m/s; lbl_err [n_obs, n_lines] m/s; line_wave [n_lines] Angstrom.
"""
def reduce(ctx):
    ds = ctx.target
    ...
    return {"lbl_rv": rv, "lbl_err": err, "line_wave": wave}      # numeric arrays only

# algorithm.py
def run(ctx):
    p = ctx.reduction()           # exactly what reduce(ctx) returned, as read-only arrays
    ...
```

* `reduce(ctx)` runs in its own process before `run(ctx)`, on the same data (including
  any hidden planets). It gets the same `ctx` except `ctx.metric`, `ctx.diagnostic` and
  `ctx.reduction`. It returns a dict of numeric arrays (names are identifiers; at most 64
  arrays); `run(ctx)` receives nothing else from it.
* All its code lives in `reduction/` (`.py` files that import each other, and an
  optional `reduction/native/`). `algorithm.py` cannot import from `reduction/`.
* The harness caches the products of the run WITHOUT hidden planets, under the exact
  code of `reduction/` and the data set. Every later algorithm whose `reduction/` is
  byte-identical gets them without recomputing. Runs with hidden planets always
  recompute the step: each algorithm has its own planets.
* Proposals declare it in the front matter: `reduction: none`, `new` (specified in the
  proposal) or `reuse <algorithm id>`. For `reuse`, the orchestrator copies that
  algorithm's `reduction/` unchanged and it is read-only for the implementer.
  `workspace/registry/REDUCTIONS.md` lists the reduction steps that exist.

## 3. `ctx`

| member | meaning |
|---|---|
| `ctx.target` | `SolarDataset` (section 4) |
| `ctx.n_cpus`, `ctx.gpu` | granted CPU cores; whether a CUDA GPU is available |
| `ctx.time_budget_s`, `ctx.elapsed()` | wall-time budget for this run (the same in every run of an evaluation) and seconds used so far. Exceeding it kills the run. |
| `ctx.seed` | integer; use it for all random number generators (identical across runs) |
| `ctx.scratch_file(name)` | path string of a private scratch file (flat names like `"feat.npy"`); use with `np.save`, `np.load`, `np.memmap`, `torch.save`. Deleted when the run's process ends. |
| `ctx.reduction()` | dict of the products of the algorithm's preliminary reduction step (section 2a) |
| `ctx.native_library(openmp=False, flags=())`, `ctx.native_file(name)` | compiled-language support, section 5a |
| `ctx.log(*parts)` | progress message, with the elapsed time |
| `ctx.metric(name, value)` | record a scalar intermediate check for the result report |
| `ctx.diagnostic(name, array)` | record a small array (periodogram, eigen-spectrum, coefficients...) for the result report. 20 MB total. |

`ctx.metric` / `ctx.diagnostic` are how proposers get the intermediate checks they
asked for. Reports on real data show what the algorithm itself prints and records only
in reduced form, so that per-epoch values of the data never reach the collaboration:

* development data (screen, full), un-injected run only: up to 20 `ctx.metric` values,
  rounded to 3 significant digits; up to 8 `ctx.diagnostic` arrays, each as shape and
  summary statistics, with the values themselves only for arrays of at most 16 numbers
  (rounded); the last 40 lines of the log with every digit shown as `#`, except the time
  stamps of `ctx.log` lines and the line numbers in tracebacks. Error messages are masked
  the same way.
* sealed data: only the masked error message, if the run failed.
* the local self-test (`run_tests`, section 7): everything, unchanged.

So put the numbers you need into `ctx.metric` and into short `ctx.diagnostic` arrays (an
eigenvalue spectrum, a few coefficients, a coarse periodogram), not into log lines.

## 4. `SolarDataset`

All RV-bearing quantities already contain any hidden planet, consistently.

**Every index is 0-based** (Python / C convention): observation `i` in 0 ... `n_obs`-1,
order index 0 ... 121, pixel 0 ... 9215, velocity step 0 ... 803. Arrays are NumPy arrays
in C (row-major) order. Fortran and Julia count from 1 and store column-major: see
"Index convention" in the data description and section 5a.

Attributes (read-only NumPy arrays, length `n_obs`, sorted by time):

| attribute | meaning |
|---|---|
| `n_obs`, `n_orders` (122), `n_pix` (9216), `n_vel` (804) | sizes |
| `obs_id` | observation identifiers |
| `time` | BJD_TDB of flux-weighted mid-exposure, days |
| `day` | integer observing-day label (one value per calendar day of observing) |
| `rv_pipeline`, `rv_pipeline_err` | NEID DRP order-weighted CCF RV and its photon-noise estimate, m/s |
| `rv_pipeline_sum` | DRP RV from the summed CCF, m/s |
| `order_rv`, `order_weight` | `[n_obs, 122]` per-order DRP CCF RVs (m/s; NaN where the DRP has none) and DRP order weights |
| `physical_order` | `[122]` echelle order number of each 0-based order index (173 at index 0 ... 52 at index 121) |
| `scalar_names`, `activity_names` | available names for the two accessors below |
| `has_spectra` | True when `spectrum()` is available |

Methods:

| call | returns |
|---|---|
| `ds.scalar(name)` | `[n_obs]` header scalar. Includes CCF shape (`FWHMMOD`, `BISMOD`, `FITMODV0/2/3`, ...), observing geometry (`SUNAGL`, `ZENITH` = solar zenith angle; `AIRMASS` and `TCS*` describe the idle telescope, not the Sun), `NCOMBINE`, `EXPTIME`, `EXTSNR`, drift (`DRIFTRV0`), telluric model (`WVAPOR`, `ZENITH`), environment and thermal telemetry (`ENV*`, `LS218_*`, `CCDTEMP`, `TMC*`). |
| `ds.activity(name)` | `(value, uncertainty)`, each `[n_obs]`: DRP activity indices (`CaIIHK`, `Ha06_1`, `NaI`, `CaIRT1..3`, `HeI_1`, `Mn539`, `PaDelta`, ... and `_tellcorr` variants) |
| `ds.ccf(sel=None)` | `(velocity, ccf)`: `velocity [k, 804]` km/s and `ccf [k, 122, 804]` float32 per-order CCFs (G2 ESPRESSO mask, 0.25 km/s steps, about -100 ... +100 km/s). Orders without mask lines are all zeros. `sel` = 0-based index array / boolean mask / None (all; can be very large -- prefer batches). |
| `ds.ccf_batches(batch_size=256)` | iterator of `(obs_indices, velocity, ccf)` over the whole set |
| `ds.spectrum(i, orders=None)` | dict for observation `i` (0-based) and the selected 0-based order indices (default all): `wave` (float64, vacuum Angstrom, **observatory frame**), `flux`, `var` (electrons; not blaze corrected; already telluric-corrected), `blaze`, each `[n_sel, 9216]` (no telluric model arrays are provided); `ssb_z` `[n_sel]` (DRP barycentric redshift; `wave * (1 + ssb_z)` is the barycentric wavelength); `order_index`. Pixels may be NaN. |
| `ds.spectra(indices=None, orders=None, workers=4)` | iterator of `(i, spectrum_dict)` with threaded read-ahead |
| `ds.subset(sel)` | new `SolarDataset` restricted to a mask / index array |

A `SolarDataset` cannot be pickled. For parallel work use threads, `ds.spectra(workers=...)`,
or process pools that receive plain NumPy arrays.

Cost guide: scalars, RVs and indices are in memory. CCFs cost about 0.4 MB per
observation to load. A full spectrum is about 30 MB and roughly 0.1-0.3 s to read. The
data set is a few hundred to a couple of thousand observations (20-minute averages,
usually one per observing day), so a pass over every spectrum is affordable; still read
only the orders you need and cache derived features with `ctx.scratch_file`.

## 5. Rules for code (enforced by an automatic audit before every run)

* Imports are limited to: numpy, scipy, sklearn, torch, jax, optax, numba, pandas,
  statsmodels, astropy (not `astropy.io`), celerite2, george, tinygp, emcee, pywt,
  xgboost, lightgbm, einops, joblib, and the standard-library modules math, cmath,
  dataclasses, typing, functools, itertools, collections, time, warnings, json,
  logging, random, statistics, heapq, bisect, copy, enum, abc, re, string, datetime,
  concurrent, multiprocessing, threading, queue, numbers, contextlib,
  hashlib, textwrap, fractions, decimal.
* **More packages are available on request.** If a well-known astronomy or data-science
  package (or a compiled language, section 5a) would let a method be implemented better,
  list its import name under `requires:` in the proposal front matter. The human
  maintainers install it; a selected specification waits in
  `workspace/environment_requests/PENDING.md` until then and is implemented
  automatically afterwards. Do not distort a method to fit the default list.
* `astropy.io`, `scipy.io`, `pandas.io` and similar file readers are excluded on purpose:
  an algorithm never opens the FITS files. The harness reads them and hands over the
  data through `SolarDataset`, already carrying the hidden planets; direct file access
  would bypass that. Everything in the L2 files is reachable through `SolarDataset`; if
  something is missing, ask for it in `workspace/diagnostic_requests/`.
* Not allowed in Python code: `os`, `sys`, `subprocess`, `pathlib`, `open`, `eval`, `exec`,
  `getattr`/`setattr`, `vars`, `globals`, dunder attributes other than `__init__`-style
  methods, private attributes (`._x`) of objects other than `self`, path-like string
  literals, network access, reading any file other than your own scratch files.
* **Only information contained in the NEID L2 solar spectra files may be used.** No
  external solar data (SDO, sunspot numbers, F10.7, magnetograms, other
  spectrographs), no hard-coded tables derived from such data, no pre-trained
  weights. General physical constants, atomic line lists typed into the code with a
  citation, and textbook ephemeris formulae are fine.
* No hard-coded dates, observation ids or per-epoch corrections. The same code
  must work on any time span of this instrument.

## 5a. Other languages: C, C++, Fortran, Julia

The entry point is always `run(ctx)` in `algorithm.py`, which reads the data through
`SolarDataset` and returns the result. **Python is the preferred language**: use
another one only for a numerical core that would be significantly slower in Python
(NumPy, Numba, JAX, PyTorch), and say why in the specification or the notes. Such a
core may then be written in C, C++, Fortran or Julia. Put the sources, flat, in `native/` (`.c .h .cpp .hpp .f90 .f .jl`; no
binaries, no build scripts). Native code receives NumPy arrays from `algorithm.py` and
must not touch files, processes, the network or environment variables itself.

**C, C++, Fortran.** `lib = ctx.native_library(openmp=False, flags=())` compiles every
source in `native/` into one shared library (`cc` / `c++` / `gfortran`, `-O2 -fPIC
-shared`, plus `-fopenmp` on request) and returns a `ctypes.CDLL`. Optional `flags` from:
`-O3 -ffast-math -march=native -funroll-loops -std=c11 -std=c99 -std=c++17 -std=f2008
-llapack -lblas -lfftw3 -lgsl -lgslcblas`. `import ctypes` and `numpy.ctypeslib` are
allowed in `algorithm.py` when `native/` exists. Export C-callable symbols (`extern "C"`
in C++, `bind(C)` in Fortran):

```python
import ctypes, numpy as np
def run(ctx):
    lib = ctx.native_library(openmp=True)
    vec = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.my_kernel.argtypes = [vec, vec, ctypes.c_int]; lib.my_kernel.restype = ctypes.c_double
    ...
```

**Indices and array layout across the language boundary.** Everything in this document and
in `SolarDataset` is 0-based and row-major. C and C++ share that convention. Fortran and
Julia do not: their arrays start at 1, and a C-ordered NumPy array of shape `[n_orders, n_pix]`
arrives there as an array of shape `(n_pix, n_orders)` over the same memory. In native code:
convert every index that crosses the boundary explicitly (0-based index `k` is element
`k + 1`), pass array sizes as arguments instead of assuming them, write the base of each
index in a comment, and pass `np.ascontiguousarray(x, dtype=np.float64)` so the layout is
the one the native routine expects.

**Julia.** `from juliacall import Main as jl`, then
`jl.include(ctx.native_file("model.jl"))` and call the Julia functions with NumPy
arrays. Julia packages beyond the standard library must be requested (`requires:`).

Which languages are enabled is decided by the human maintainers; the audit reports a
language that is not enabled, and it can be requested like a package. The toy self-test
(`run_tests`) builds and runs native code exactly as the cluster does.

## 6. How an evaluation is scored (fixed; proposers and implementers have no say)

1. Your code runs on the data as it is.
   **Pre-check (screen stage):** the RMS per epoch of your `rv_clean` is compared with that
   of the pipeline RVs (`rv_pipeline`, the DRP's CCFRVMOD). If it is not below 0.95 times
   the pipeline value, the evaluation stops there: no planet injection is run, the result
   is a gate failure, and the report gives the scatter numbers but no planet transfer.
   Such an algorithm could not be promoted in any case, and the CPU time is needed elsewhere.
   Otherwise the code runs again on copies with hidden circular-orbit planets (one per
   period band between 2 and 400 days, semi-amplitudes of order 10-25 cm/s)
   Doppler-injected at the spectrum level.
2. Outputs are combined per epoch (observing day) with fixed pipeline weights; with one
   spectrum per day an epoch is a single observation.
3. **RMS** = standard deviation over epochs of `rv_clean` of the un-injected run.
4. **Transfer** per planet = recovered / injected amplitude, from the difference
   between injected and un-injected outputs. It must lie in [0.8, 1.2] for every
   planet, with little phase distortion (the *gate*).
5. **Effective RMS** = RMS / min(1, smallest transfer). This is the ranking metric.
6. Goal: gate passed and effective RMS below 0.20 m/s, confirmed on sealed data that no
   one has developed against.

Every development-data report also has a section **Residual diagnostics**, computed by
the harness from the un-injected run: the shape of the residual distribution;
correlations of the corrected RVs with observing conditions (barycentric correction,
solar zenith angle and airmass, hour angle, water vapour) and with activity indicators
(FWHM, bisector, Ca II H&K, H-alpha); the structure function and the periodicities left;
mean and RMS per season and per year. Each is shown next to the same statistic for the
pipeline RVs. Use them to find what structure is left in the residuals. The human
scientists maintain this list: ask for another diagnostic with a `REQUEST TO HUMANS:`
line in a post-mortem. Sealed reports contain none.

Stages: `screen` (development data, 1 injection) -> `full` (development
data, 2 injections; its un-injected run is the screen one when the code is unchanged) ->
`sealed` (rationed). Promotion between stages follows fixed numerical rules.

## 6a. Reviews between implementation and evaluation

Before an implementation is queued for the cluster it is examined by three independent
agents that did not write it: a **code reviewer** (integrity, fidelity to the
specification, correctness), a **numerical analyst** (indexing, precision, conditioning,
convergence, bad values) and, once those two pass it, a **performance engineer** (CPU
time, resource request). Performance changes must leave the results unchanged; the
harness compares the test output before and after. Their findings are in
`algorithms/<alg_id>/REVIEW.md`.

## 7. Local self-test available to implementers

The `run_tests` tool runs the audit and then the algorithm on a tiny local data set,
because the orchestrator host has little compute. Normally this is **about five real
NEID spectra picked by the human scientists** (the report is then titled "Smoke test").
It proves that the code runs on real data and returns the right shapes, nothing more:
five spectra cannot measure scatter or planet preservation, which are measured only on
the cluster. Consequences for implementers:

* The code must run without crashing when `n_obs` is about 5: clip ranks and numbers of
  components to the data, and fall back to the pipeline RV where a model cannot be fitted.
* Never tune anything to those few spectra.
* The report lists, per spectrum, the injected shift and the change in your output; for a
  planet-preserving method these agree.
* It shows your log, `ctx.metric` values and `ctx.diagnostic` arrays unchanged (reports
  on real data show them only in reduced form, section 3).

If no real frames are configured, a toy synthetic Sun with the same array shapes is used
instead (about 300 observations over about 900 days). It is not physically faithful.

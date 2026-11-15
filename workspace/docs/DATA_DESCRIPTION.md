# The NEID L2 solar data, as seen through the harness

Facts below were measured on real files; treat them as reliable.

## Observations

* Sun-as-a-star through the NEID solar feed (integrating sphere, HR fibre), `OBSTYPE=Eng`,
  `OBJECT=Sun`. The instrument takes 55 s exposures at about 93 s cadence. **Each spectrum you
  are given is the mean of 4 exposures taken within about 5 to 10 minutes** (header
  scalar `NCOMBINE` = 4), which integrates over the 5-minute p-mode oscillations. The
  four frames were each shifted to the barycentric frame, interpolated onto the
  wavelength grid of the first frame and averaged. Measured on sample files: the flux
  variance is that of the mean, and S/N per pixel near 660 nm is about 1150 (a single
  exposure: about 550).
* **What the header-derived values mean for a combined spectrum** (checked on sample files):
  - `time`, `ssb_z` and the `OBSJD` scalar are those of the FIRST of the four frames.
    The effective mid-time is 2 to 5 minutes later. `ssb_z` is the correct one for the
    wavelength grid you are given.
  - `rv_pipeline`, `rv_pipeline_sum`, `order_rv`, the CCF arrays, and the other numeric
    header scalars are the MEAN over the four frames. A Gaussian refit of the averaged
    CCF reproduces `rv_pipeline` to about 1 mm/s.
  - `rv_pipeline_err` is the mean single-frame value divided by sqrt(`NCOMBINE`): about
    0.12 m/s. The raw scalars `DVRMSMOD`, `EFWHMMOD`, `EBISMOD` and `EXTSNR` are still
    single-frame means (about 0.245 m/s and S/N 510), not values for the combined spectrum.
  - `EXPTIME` is the single-frame exposure time (55 s), not the total.
  - The `AIRMASS` scalar is the pointing of the idle night-time telescope and says nothing
    about the Sun. Use `ZENITH` (solar zenith angle, degrees) instead.
  - Whether the activity-index table is the mean of four frames has not been verified.
* **The flux is already telluric-corrected** (divided by the DRP telluric model), and no
  telluric model arrays are provided. In saturated telluric cores (for example the O2 A
  band near 7600-7630 A) the division leaves wild values: mask such regions yourself.
* **Sampling: about 500 such spectra over about five years**, curated by the human
  scientists; typically a single spectrum on an observing day, with gaps of days between
  them. There is no within-day time series: methods that rely on dense sampling in time
  (intra-day modelling, nightly binning, short-timescale filters) have nothing to work
  with, and data-driven models must be fitted on a few hundred spectra.
* Calibration fibre: etalon simultaneously. Wavelength solution: LFC + ThAr with a
  daily drift model (`WAVECAL=LFCplusThAr`, `DRIFTFUN=dailymodel0`).
* The numbers quoted in the sections below were measured on single 55 s exposures
  (S/N per pixel near 550 nm about 350-400, photon RV uncertainty about 0.27 m/s).

## Spectral format

* 122 extracted orders x 9216 pixels. Order index 0 is echelle order 173 (~370 nm),
  index 121 is echelle order 52 (~1.1 micron). Order `m` is centred near
  613000/m Angstrom and spans about lambda/52.
  Examples: index 10 = 3730-3797 A; 30 = 4245-4327 A; 60 = 5373-5476 A;
  90 = 7315-7455 A; 110 = 9637-9822 A.
* Flux is in electrons, not blaze-corrected; `blaze` is the blaze function; `var` the
  variance. Bluest orders have low S/N (index 10: S/N ~ 3 per pixel; index 30: ~80;
  index 60: ~280; index 90: ~450).
* Wavelengths are vacuum, in the observatory rest frame. Neighbouring orders overlap.
  A few pixels per order are NaN.
* The DRP's telluric model arrays are not available (the flux is already corrected with
  them); `WVAPOR` and `ZENITH` are the model's fitted water column and zenith angle.

## Reference frame -- read this before using spectra

The line-of-sight velocity of the telescope relative to the Sun changes by roughly
100 m/s per hour (Earth rotation) and by up to +-500 m/s over the year (orbital
eccentricity). The DRP's CCF RVs (`rv_pipeline`, `order_rv`) are already in the
heliocentric frame: the DRP (using barycorrpy) removes the observer's motion and the
reflex motion of the Sun caused by all solar-system planets. No real planetary signal
is left in them, so every variation in the un-injected pipeline RVs is solar activity
or instrumental noise. The `ssb_z` value in the spectrum dict is the DRP's barycentric redshift;
`wave * (1 + ssb_z)` removes most, **but not all**, of the observer's motion for solar
data: between two spectra one hour apart, the stellar lines moved by 112 m/s in the
observatory frame while `ssb_z * c` changed by 91 m/s. An algorithm that measures its
own velocities from spectra must therefore remove the residual observer motion
itself, for example by (a) anchoring its per-observation zero point to quantities that
a planet shifts identically (careful: subtracting `rv_pipeline` outright removes the
planet as well), or (b) computing the heliocentric velocity of Kitt Peak
(longitude -111.600 deg, latitude +31.958 deg, altitude 2091 m) from standard
ephemeris formulae and `time`.

## Pipeline CCFs

* G2 ESPRESSO mask, 804 velocity steps of 0.25 km/s from about -100 to +100 km/s,
  per order, in electrons. About 77 orders have a CCF (indices ~11-95); the others are
  zeros. The summed-CCF minimum sits near -0.64 km/s, FWHM about 7.07 km/s,
  BIS about -48 m/s.
* `order_weight` is the DRP's per-order weight; `order_rv` exists for more orders
  than have CCF weight.

## Activity indices available via `ds.activity(name)`

`CaIIHK`, `HeI_1`, `HeI_2`, `NaI`, `Ha06_1`, `Ha06_2`, `Ha16_1`, `Ha16_2`, `CaI_1`,
`CaI_2`, `CaIRT1`, `CaIRT2`, `CaIRT3`, `NaINIR`, `PaDelta`, `Mn539`, each also as
`<name>_tellcorr`, plus `CCFFWHMMOD`, `CCFBISMOD`. Typical per-exposure uncertainties
are 1e-4 to 4e-4.

## Known nuisance signals to expect

* p-modes (~5 min, ~0.5-1 m/s per exposure), granulation (minutes-hours),
  supergranulation (~1-2 days, a few tens of cm/s) -- largely shape-less.
* Rotationally modulated activity (~27 d and harmonics) and its slow evolution over
  the rise of Solar Cycle 25.
* Differential atmospheric extinction across the solar disc (airmass / hour-angle
  dependent, largest at high airmass), telluric contamination varying with water
  column and with the annual +-0.5 km/s frame shift, instrumental drifts,
  detector-level effects, occasional calibration regime changes.

## The toy data used by `run_tests`

Same array shapes and names as above, but a synthetic Sun (random line list,
Gaussian CCFs, simple correlated activity). It is for checking that code runs. It
says nothing about what works on the real Sun.

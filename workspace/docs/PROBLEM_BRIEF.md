# Problem brief: stellar activity in extreme-precision radial velocities

## Goal

Detecting an Earth-mass planet in the habitable zone of a Sun-like star needs
radial-velocity (RV) precision near 10 cm/s over years. Spectrographs now reach
that instrumentally; the star itself does not cooperate. Magnetic activity (spots,
faculae/plage, suppression of convective blueshift), granulation, supergranulation
and oscillations change the shapes and positions of spectral lines and produce
apparent RV variations of 0.5 - several m/s.

We have a curated set of cleaned, vetted Sun-as-a-star spectra from the NEID
spectrograph (WIYN 3.5 m, Kitt Peak; R ~ 110,000; 380-930 nm) spread over about five
years: roughly 500 spectra, each the mean of 16 consecutive exposures (about 20 minutes)
that already integrates over the solar p-mode oscillations, with a photon-noise RV
uncertainty near 6 cm/s, well below the goal. They are
sparse in time (typically one spectrum on an observing day, days apart), like a
realistic survey of another star. The Sun's true centre-of-mass RV in these data is
constant (the pipeline RVs are heliocentric), so every variation is nuisance signal
-- a perfect laboratory. The set may later grow to 1000-2000 spectra.

**Task: find algorithms that turn these spectra into an RV time series whose scatter
is as close to the photon limit as possible -- below 20 cm/s is the goal -- while
leaving real planetary Doppler shifts untouched.**

The scatter is dominated by stellar activity, not photons. A 10 cm/s planet does not
need 10 cm/s per-epoch scatter: with N epochs of scatter sigma, an amplitude of about
5 sigma sqrt(2/N) is detectable in a periodogram (about 6 cm/s for sigma = 20 cm/s
and N = 500). Work on the premise that algorithms reaching the goal exist: the
information that distinguishes a line-shape change from a Doppler shift is in the
spectra. Several different algorithms that reach the goal are wanted, from different
families of ideas.

## Ground rules

1. **Only NEID L2 solar spectra.** Everything an algorithm uses must come from the
   L2 files (spectra, variances, wavelength solutions, blaze, CCFs,
   DRP activity indices, header telemetry). No other observatory, no solar images or
   indices, no pre-trained models.
2. **Planets must survive.** Every evaluation injects hidden low-amplitude planets
   at the spectrum level before the algorithm sees the data and measures how much of
   each survives. Reducing scatter by absorbing signal is worthless and is caught.
3. **The evaluation is not negotiable.** It is run by a fixed script on a cluster.
   Agents may request extra *diagnostics* but never changes to scoring, data
   selection or promotion.
4. **Published sources only, always cited.** Build on the published literature and
   arXiv preprints, with citations. Do not use unpublished or private material.
5. **Everything in writing.** Proposals, critiques, specifications, implementation
   notes, results and post-mortems are human-readable Markdown in `workspace/`.
   Humans monitor these files and may drop suggestions in
   `workspace/human_suggestions/`.

## What "better" means

See `ALGORITHM_INTERFACE.md` section 6. In short: RMS of the corrected RVs over all epochs, divided by the worst planet-amplitude transfer when that is below 1, with a gate
requiring every injected planet to be recovered within 20 % in amplitude.

## The spirit of the collaboration

* Diversity of approach is an explicit goal. A portfolio of genuinely different
  formulations is worth more than many variants of the current front-runner; early
  results are weak evidence about where the ceiling of an idea is.
* Bold, physically or statistically motivated ideas are welcome even when their
  chance of success is modest. A clear mechanism and a clear test matter more than
  caution.
* Criticism is for designing the test that would expose a flaw, and for finding the
  cheapest fix -- not for talking an idea out of being tried. The data decide.
* Negative results are valuable when the post-mortem explains *why*.

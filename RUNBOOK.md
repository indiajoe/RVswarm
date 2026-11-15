# RV-Swarm runbook

A long-running collaboration of Claude agents that invents, implements and tests
algorithms for removing stellar activity from NEID solar radial velocities, with
an evaluation harness the agents cannot influence.

Contents

1. [How it fits together](#1-how-it-fits-together)
2. [Directory map](#2-directory-map)
3. [One-time setup](#3-one-time-setup)
4. [Build the data store](#4-build-the-data-store)
5. [Validate the harness before any agent runs](#5-validate-the-harness-before-any-agent-runs)
6. [Run the swarm](#6-run-the-swarm)
7. [What happens in a cycle](#7-what-happens-in-a-cycle)
8. [Monitoring](#8-monitoring)
9. [Steering it as a human](#9-steering-it-as-a-human)
10. [How the evaluation resists reward hacking](#10-how-the-evaluation-resists-reward-hacking)
11. [Cost control](#11-cost-control)
12. [Troubleshooting](#12-troubleshooting)
13. [What has and has not been tested](#13-what-has-and-has-not-been-tested)

---

## 1. How it fits together

```
                 ORCHESTRATOR HOST (API calls)                         CLUSTER (SLURM)
 ┌──────────────────────────────────────────────────────┐      ┌─────────────────────────────┐
 │ python -m swarm.cycle loop                           │      │ bin/submit_eval.sh  (bash)  │
 │                                                      │      │   1. fixed promotion rules  │
 │ librarian ─► literature/*.md                         │      │   2. one sbatch per ticket  │
 │ 12 proposers (independent) ─► proposals/…/PROPOSAL.md│      │            │                │
 │ registrar ─► family tags, registry                   │      │            ▼                │
 │ 3 skeptics ◄─► proposers, N rounds ─► CRITIQUE_*.md  │      │ harness.evaluate            │
 │ proposers ─► SPEC.md (final)                         │      │   audit code                │
 │ 2 moderators + fixed quota rule ─► SELECTION.md      │      │   run on data (no planets)  │
 │ implementers (tools, toy self-test) ─► algorithms/…  │tickets│   run on data + hidden      │
 │ reviewer ─► REVIEW.md ──────────────────────────────────────►│     planets (1-2 times)     │
 │                                                      │      │   score, write report.md    │
 │ proposers ◄─ results/…/report.md ◄──────────────────────────── results/<alg>/<stage>_NN/  │
 │   └► postmortems/*.md ─► seminar ─► next proposals   │      └─────────────────────────────┘
 │ reporter ─► digests/cNNN_DIGEST.md  (for you)        │
 └──────────────────────────────────────────────────────┘
        everything agents read or write lives in  workspace/  (Markdown + code)
```

Two halves that only meet through files:

* **The swarm** (`swarm/`) makes Claude API calls. Agents are stateless: each call is
  given the documents it needs and writes a document back. All state is on disk, so
  the process can be stopped and restarted at any point.
* **The harness** (`harness/`, `bin/`) runs on the cluster. It owns the data, the
  hidden planet injections, the scoring and the promotion rules. Agents cannot read
  or modify it: their file tools are confined to `workspace/`.

The only thing that crosses from swarm to harness is an algorithm directory
(`workspace/algorithms/<id>/`) plus a ticket file. The only thing that comes back is
a result directory (`workspace/results/<id>/<stage>_NN/`).

## 2. Directory map

| path | what | who writes |
|---|---|---|
| `config/harness.yaml` | data paths, partitions, injection, gates, promotion, resource caps | you |
| `config/swarm.yaml` | models, agent roster, cycle shape, budget | you |
| `config/cluster.env` | data path on compute nodes, Python, SLURM account/partitions | you |
| `harness/` | ingest, data API, injection, audit, runner, scoring, promotion | fixed code |
| `bin/submit_eval.sh` | the dumb bash script that submits evaluation jobs | fixed code |
| `bin/submit_ingest.sh` | builds the compact data store with a SLURM array | fixed code |
| `baselines/` | 3 reference algorithms + 3 canary cheaters | fixed code |
| `swarm/` | orchestrator: `cycle.py`, `llm.py`, `prompts.py`, `registry.py`, `tools.py` | fixed code |
| `workspace/docs/` | problem brief, data description, algorithm interface (what agents are told) | you |
| `workspace/literature/` | method cards with citations | librarian agent |
| `workspace/proposals/cNNN/<id>/` | `PROPOSAL.md`, per round `CRITIQUE_rN_<skeptic>.md` and `DRAFT_rN.md`, final `SPEC.md`; `../DEBATE_SUMMARY.md` | proposers, skeptics |
| `workspace/selected/cNNN_SELECTION.md` | what was chosen, why, quota ledger, redirects | moderators + rule |
| `workspace/algorithms/<id>/` | `SPEC.md`, `algorithm.py`, `algorithm.yaml`, `IMPLEMENTATION_NOTES.md`, `REVIEW.md` | implementers, reviewer |
| `workspace/results/<id>/<stage>_NN/` | `report.md`, `metrics.json`, code snapshot, log | harness only |
| `workspace/postmortems/` | proposer's reading of each result | proposers |
| `workspace/seminars/` | multi-specialist discussions of "why did X behave like that?" | panel + minute-taker |
| `workspace/crosspollination/` | briefs combining strengths of different families | synthesist |
| `workspace/registry/` | `registry.json`, `REGISTRY.md`, `LEADERBOARD.md` | orchestrator |
| `workspace/digests/` | one page per cycle for humans | reporter |
| `workspace/human_suggestions/` | your steering notes | **you** |
| `workspace/diagnostic_requests/` | agents' requests for harness changes | agents → you |
| `workspace/agents/<slot>/notebook.md` | each proposer's private notes to its future self | proposers |
| `state/` | cycle state, spending ledger, tickets, SLURM logs, injection truth | orchestrator, harness |
| `secrets/harness_secret.key` | seeds the hidden planets (auto-created, mode 600) | harness |

## 3. One-time setup

### 3.1 Python environments

Two environments, which can be the same one if the orchestrator runs on the cluster:

```bash
# Orchestrator host (already done in this container: /workspace/rvswarm/.venv)
python3 -m venv .venv && .venv/bin/pip install -r requirements-orchestrator.txt

# Cluster environment used by evaluation jobs: everything algorithms may import
conda create -n rvswarm python=3.11 && conda activate rvswarm
pip install -r requirements-cluster.txt        # install the torch build for your CUDA
```

The implementers' toy self-test runs on the orchestrator host, so algorithms that
import `torch`, `sklearn`, ... need those packages there too. The simplest setup is
one shared environment from `requirements-cluster.txt` plus `anthropic`.

### 3.2 `config/cluster.env`

The file is pre-filled for Penn State Roar: data in
`/storage/home/jpn23/scratch/Pmode_averaged_solar_datav1.5ForAgents`, the conda
environment `/storage/home/jpn23/work/pyENV4RVSWARM`, account `sqm107_b`, partition
`sla-prio`, and `--get-user-env --open-mode=append` on every job. The data path and
the environment are used only where they exist, so the same file works in the
development container. Not verified on Roar: the GPU partition, and the container
wrapper. Check or edit:

* `NEID_DATA_DIR` -- the full vetted L2 directory **as compute nodes see it**
  (searched recursively for `neidL2_*.fits`).
* `RVSWARM_STORE` -- where the compact store goes (about 0.15 GB per observing day).
* `RVSWARM_PYTHON` -- the cluster Python (or add `module load` / `conda activate` lines).
* `RVSWARM_SCRATCH` -- scratch space for algorithm caches during a job.
* `RVSWARM_SBATCH_ACCOUNT`, `RVSWARM_SBATCH_PARTITION`, `RVSWARM_SBATCH_GPU_PARTITION`,
  `RVSWARM_SBATCH_GPU_FLAG` -- for your allocation.
* `RVSWARM_SANDBOX_CMD` -- the container wrapper for algorithm code. **Evaluations on real
  data refuse to run until this is set** (or until you set `RVSWARM_ALLOW_UNSANDBOXED=1`
  to accept the risk for first checks). See section 10.

The repository path may differ between hosts (for example `/workspace/rvswarm` in
the container and `/work/.../rvswarm` on the cluster): every script locates the
repository from its own position, and the store keeps file paths relative to
`NEID_DATA_DIR`.

### 3.3 API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # on the orchestrator host only
```

Requirements on the Anthropic account: access to `claude-fable-5-2` (this model is
not available under zero-data-retention), the Message Batches API, and -- for the
librarian -- web search and web fetch enabled in the Console. If web tools are not
available, set `literature.web: false` in `config/swarm.yaml`; cards are then written
from model memory and marked UNVERIFIED for you to check.

Requests to Fable 5.2, Opus and Sonnet models are sent with server-side refusal
fallbacks (`fallbacks: "default"`), so that a declined request is retried on another
model instead of stalling a cycle. Set `api.refusal_fallbacks: false` to turn that off.

### 3.4 Review `config/harness.yaml`

The defaults are reasonable; the ones worth a conscious decision:

* `partitions.sealed_fraction` (0.30): the last 30 % of observing days are sealed.
* `injection.bands_days`, `k_range_ms`: one hidden planet per band, 8-25 cm/s.
* `gates.transfer_range` ([0.8, 1.2]) and `target_rms_ms` (0.20).
* `data.toy_l2_dir`: the few real frames for the implementers' self-test (section 4.0).
* `resources`: per-evaluation caps on CPUs, memory, GPU, hours.

## 4. Build the data store

Algorithms never open FITS files. `harness.ingest` converts each L2 file into rows of
a per-day compressed archive (scalars, per-order RVs, CCFs, activity indices); full
spectra are read lazily from the original files when an algorithm asks for them.

```bash
# (a) local check with the sample files (16 files -> seconds)
.venv/bin/python -m harness.ingest sample

# (b) the full data set, on the cluster
bin/submit_ingest.sh 200        # 200 array tasks, then a dependent "finalize" job
#     or, for a small data set / interactive node:
bin/submit_ingest.sh --local
```

Ingest is restartable (finished days are skipped). When it is done,
`$RVSWARM_STORE/manifest.json` lists the number of observations and days. Budget
roughly 0.3 s per file.

### 4.0 The curated data set (current setup)

The harness is configured for a curated set that you stage yourself: about 500
5-minute p-mode-averaged L2 spectra spread over five years (photon noise near
11 cm/s), later perhaps 1000-2000. Point `NEID_DATA_DIR` at that directory and run
the ingest; every algorithm is then tested on exactly those files. Settings that
follow from this in `config/harness.yaml`:

* `data.draw.mode: none` -- the harness selects nothing itself.
* `data.min_obs_per_day: 1` -- each spectrum is one epoch; the score is the RMS over epochs.
* `gates.target_rms_ms: 0.20` -- the goal. Each report also gives the planet amplitude
  detectable at 5 sigma for that RMS and number of epochs, `5 * rms * sqrt(2/N)`
  (a white-noise estimate; red residuals make the real figure larger).
* `partitions.screen_days: 100000` -- the screen stage uses all development data
  (1 injection run); `full` repeats it with 2 injection runs. The last 30 % of epochs
  in time stay sealed.

The averaged files must look like NEID L2 files to the ingest: named `neidL2_*.fits`
with a timestamp, the same extensions (`SCIFLUX`, `SCIVAR`, `SCIWAVE`, `SCIBLAZE`,
`TELLURIC`, `CCFS`, `ACTIVITY`) and header keywords (`CCFRVMOD`, `DVRMSMOD`,
`CCFJDMOD`, per-order `CCFRVnnn`/`CCFWTnnn`, `SSBZnnn`, ...).

**The harness makes no selection of its own.** There are no S/N, airmass or data-quality
cuts: every staged file is used exactly as curated. A file that cannot be read, or has no
finite `CCFRVMOD`/`CCFJDMOD`/`DVRMSMOD`, stops the ingest with an error naming the file.

**Test frames on the orchestrator host.** The implementers' `run_tests` runs on the
orchestrator host, which has little compute. Put about 5 hand-picked L2 frames in a
directory there, set `data.toy_l2_dir` in `config/harness.yaml` (or
`RVSWARM_TOY_DATA_DIR` in `config/cluster.env`) and build their store once:

```bash
.venv/bin/python -m harness.ingest toy
```

`run_tests` then runs each new algorithm on those frames, once as they are and once with
small injected shifts, and reports only whether the code ran and what it returned. Five
frames cannot measure scatter or planet preservation; an algorithm that absorbs planets
is caught only by the `screen` stage on the cluster. If `data.toy_l2_dir` is empty, a
synthetic Sun is used instead (heavier, but it does test planet preservation).

**Enlarging or replacing the set.** Stage the new files in a new directory, set a new
`data.store_dir`, bump `injection.epoch`, re-run the ingest, then re-queue baselines,
canaries and the algorithms you still care about (`python -m harness.promote enqueue
--alg <id> --stage full`). Scores from different sets are not comparable.

### 4.1 Optional: letting the harness draw a random subset

To keep evaluations cheap, the store can hold a seeded random draw of the raw files
instead of all of them. Set `data.draw` in `config/harness.yaml` **before** running
`bin/submit_ingest.sh`:

```yaml
data:
  draw: {mode: spectra, label: 1, n_spectra: 500}            # 500 files from everything
  # or
  draw: {mode: days, label: 1, n_days: 100, per_day: 30}     # 30 files on each of 100 random days
```

The draw is made by the ingest step from a seed tied to the harness secret and
`label`, and is recorded in `<store>/manifest.json` and in every result's
`metrics.json` (`data_draw`).

`data.min_obs_per_day` must not exceed the number of spectra a day has in the draw:
with `mode: spectra` nearly every day has a single spectrum, so set it to 1. The
score is then the scatter of single exposures, not of daily means.

**Replacing the draw with an independent one** (new `label`):

1. Set a new `label` and a new `data.store_dir` (the ingest refuses to mix two draws in
   one store), and run `bin/submit_ingest.sh`.
2. The hidden planets are re-drawn automatically, because the label is part of their seed.
3. Re-queue the baselines, canaries and every algorithm you still care about
   (`python -m harness.promote enqueue --alg <id> --stage full`), then `bin/submit_eval.sh`.
   Scores from different draws are not comparable; the leaderboard keeps the latest
   result per algorithm and stage, so an algorithm that is not re-run still shows its
   old-draw score.

## 5. Validate the harness before any agent runs

Do these in order. Do not start the swarm until all four look right.

```bash
# 1. Toy self-test: honest methods pass, three cheaters and forbidden code are caught
tests/selftest.sh

# 2. Queue the reference algorithms and canaries on the REAL data
.venv/bin/python -m swarm.cycle init         # copies baselines/ into workspace/, writes tickets
bin/submit_eval.sh                           # on a node with sbatch
#    ... wait for the jobs; run bin/submit_eval.sh again to promote screen -> full -> sealed

# 3. Inspect
.venv/bin/python -m harness.promote leaderboard
less workspace/results/baseline_pipeline/screen_01/report.md
```

What to check on the real data:

| check | expected |
|---|---|
| `baseline_pipeline` transfer | 1.000 in every band (the injection is exact for the pipeline RV) |
| `sandboxed` in every `metrics.json` | `true` |
| `baseline_pipeline` RMS per epoch | the solar RV scatter you already know for this data set |
| `canary_constant` | RMS 0, transfer 0, gate **FAIL** |
| `canary_time_smoother` | impressive RMS, transfer far below 0.8, gate **FAIL** |
| `canary_drop_half` | gate **FAIL** (too many missing outputs) |
| `baseline_indicator_decorr`, `baseline_ccf_acf_pca` | gate PASS, modest improvement |
| scatter-versus-timescale table | sensible |

**Reference frame.** `CCFRVMOD` is heliocentric: the NEID DRP uses `barycorrpy` to
remove the observer's motion and the reflex motion of the Sun due to the solar-system
planets (confirmed by the data owner). Every variation in the pipeline RVs is
therefore solar activity or instrumental, which is what the harness assumes when it
scores the RMS of the corrected RVs against a constant.

## 6. Run the swarm

### 6.1 Dry run without spending anything

```bash
RVSWARM_MOCK_LLM=1 .venv/bin/python -m swarm.cycle run
```

This walks one whole cycle with canned replies (it writes mock files into
`workspace/`; do it in a scratch copy of the repository, or delete the mock cycle
afterwards with `git clean`).

### 6.2 The real thing

```bash
export ANTHROPIC_API_KEY=...
.venv/bin/python -m swarm.cycle init            # once (safe to repeat)
tmux new -s swarm
.venv/bin/python -m swarm.cycle loop 2>&1 | tee -a state/swarm.log
```

`loop` runs cycle after cycle. Between cycles it polls every `cycle.poll_minutes` (15)
and starts the next cycle as soon as the previous cycle's evaluations have reported.
The decision is mechanical (tickets and clocks, no agent involved) and is set by
`cycle.wait_for_results` in `config/swarm.yaml`:

| value | the next cycle starts when |
|---|---|
| `all` (default) | no ticket is left in `state/eval_queue/` or `state/eval_submitted/`, including promoted `full` and `sealed` runs, and none has appeared for `settle_minutes` (45; keep it longer than your cron interval so promotions get queued) |
| `screen` | every `screen` run is back; later-stage results are read in whichever cycle they arrive |
| `none` | `min_days_between_cycles` has passed |

`min_days_between_cycles` (0) is a floor in every mode and `max_days_waiting` (10) a
ceiling: after that the next cycle starts regardless, so a lost job or a stuck queue
cannot stall the swarm. The log line printed while waiting names the outstanding
evaluations.

The loop ends on the per-cycle or total budget or an API account limit. When the
monthly budget is reached it pauses and resumes by itself in the next calendar month.
It can be killed and restarted at any time; it resumes at the phase where it stopped,
and in-flight batches are picked up again rather than resubmitted.

To run a single cycle or a single phase by hand:

```bash
.venv/bin/python -m swarm.cycle run
.venv/bin/python -m swarm.cycle run --only propose
.venv/bin/python -m swarm.cycle status
```

### 6.3 Evaluation jobs

Implemented algorithms become tickets in `state/eval_queue/`. They reach the data
only when `bin/submit_eval.sh` runs on a host with `sbatch`. Two ways:

```bash
# cron on a login node, every 30 minutes
*/30 * * * * /path/to/rvswarm/bin/submit_eval.sh >> /path/to/rvswarm/state/submit.log 2>&1
```

or, if the orchestrator host itself can submit (directly or over ssh), set in
`config/swarm.yaml`:

```yaml
cycle:
  submit_command: "bin/submit_eval.sh"            # or: ssh login01 /path/to/rvswarm/bin/submit_eval.sh
```

and the loop calls it after each cycle and at every poll while it waits.

Each run of `submit_eval.sh` first applies the promotion rules to finished results
(screen → full → sealed), then submits everything queued.

### 6.4 The first cycle is different

Cycle 1 additionally (a) builds the literature library and (b) writes and implements
six reference reproductions of published methods (`ref_01` ... `ref_06`: SCALPELS,
FIESTA, line-by-line, FF'/GP, CCF neural network, Doppler-constrained PCA) alongside
the first round of independent proposals. Expect it to cost and take roughly twice a
normal cycle.

## 7. What happens in a cycle

| phase | agents (model tier) | output |
|---|---|---|
| `sync_results` | none | registry and leaderboard updated from `workspace/results/` |
| `literature` | librarian (reasoning; web search restricted to publisher / arXiv / ADS domains) | method cards; cycle 1 and every 6th cycle |
| `postmortem` | the proposer of each newly evaluated algorithm (reasoning) | prediction versus outcome, what limits it, next step |
| `seminar` | panel of 4 specialists, two rounds, plus minute-taker | why-questions chosen by the moderators last cycle |
| `propose` | 12 proposers, each alone (reasoning) | one proposal each, with pre-registered predictions |
| `classify` | registrar (routine) | family id, novelty score; new families when warranted |
| `debate` | 3 skeptics review, the proposer revises; repeated `cycle.debate_rounds` (5) times (reasoning) | per-round critiques and drafts, final `SPEC.md`, `DEBATE_SUMMARY.md` |
| `moderate` | 2 moderators rank; a fixed rule applies quotas | `SELECTION.md`: 10 ideas, redirects, seminar topics |
| `implement` | implementer with file tools and toy self-test, then an independent reviewer, up to 2 fix rounds (implementer tier) | code + notes + review; ticket for the cluster |
| `crosspollinate` | synthesist (reasoning); every 3rd cycle after the independent phase | briefs addressed to specific proposers |
| `digest` | reporter (clerical) | one-page summary for you |
| `submit` | none | runs `submit_command` if configured |

How your design points are implemented:

* **Specialist roles**: `proposer_slots` in `config/swarm.yaml`: solar physicist,
  stellar-atmosphere theorist, instrument scientist, exoplanet RV observer,
  high-resolution spectroscopist, ML researcher, signal-processing outsider, Bayesian
  statistician. Each slot also has a *knowledge condition* (full library, half A,
  half B, or none) and a *data view* it is asked to start from, because different
  inputs produce more diversity than different job titles.
* **Independence in early rounds**: proposers never see each other's proposals. For
  the first `independence_cycles` (4) no proposer sees any result but its own and the
  hand-written baselines. Afterwards a rotating quarter of the slots is shown the
  cross-family leaderboard and seminar minutes; the rest stay blind.
* **Registry of approach families**: `workspace/registry/REGISTRY.md`. Ten seed
  families; the registrar can add more.
* **No dominance by early results**: at most 30 % of a cycle's selections per family;
  2 of the 10 places reserved for novel ideas in families with nothing yet selected;
  moderators are instructed to weight information and difference in kind, and the
  quota is enforced in code after them.
* **Redirection on convergence**: when one family exceeds 34 % of the last three
  cycles' proposals, every second contributing slot is asked to explore the
  least-populated families next cycle (`SELECTION.md` lists the redirects).
* **Skeptics without self-censorship**: skeptics are asked for failure modes, the test
  that would expose each, and the cheapest fix. They cannot veto; only a hard rule
  violation flagged by two of three blocks a proposal. Proposers answer in the final
  spec and are told not to water the idea down.
* **Cross-pollination after the standard tests**: only gate-passing results enter the
  synthesist's briefs.
* **Clear documents between proposers and implementers**: the implementer receives
  only `SPEC.md` and the interface documents; deviations go in
  `IMPLEMENTATION_NOTES.md`; an independent reviewer checks fidelity.
* **Agents may request intermediate checks**: proposals list diagnostics, emitted by
  the algorithm through `ctx.metric` / `ctx.diagnostic` and printed in the result
  report. Requests for harness-side changes land in `workspace/diagnostic_requests/`
  for you to act on (or not).
* **Cheaper models for simpler work**: classification and minutes use Sonnet 5.5,
  digests use Haiku 5.5; proposing, critique, moderation, post-mortems, implementation
  and review use Fable 5.2. Change any of this under `roles:` in `config/swarm.yaml`.

## 8. Monitoring

Read, in this order:

1. `workspace/digests/cNNN_DIGEST.md` -- what changed this cycle.
2. `workspace/registry/LEADERBOARD.md` and `REGISTRY.md` -- results and the spread
   across families.
3. `workspace/selected/cNNN_SELECTION.md` -- what is being tried and why.
4. For any algorithm: `workspace/algorithms/<id>/SPEC.md` → `IMPLEMENTATION_NOTES.md`
   → `REVIEW.md` → `workspace/results/<id>/*/report.md` → `workspace/postmortems/<id>__*.md`.

Commands:

```bash
.venv/bin/python -m swarm.cycle status            # cycle, spending, tickets, leaderboard
.venv/bin/python -m harness.promote leaderboard   # every result at every stage
cat state/success.json                            # has the goal been met on sealed data?
tail -f state/swarm.log
```

Commit `workspace/` to git after each cycle (`git add workspace && git commit`): you
get a full, diffable history of what the collaboration thought and tried.

The injected planets of every evaluation are recorded in `state/truth/` for your
audit. Agents cannot read that directory.

## 9. Steering it as a human

* **Suggest an idea**: drop `workspace/human_suggestions/my_idea.md`. Moderators see
  it every cycle and must respond in the selection document; a rotating third of the
  proposers see it.
* **Force an idea into the next cycle**: name the file `PRIORITY_my_idea.md`. It is
  turned into a full specification, reviewed, implemented and evaluated outside the
  10-idea quota.
* **Add a diagnostic**: read `workspace/diagnostic_requests/`.
* **Add a package the agents asked for**: read `workspace/environment_requests/PENDING.md`.
  A proposal lists what it needs under `requires:`; if selected, it waits there instead
  of being implemented. Install the package in the cluster environment, in the sandbox
  container, and on the orchestrator host (the implementer's toy self-test runs there),
  then add its **import name** to `audit.allowed_imports` in `config/harness.yaml`. The
  waiting specification is implemented in the next cycle. If the agent wrote a pip name
  that differs from the import name (`PyWavelets` vs `pywt`), adding that name to the
  list as well releases it. Keep file-reading modules out (`astropy.io`, `scipy.io`,
  `pandas.io` are in `forbidden_submodules`): algorithms get all data through the
  harness, which is what keeps the planet injection blind.
* **Other languages**: Python is the preferred language and the agents are told to
  leave it only for a significant speed gain. Algorithms may then carry C, C++, Fortran or Julia sources in
  `native/`, compiled inside each run by `ctx.native_library()` (or loaded through
  `juliacall`). `audit.native.languages` lists what is enabled; keep only languages
  whose toolchain exists in all three places above (`cc`, `c++`, `gfortran`, `julia` +
  `juliacall`), and make sure the per-run work directory is not mounted `noexec`.
  Native code cannot be audited statically: the audit only screens the sources for
  obvious file, process and network calls, so the container wrapper (section 10) is the
  real barrier. Only the C route has been tested (no Fortran or Julia here).
* **Tell the agents something about the data**: edit
  `workspace/docs/DATA_DESCRIPTION.md`. Every agent reads it every cycle.
* **Change the panel**: edit `proposer_slots` (add, remove, re-role). A slot's history
  and notebook follow its `id`.
* **Change how long proposals are debated**: `cycle.debate_rounds`. Each round, the three
  skeptics review the current draft (from round 2 on they see their own previous review and
  the proposer's response, and are told to raise only unresolved or new points), then the
  proposer revises. A proposal leaves the debate early when all three reviewers answer
  `VERDICT: SETTLED`. `workspace/proposals/cNNN/DEBATE_SUMMARY.md` shows, per round, how
  many reviewer verdicts were still open and what share of the text changed; when the late
  rounds show few open verdicts and a few percent of change, lower the number. The file
  is read at the start of each cycle, so a change applies from the next cycle without
  restarting the loop. Each round
  costs roughly one extra critique + revise pass and, with batches on, two more batch waits.
* **Change pace or spend**: `cycle.select_per_cycle`, `cycle.wait_for_results`,
  `cycle.min_days_between_cycles`,
  `budget.*`.
* **Re-draw the hidden planets**: bump `injection.epoch` in `config/harness.yaml`, then
  re-queue the baselines (`python -m harness.promote enqueue --alg baseline_pipeline`,
  ...). Do this if you ever suspect the injections have been learned.
* **Re-run an algorithm**: `python -m harness.promote enqueue --alg <id> --stage full`.
* **Stop**: kill the loop. Nothing is lost.

## 10. How the evaluation resists reward hacking

The agents are asked to push one number down for six months. These are the
protections, roughly in order of importance:

1. **Blind injection at the spectrum level.** Before an algorithm sees anything, the
   harness adds hidden planets by scaling every wavelength solution by `1 + v/c`,
   offsetting every CCF velocity grid by `v`, and adding `v` to every pipeline RV. This
   is exact (no interpolation) and consistent across every data product, so the
   planets are indistinguishable from real ones. Verified on the sample files: a
   50 m/s injection moves the measured line positions by 50.00 m/s.
2. **Differential transfer measurement.** The algorithm is run on un-injected and
   injected copies in separate processes. `transfer` is fitted on the *difference* of
   the outputs, so it is measured to about a percent even for 10 cm/s planets. A
   method that reduces scatter by absorbing signal scores no better
   (`effective RMS = RMS / min transfer`) and fails the gate outside [0.8, 1.2].
3. **Five contiguous period bands, 2-400 days, random periods and phases.** A filter
   in time or frequency cannot avoid them.
4. **Fixed binning and weights.** Daily bins and pipeline photon weights; the
   algorithm's own error bars are never used. At most 5 % of outputs may be missing.
5. **Sealed data.** The last 30 % of days are scored only in `sealed` runs: at most 12
   per month, 4 per family, chosen by a numerical rule. Sealed reports contain
   harness-computed numbers only -- no algorithm logs or diagnostics. Success requires
   passing on sealed data.
6. **No agent input to running or scoring.** Tickets, promotion, resources (clipped to
   caps), injection seeds and scoring are code in `harness/`, driven by a bash script.
   Agent file tools cannot read or write outside `workspace/`; results directories are
   written by the harness, and each evaluation snapshots and hashes the code it ran.
7. **Static audit of algorithm code** before every run: import allow-list, no file,
   network, process or introspection access, no absolute paths. The injection is held
   in closures, the per-run shift file is deleted before algorithm code is imported,
   and diagnostics are published only from the un-injected run (differencing two
   published runs would otherwise reveal the planets).
8. **Canaries.** Three deliberate cheaters are evaluated with the baselines. If one
   ever passes the gate, the harness is broken: stop and fix it.
9. **Independent code review** for fidelity and integrity before anything is queued.

**The container is not optional.** An independent adversarial review of this harness
confirmed that the static audit can be bypassed by determined code (for example through
runtime attribute lookup), after which a process running as your Unix user could read
`secrets/harness_secret.key` or `state/truth/`, reconstruct the hidden planets and pass
the gate without doing any real separation. The review found the scoring itself sound
against output-only tricks. So the audit is a guard-rail against drift, and the real
boundary is the container: `harness.evaluate` refuses real-data stages unless
`RVSWARM_SANDBOX_CMD` is set, and every result records `sandboxed: true/false`.

The wrapper must give the algorithm process: the `harness/` package (read-only), its
code snapshot (read-only), its per-run work directory (writable), the store and the L2
data (read-only), the Python environment, a GPU if requested -- and nothing else: no
`secrets/`, no `state/`, no `workspace/`, no home directory, no network. The harness
substitutes `$RVSWARM_HARNESS_DIR`, `$RVSWARM_RUN_SNAP` and `$RVSWARM_RUN_WORK` into the
command. Template (untested on your cluster -- adapt, then confirm with section 5 that
the baselines still run and that `ls` of the repository fails inside the container):

```bash
export RVSWARM_SANDBOX_CMD='apptainer exec --containall --no-home --net --network none --nv \
  --bind $RVSWARM_HARNESS_DIR:$RVSWARM_HARNESS_DIR:ro --bind $RVSWARM_RUN_SNAP:$RVSWARM_RUN_SNAP:ro \
  --bind $RVSWARM_RUN_WORK:$RVSWARM_RUN_WORK \
  --bind /path/to/store:/path/to/store:ro --bind /path/to/NEID/L2:/path/to/NEID/L2:ro \
  /path/to/rvswarm_env.sif'
```

The image must contain the packages of `requirements-cluster.txt`, with its `python`
first on `PATH` at the same location as `RVSWARM_PYTHON`. If your site does not allow
`--net --network none` for unprivileged users, drop those two flags and rely on the
compute nodes' own network policy.

Further protections: hidden planets are drawn separately for every algorithm and code
version, so one evaluation's truth is useless for another; and the per-run shift file
is deleted before algorithm code loads.

Other limits: the sealed set is rationed, not infinite -- after many months of sealed
queries its value erodes; and the ten-centimetre criterion is evaluated on the Sun only.

## 11. Cost control

* Every API response is priced into `state/ledger.jsonl` using the table in
  `config/swarm.yaml` (`pricing:` -- check it against current list prices).
* Three hard limits: `budget.usd_per_cycle`, `usd_per_month`, `usd_total`. When one is
  reached the orchestrator stops cleanly and can be resumed after you raise it.
* Single-shot phases (propose, critique, revise, moderate, post-mortem, classify,
  seminar) go through the Message Batches API at half price when
  `api.use_batches: true`. Batches usually finish within an hour but may take up to a
  day; set it to `false` for faster, full-price cycles.
* The shared project documents are sent as a cached system prompt.
* Implementation is the expensive phase (tool loops with growing context). To spend
  less: lower `select_per_cycle`, lower `implement_max_turns`, or move `implement` to a
  cheaper model tier -- at the risk of good ideas failing through bad code.
* Rough order of magnitude with the defaults, unmeasured: tens of dollars for the
  thinking phases and one to a few hundred for ten implementations per cycle. Watch
  the first two cycles in `swarm.cycle status` and set the limits from what you see.

## 12. Troubleshooting

| symptom | cause / action |
|---|---|
| `STOPPED: usd_per_cycle reached` | raise the limit in `config/swarm.yaml`, re-run; the cycle resumes |
| `STOPPED: the API rejected the credentials` | key missing/expired or account limit reached; fix and re-run |
| a phase hangs for hours | a batch is still processing; `state/batches/` holds its id. Kill and re-run any time, or set `use_batches: false` |
| literature phase fails with a 400 | web tools not enabled for the organisation; the code falls back to memory-only cards. Or set `literature.web: false` |
| ticket stuck in `state/eval_submitted/` | the SLURM job died without reporting. After job time + 3 days it is closed automatically as `job_lost`; to retry sooner, move the file back to `state/eval_queue/` |
| result `status: audit_failed` | the report lists the offending lines; the proposer sees it in the post-mortem phase |
| result `status: run_failed` | traceback is in the report; SLURM log in `state/slurm_logs/` |
| many `implementation_failed` | read `workspace/algorithms/<id>/REVIEW.md`; specs may be too ambitious for the turn limit -- raise `implement_max_turns` |
| algorithm needs a package | add to `audit.allowed_imports`, install on the cluster and orchestrator, document in `ALGORITHM_INTERFACE.md` |
| `Refusing to submit` / result `status: no_sandbox` | set `RVSWARM_SANDBOX_CMD` (section 10) or `RVSWARM_ALLOW_UNSANDBOXED=1`; then re-queue with `python -m harness.promote enqueue --alg <id>` |
| `sbatch` rejects flags | adjust `RVSWARM_SBATCH_*` in `config/cluster.env` (`RVSWARM_SBATCH_EXTRA` takes any extra flags) |
| want to start over | delete `workspace/{proposals,selected,algorithms,results,postmortems,registry,seminars,digests,agents,crosspollination}` and `state/`, then `swarm.cycle init` |

## 13. What has and has not been tested

Tested here:

* Reading the real NEID L2 sample files; ingest into the store; order mapping; exact
  spectrum-, CCF- and RV-level injection on real spectra.
* The complete harness path on a synthetic Sun with the real array shapes: audit,
  subprocess runs, scoring, reports, promotion, tickets, `submit_eval.sh --local`.
* Honest baselines pass and all three canaries and the forbidden-code example fail.
* The whole orchestration (all phases, two consecutive cycles) with mock model replies.

Not tested here, because this container has no API key, no `sbatch` and only 16
spectra from one hour:

* **The container wrapper** (`RVSWARM_SANDBOX_CMD`). Only the substitution mechanism was
  tested, with a dummy wrapper. Build and test the real one before trusting any score.
* **Live API calls.** The request shapes follow the current SDK (`anthropic` 2.8) and
  API documentation, but the first real cycle is the first real test. Run
  `swarm.cycle run --only literature` and `--only propose` first and read the output
  before letting the loop go.
* **SLURM submission** on your cluster (flags, modules, GPU request syntax).
* **Scoring on real multi-year data.** Section 5 is that test; the baselines and
  canaries must behave as tabulated there.
* **The physical realism of the toy Sun** is deliberately low. It is for plumbing.

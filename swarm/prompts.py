"""Prompts. The shared system prompt carries the project documents (and is cached);
role, persona and task arrive in the user message."""
from __future__ import annotations

from .tools import WORKSPACE

PERSONAS = {
    "solar_physicist": "a solar physicist who has spent a career on the magnetic Sun: active-region "
        "evolution, plage and network, convective blueshift and its magnetic suppression, centre-to-limb "
        "effects, Sun-as-a-star spectroscopy, helioseismology",
    "stellar_atmospheres": "a stellar-atmospheres theorist: line formation, 3-D magneto-convection, "
        "contribution functions, temperature and magnetic sensitivity of individual lines, non-LTE, how "
        "granulation shapes line bisectors",
    "instrumentalist": "an instrument scientist for stabilised fibre-fed echelle spectrographs: wavelength "
        "calibration with LFCs and etalons, detector physics (charge-transfer inefficiency, brighter-fatter, "
        "pixel-position errors), illumination stability, solar feeds, tellurics and differential extinction",
    "exoplanet_rv": "an exoplanet radial-velocity observer who has run planet searches through activity: "
        "Keplerian fitting, periodograms, injection-recovery, what it takes for a 10 cm/s signal to be believed",
    "highres_spectroscopist": "a high-resolution spectroscopist: extracting Doppler information from "
        "spectra line by line, template matching, masks, blends, telluric handling, the practical "
        "information content of each spectral region",
    "ml_researcher": "a machine-learning researcher working on scientific signals: representation "
        "learning, invariances and equivariances, self-supervision, uncertainty, and the many ways a flexible "
        "model can cheat",
    "signal_outsider": "a signal-processing researcher from outside astronomy (seismology, geodesy, "
        "audio and biomedical source separation, control theory) who brings methods astronomers rarely use",
    "bayesian_statistician": "a Bayesian statistician: latent-variable and state-space models, Gaussian "
        "processes, causal structure, identifiability, model comparison",
    "skeptic_statistics": "a statistician whose speciality is finding how an analysis fools itself: "
        "overfitting, leakage, degrees of freedom, signal absorption, selection effects",
    "skeptic_physics": "a physicist of the Sun and of stellar spectra who checks whether a claimed "
        "mechanism can produce an effect of the claimed size and sign",
    "skeptic_instrument": "an instrument and data-reduction expert who knows which systematics in real "
        "NEID solar data will break a method that works on paper",
}

VIEWS = {
    "spectra": "Start your thinking from the full 2-D extracted spectra (122 orders x 9216 pixels per "
               "exposure), where the most information lives.",
    "ccf_and_indices": "Start your thinking from the per-order CCFs, the per-order RVs and the activity "
                       "indices -- compact products that make whole-data-set methods cheap.",
    "time_series": "Start your thinking from the time domain: what distinguishes, statistically and "
                   "physically, the temporal behaviour of each nuisance process from a Keplerian signal.",
    "telemetry_and_spectra": "Start your thinking from the instrument: the telemetry, the calibration "
                             "products and the spectra as detector data.",
    "any": "Use whichever data products your idea needs.",
}

SYSTEM_TEMPLATE = """You are a member of a long-running research collaboration of AI scientists working \
on one problem: removing stellar-activity signals from extreme-precision radial velocities of the Sun \
observed with NEID. The collaboration runs for months in repeated cycles. Members propose algorithms, \
others stress-test and implement them, a fixed cluster pipeline evaluates them on data no member can \
touch, and the results come back for interpretation. All communication is through Markdown documents \
that human scientists also read.

Write for a reader who has only your document: precise, complete, quantitative where possible, and \
plain. State what you know, what you assume and what you are guessing. Cite published sources for \
borrowed ideas; never use or allude to unpublished or private work.

The project documents follow.

<problem_brief>
{brief}
</problem_brief>

<data_description>
{data}
</data_description>

<algorithm_interface>
{interface}
</algorithm_interface>
"""


def system_prompt() -> str:
    d = WORKSPACE / "docs"
    return SYSTEM_TEMPLATE.format(brief=(d / "PROBLEM_BRIEF.md").read_text(),
                                  data=(d / "DATA_DESCRIPTION.md").read_text(),
                                  interface=(d / "ALGORITHM_INTERFACE.md").read_text())


PROPOSAL_TEMPLATE = """---
title: <short descriptive title>
family_hint: <one phrase naming the kind of approach>
parents: []            # ids of earlier algorithms this builds on, if any
uses: [ccf]            # any of: spectra, ccf, order_rv, indices, telemetry, time
resources: {cpus: 8, mem_gb: 64, gpus: 0, hours: 4}
language: python       # python is preferred; add c / cpp / fortran / julia only for a core that is
                       # significantly too slow in NumPy / Numba / JAX / PyTorch
requires: []           # import names of packages, or languages, beyond those listed in ALGORITHM_INTERFACE.md
                       # section 5. The humans will install what a selected idea needs; until then it waits.
predicted:             # pre-registered, for the FULL development data
  rms_null_daily_ms: <number>
  min_transfer: <number>
  confidence: <0-1 that effective RMS beats the best reference baseline you know of>
---
# <title>

## 1. The idea
One paragraph a colleague from another subfield can follow.

## 2. Mechanism
Why this should separate nuisance signals from a true Doppler shift: the physics or statistics, \
with expected sizes of effects.

## 3. Algorithm specification
Numbered steps precise enough that a careful programmer who cannot ask you questions will build \
what you mean: inputs (exact `SolarDataset` members), every transformation with formulae, every \
hyper-parameter with its value and how it was chosen, numerical details that matter, outputs. \
Whenever you give an order, pixel or velocity-step index as a number, write it as 0-based (the \
convention of the project documents) and say so, for example "order index 60 (0-based)"; give the \
echelle order number or the wavelength as well when it helps. \
Say how it fits the resource request. If the method would be clearer, faster or only practical \
with a package that is not on the allow-list, say so and list it under `requires` -- do not bend \
the method to fit the default environment. Python is the preferred language: ask for a core in C, \
C++, Fortran or Julia only when Python (NumPy, Numba, JAX, PyTorch) would be significantly slower, \
and say why.

## 4. Why planets survive
The argument that a pure Doppler shift of the whole spectrum passes through with unit gain at \
periods from 2 to 400 days, and where that argument is weakest.

## 5. Predictions and falsification
What result would show the mechanism is wrong (not merely that the code has a bug)? Which \
intermediate quantity should look like what?

## 6. Requested diagnostics
Exact `ctx.metric` / `ctx.diagnostic` outputs you want back, each tied to a question in section 5.

## 7. Sources
Published papers or arXiv ids for anything borrowed. "None (first principles)" is acceptable.
"""


def proposer_prompt(slot: dict, cycle: int, library: str, notebook: str, history: str,
                    directive: str, shared: str, human: str) -> str:
    parts = [f"You are proposer {slot['id']}: {PERSONAS[slot['persona']]}. This is cycle {cycle}.",
             VIEWS[slot["view"]]]
    if library:
        parts.append("Your slice of the literature library (method cards written by the collaboration's "
                     "librarian from published papers):\n<library>\n" + library + "\n</library>")
    else:
        parts.append("You have deliberately been given no literature library: reason from first principles "
                     "and your own expertise. Independent re-derivation and unfamiliar routes are both valuable.")
    if notebook:
        parts.append("Your own notebook from earlier cycles:\n<notebook>\n" + notebook + "\n</notebook>")
    if history:
        parts.append("Your earlier proposals, with the moderators' selection reasoning, and evaluation "
                     "results and post-mortems where they exist. A proposal that was not selected has not "
                     "been tested and is not refuted: you may strengthen it in light of the moderators' "
                     "reasons and propose it again, or move on.\n<your_history>\n" + history
                     + "\n</your_history>")
    if shared:
        parts.append("Material shared with you this cycle from the wider collaboration:\n<shared>\n"
                     + shared + "\n</shared>")
    if human:
        parts.append("Suggestions from the human scientists (consider them seriously; you may adopt, adapt "
                     "or argue against them):\n<human_suggestions>\n" + human + "\n</human_suggestions>")
    if directive:
        parts.append("Direction for this cycle from the moderators:\n" + directive)
    parts.append(
        "Write ONE proposal for an algorithm to be implemented and evaluated this cycle. You work "
        "independently: other proposers are pursuing their own lines and you are not told what currently "
        "scores best, on purpose -- the collaboration needs approaches that differ in kind, not variations on "
        "a front-runner. Follow your own best judgement of what could reach the goal. You may deepen your own "
        "earlier line if its post-mortem points somewhere specific, or open a new one. Ambitious ideas with a "
        "clear mechanism are wanted; you do not need to be confident it will work, you need to be clear about "
        "why it might and how we would know.\n\n"
        "Reply with the proposal only, in exactly this structure (YAML front matter, then the sections):\n\n"
        + PROPOSAL_TEMPLATE +
        "\nAfter the proposal, add a final section `## Notebook` with up to 150 words of notes to your future "
        "self (ideas parked, things to check next cycle). It is stored privately for you.")
    return "\n\n".join(parts)


def reproduce_prompt(method: str, library: str) -> str:
    return (f"You are the collaboration's reference-implementation specialist. Write the specification for a "
            f"faithful reproduction of a published method, adapted to NEID solar data and to the algorithm "
            f"interface, so that the collaboration has a calibrated reference point.\n\nMethod: {method}\n\n"
            "Stay as close to the published algorithm as the available data products allow; where the "
            "papers leave a choice open, choose the most standard option and say so. Where a paper relies on "
            "data we do not have, substitute the closest quantity derivable from NEID L2 files and flag it.\n\n"
            "<library>\n" + library + "\n</library>\n\n"
            "Reply with the specification only, in exactly this structure:\n\n" + PROPOSAL_TEMPLATE)


def critique_prompt(skeptic: str, pid: str, proposal: str, rnd: int = 1, rounds: int = 1,
                    previous: str = "") -> str:
    head = (f"You are {PERSONAS[skeptic]}. You are reviewing proposal `{pid}` before it is implemented "
            f"(review round {rnd} of at most {rounds}).\n\n"
            "Your job is to make the experiment more informative, not to decide whether it runs: selection "
            "is made elsewhere, and unlikely-looking ideas are deliberately given their chance. So do not "
            "argue that the idea should be dropped. ")
    rule = ("**Rule check** -- say `RULES: OK`, or `RULES: VIOLATION` followed by the reason, only for "
            "hard rule breaks: use of information outside NEID L2 files, hard-coded epochs, anything that "
            "tries to influence scoring, or a method that cannot run within the resource caps.")
    verdict = ("Start your reply with one line: `VERDICT: OPEN` if you have at least one substantive point "
               "the proposer should still act on, or `VERDICT: SETTLED` if the specification is ready to "
               "implement as far as your expertise is concerned. ")
    if rnd == 1:
        body = ("Instead:\n\n"
                "1. **Failure modes** -- the two or three most likely ways this falls short or fools us, each "
                "with your rough estimate of how big the effect is.\n"
                "2. **The test for each** -- a concrete diagnostic (`ctx.metric` / `ctx.diagnostic`, or a check "
                "the fixed score already provides) whose value would tell us whether that failure happened.\n"
                "3. **Cheapest fix** -- for each, the smallest change to the specification that would remove or "
                "reduce the risk.\n"
                "4. **Ambiguities** -- places where an implementer would have to guess.\n"
                f"5. {rule}\n\n" + verdict +
                "Be specific and brief (under 500 words). If you see what would make the idea stronger, say it.")
        prev = ""
    else:
        body = ("The proposer has revised the specification after the previous round; its closing section "
                "`## Response to reviewers` says what was adopted and what was declined. Your previous review "
                "is included below. Now:\n\n"
                "1. **Still open** -- points from your previous review that the revision neither fixed nor "
                "answered convincingly. If the proposer declined a point with a reason, drop it unless you "
                "have a new argument or number; do not repeat yourself.\n"
                "2. **New** -- problems introduced by the revision, or that the added detail has exposed, each "
                "with the diagnostic that would reveal it and the cheapest fix.\n"
                "3. **Remaining ambiguities** for an implementer who cannot ask questions.\n"
                f"4. {rule}\n\n" + verdict +
                "Do not invent objections to fill the round, and do not push the proposer towards a safer or "
                "more conventional method: a short `VERDICT: SETTLED` reply is a good outcome. Under 350 words.")
        prev = f"\n\n<your_previous_review>\n{previous}\n</your_previous_review>" if previous else ""
    return f"{head}{body}\n\n<proposal id=\"{pid}\">\n{proposal}\n</proposal>{prev}"


def revise_prompt(slot: dict, pid: str, proposal: str, critiques: str, rnd: int = 1, rounds: int = 1) -> str:
    who = f"proposer {slot['id']}: {PERSONAS[slot['persona']]}" if slot else "the reference-implementation specialist"
    last = rnd >= rounds
    return (f"You are {who}. Reviewers have read the current version of your proposal `{pid}` (review round "
            f"{rnd} of at most {rounds}). Their role is to find failure modes and the tests that would "
            "reveal them.\n\n"
            f"<your_proposal>\n{proposal}\n</your_proposal>\n\n<reviews>\n{critiques}\n</reviews>\n\n"
            + ("Produce the final specification that will be handed to an implementer who cannot ask you "
               "questions. " if last else
               "Produce the revised specification. The reviewers will read it once more, but write it so "
               "that it could be handed as it stands to an implementer who cannot ask you questions. ") +
            "Keep your idea and your ambition: adopt fixes and diagnostics you find sound, "
            "resolve every ambiguity raised, and decline criticism you disagree with in one line each under "
            "a closing section `## Response to reviewers` (replace that section each round; it covers this "
            "round's reviews only). Do not water the idea down to make it safer, and do not let the "
            "specification grow with defensive detail that the method does not need. Use "
            "the same structure as the proposal (YAML front matter first), with updated predictions if your "
            "view changed. Reply with the specification only.")


CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "family": {"type": "string", "description": "existing family id such as F03, or NEW"},
        "new_family_name": {"type": "string"},
        "new_family_description": {"type": "string"},
        "novelty": {"type": "number", "description": "0 = repeat of an existing registry entry, 1 = unlike anything in the registry"},
        "closest_existing": {"type": "array", "items": {"type": "string"}},
        "one_line_summary": {"type": "string"},
    },
    "required": ["family", "new_family_name", "new_family_description", "novelty", "closest_existing",
                 "one_line_summary"],
    "additionalProperties": False,
}


def classify_prompt(pid: str, proposal: str, families: str, existing: str) -> str:
    return ("You maintain the collaboration's registry of approach families. Classify the proposal below.\n"
            "Assign the existing family whose core mechanism matches best. Answer NEW (and supply a name and "
            "a two-sentence description) only when the core mechanism fits none of them; a new data product "
            "or a new regulariser inside a known mechanism is not a new family. A proposal that genuinely "
            "combines mechanisms of two families belongs to F10.\n\n"
            f"<families>\n{families}\n</families>\n\n<existing_entries>\n{existing}\n</existing_entries>\n\n"
            f"<proposal id=\"{pid}\">\n{proposal}\n</proposal>")


MODERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "rationale": {"type": "string"}},
            "required": ["id", "rationale"], "additionalProperties": False}},
        "portfolio_comment": {"type": "string"},
        "human_suggestions_response": {"type": "string"},
        "seminar_topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ranking", "portfolio_comment", "human_suggestions_response", "seminar_topics"],
    "additionalProperties": False,
}


def moderate_prompt(k: int, n_select: int, specs: str, registry_md: str, critiques: str, human: str,
                    early: bool) -> str:
    return (f"You are moderator {k} of the collaboration. {n_select} of this cycle's proposals will be handed "
            "to implementers. Rank ALL proposals from most to least worth implementing this cycle.\n\n"
            "What you are optimising is the collaboration's chance of ending with several different "
            "algorithms that meet the goal, months from now -- not the current leaderboard. Therefore:\n"
            "- Value information: an experiment whose outcome would teach us something whichever way it "
            "goes ranks above a safe increment.\n"
            "- Value difference in kind. A family that is ahead after a few cycles is ahead on little "
            "evidence; families with few or no evaluated members are under-sampled, not inferior.\n"
            "- Do not rank an idea down because reviewers think it may fail. Rank down only for a hard rule "
            "violation, a specification too vague to implement, or near-duplication of something already "
            "evaluated.\n"
            "- A proposal that responds precisely to its own earlier post-mortem deserves credit.\n"
            + ("- The collaboration is in its early, independent phase: breadth matters most.\n" if early else "")
            + "\nA fixed rule applied after you will cap any one family's share of the selection and reserve "
            "places for novel ideas in untried families, so rank on merit within those principles.\n\n"
            "Also give: a short comment on the health of the portfolio; a response to each human suggestion "
            "(adopted by which proposal, or why not yet); and up to three questions for this cycle's seminar "
            "-- specific 'why did X behave like that?' or 'what limits Y?' questions raised by results so far.\n\n"
            f"<registry>\n{registry_md}\n</registry>\n\n<human_suggestions>\n{human or '(none)'}\n</human_suggestions>\n\n"
            f"<proposals>\n{specs}\n</proposals>\n\n<reviewer_rule_checks>\n{critiques}\n</reviewer_rule_checks>")


def implement_prompt(alg_id: str, parents: list[str]) -> str:
    extra = ""
    if parents:
        extra = ("The specification builds on earlier algorithms whose code you may read for reuse: "
                 + ", ".join(f"`algorithms/{p}/`" for p in parents) + ".\n")
    return (f"You are an implementer in the collaboration. Implement the algorithm specified in "
            f"`algorithms/{alg_id}/SPEC.md` so that it can be evaluated on the cluster.\n\n"
            f"Your writable directory is `algorithms/{alg_id}/`. You can read `docs/`, `literature/` and the "
            "reference implementations `algorithms/baseline_*/` (useful as working examples of the interface). "
            + extra +
            "\nDeliver, in your directory:\n"
            "- `algorithm.py` (plus helper modules if useful) implementing `run(ctx)` per the interface. "
            "Write it in Python unless that is significantly slower: only when the specification calls for "
            "it, or vectorised NumPy / Numba / JAX / PyTorch cannot make the numerical core fast enough, put "
            "C, C++, Fortran or Julia sources in `native/` and call them from `algorithm.py` (interface "
            "section 5a), and record the reason in the notes;\n"
            "- `algorithm.yaml` with name, family and the resource request from the specification;\n"
            "- `IMPLEMENTATION_NOTES.md`: what you built, every deviation from the specification with the "
            "reason, known weaknesses, and anything the proposer should know when reading results.\n\n"
            "What matters:\n"
            "- Fidelity. The proposer's idea is being tested, not yours. Implement what is specified, "
            "including every requested diagnostic. Where the specification is ambiguous or cannot work as "
            "written, make the choice that best serves its stated mechanism and record it in the notes.\n"
            "- It must run unattended on the real data (a few hundred to a couple of thousand sparse "
            "observations over about five years) within the requested resources: vectorise, batch, cache features with "
            "`ctx.scratch_file`, use `ctx.n_cpus` / the GPU if requested, log progress with `ctx.log`, and "
            "degrade gracefully (NaN for an observation, not a crash) on bad data.\n"
            "- Determinism: seed everything from `ctx.seed`.\n"
            "- Indices. Every index in the project documents and in `SolarDataset` is 0-based. When you write "
            "an index in a comment, a log line or the notes, say which base it uses. In Fortran or Julia "
            "code (1-based, column-major arrays) convert explicitly at the boundary and comment the "
            "conversion.\n"
            "- CPU time is scarce on the cluster. Read each spectrum once and only the orders you need, "
            "vectorise, and request in `algorithm.yaml` only the CPUs, memory and hours the code really "
            "uses. A numerical analyst and a performance engineer will examine the code after you.\n"
            "- The evaluation is fixed and you cannot see the real data. Do not add anything whose purpose is "
            "to do well on the score rather than to implement the method; in particular nothing that fits "
            "free functions of time unless the specification's mechanism calls for it.\n\n"
            "Use `run_tests` to check your work until it completes without error. The orchestrator host has "
            "little compute, so the local test data is deliberately tiny: normally about five real NEID "
            "spectra (the report then says 'Smoke test'). That proves only that the code runs on real data, "
            "so your code MUST run without crashing when `n_obs` is that small (clip ranks and component "
            "counts to the data, fall back to the pipeline RV where a model cannot be fitted) and must not "
            "be tuned to those few spectra. Compare the last two columns of the smoke table: for a method "
            "that preserves planets the change in your output follows the injected shift. If instead the "
            "report is for the synthetic toy Sun, make sure injected planets survive there (transfer near 1) "
            "unless the notes explain why the toy is misleading for this method. When you are done, reply "
            "with a three-line summary.")


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fix", "reject"]},
        "fidelity_issues": {"type": "array", "items": {"type": "string"}},
        "correctness_issues": {"type": "array", "items": {"type": "string"}},
        "integrity_issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["verdict", "fidelity_issues", "correctness_issues", "integrity_issues", "summary"],
    "additionalProperties": False,
}


def review_prompt(alg_id: str, spec: str, files: str, toy_report: str) -> str:
    return (f"You are the code reviewer for algorithm `{alg_id}`. You did not write it. Decide whether the "
            "code is a faithful, correct implementation of the specification that is ready for evaluation on "
            "the real data.\n\n"
            "Check, in this order:\n"
            "1. Integrity: nothing that reaches outside the data API, uses external information, hard-codes "
            "epochs, behaves differently depending on guesses about injection, or exists to game the score.\n"
            "2. Fidelity: each numbered step of the specification is implemented as written; deviations are "
            "listed in the notes and justified; requested diagnostics are emitted.\n"
            "3. Correctness: units (m/s vs km/s), sign conventions, array alignment with `ds.obs_id`, NaN "
            "handling, behaviour both with about 5 and with a couple of thousand observations. Indices: every "
            "index in the project documents is 0-based; check each order, pixel and velocity index against "
            "that, especially where the specification gives numbers and where arrays pass to Fortran or "
            "Julia (1-based, column-major). A numerical analyst reviews precision and conditioning "
            "separately and a performance engineer reviews speed, so do not spend effort there.\n\n"
            "Verdict `pass` when remaining issues would not change the scientific result; `fix` with concrete "
            "issues an implementer can act on; `reject` only if the code cannot be repaired without a new "
            "specification. A method you expect to perform poorly is not a reason to fail the review.\n\n"
            f"<specification>\n{spec}\n</specification>\n\n<code_and_notes>\n{files}\n</code_and_notes>\n\n"
            f"<toy_run_report>\n{toy_report}\n</toy_run_report>")


NUMERICS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fix"]},
        "numerical_issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["verdict", "numerical_issues", "summary"],
    "additionalProperties": False,
}


def numerics_prompt(alg_id: str, spec: str, files: str, toy_report: str) -> str:
    return (f"You are the numerical analyst of the collaboration, acting as a skeptic. Algorithm `{alg_id}` is "
            "about to use cluster time. You did not write it. Find the places where the NUMERICAL "
            "IMPLEMENTATION, not the idea, would produce a wrong or unstable answer. The goal is radial "
            "velocities good to 0.1 m/s, which is 3e-10 of the speed of light: errors that are harmless "
            "elsewhere matter here.\n\n"
            "Go through the code for:\n"
            "1. Indexing and memory layout. Every index in the project documents is 0-based. Look for "
            "off-by-one errors in order, pixel and velocity indices; for 0-based numbers used unchanged in "
            "Fortran or Julia code (1-based); for arrays crossing to Fortran or Julia without accounting for "
            "column-major storage; for dtype, contiguity and integer-width mismatches at a `ctypes` boundary.\n"
            "2. Precision against the goal. float32 where float64 is needed (a float32 wavelength is good to "
            "about 30 m/s; float32 CCFs summed over many pixels); catastrophic cancellation; differences of "
            "large numbers; interpolation whose error is not small compared with 0.1 m/s; inconsistent "
            "Doppler formulae or constants; accumulated rounding in long sums.\n"
            "3. Conditioning. Normal equations where QR or SVD is needed; unscaled polynomial or time bases; "
            "rank-deficient fits (also when `n_obs` is about 5); matrix inverses where a solve is meant; "
            "covariance matrices without the jitter that keeps a Cholesky factorisation alive; "
            "regularisation whose strength depends on the units of the inputs.\n"
            "4. Iterations and randomness. Convergence tests that can pass without convergence or never "
            "pass; iteration limits reached silently; dependence on starting values; results that change "
            "with the number of threads; random numbers not seeded from `ctx.seed`.\n"
            "5. Bad values. NaN or infinity spreading through sums, sorts and fits; zero or negative "
            "variances used as weights; division by quantities that can be near zero; edges of orders and "
            "of the velocity grid.\n"
            "6. If the test report says a performance optimisation changed the output, decide whether the "
            "change is rounding (far below 1 cm/s) or a real difference.\n\n"
            "For each issue give the file and function, what goes wrong and how large the effect could be "
            "(in m/s where you can estimate it), and the concrete fix. Report only issues that could change "
            "the result or make a run fail; no style remarks. Whether the idea is good, whether the code "
            "follows the specification and how fast it runs are other reviewers' business. Verdict `pass` "
            "when nothing you found would change the scientific result, otherwise `fix`.\n\n"
            f"<specification>\n{spec}\n</specification>\n\n<code_and_notes>\n{files}\n</code_and_notes>\n\n"
            f"<test_run_report>\n{toy_report}\n</test_run_report>")


PERFORMANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "optimise"]},
        "changes": {"type": "array", "items": {"type": "string"}},
        "recommended_resources": {
            "type": "object",
            "properties": {"cpus": {"type": "integer"}, "mem_gb": {"type": "integer"},
                           "hours": {"type": "number"}, "parallel_runs": {"type": "boolean"}},
            "required": ["cpus", "mem_gb", "hours", "parallel_runs"],
            "additionalProperties": False,
        },
        "estimated_core_hours_screen": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["verdict", "changes", "recommended_resources", "estimated_core_hours_screen", "summary"],
    "additionalProperties": False,
}


def performance_prompt(alg_id: str, spec: str, files: str, toy_report: str) -> str:
    return (f"You are the performance engineer of the collaboration, consulted before algorithm `{alg_id}` "
            "is run on the cluster. The code has passed review for correctness. The cluster is short of "
            "CPU time, and every evaluation runs this code two or three times on a few hundred to a couple "
            "of thousand spectra (122 orders x 9216 pixels, about 30 MB and 0.1-0.3 s to read each), so "
            "wasted CPU time delays every other algorithm. Find what would cut the CPU time of this code, "
            "and check that it asks for no more than it can use.\n\n"
            "Look for:\n"
            "1. Reading. Spectra read more than once, or whole spectra read where a few orders are used; "
            "features recomputed in a second pass instead of cached in memory or with `ctx.scratch_file`.\n"
            "2. Loops. Python loops over pixels, lines or observations that vectorised NumPy, Numba or one "
            "batched call would replace; work repeated inside a loop that could be done once outside.\n"
            "3. Algorithms. A cheaper algorithm with the SAME result: a solve instead of an inverse, a "
            "thin SVD instead of a full one, exploiting banded or low-rank structure, FFT-based "
            "correlation, fewer but sufficient optimiser or sampler iterations when convergence is checked.\n"
            "4. Parallelism. Cores requested but unused; a process or thread pool on top of multi-threaded "
            "BLAS (oversubscription); pools that copy large arrays to each worker.\n"
            "5. The resource request in `algorithm.yaml`. Estimate the wall time and memory of one run on "
            "500 and on 2000 spectra from the code and from the timings in the test report. Recommend the "
            "smallest `cpus`, `mem_gb` and `hours` that are safe (the hours cover the two runs of a screen "
            "evaluation; leave a factor of about two in time and 1.5 in memory). Single-threaded code "
            "should ask for 1 or 2 CPUs.\n"
            "6. Native code. C, C++, Fortran or Julia is justified only where Python cannot be made fast "
            "enough; conversely, say so if one hot loop deserves it.\n\n"
            "Rules. A change must leave the numerical results unchanged apart from rounding and must not "
            "alter the method. Never trade precision for speed (no float32 for wavelengths, velocities or "
            "sums that feed the RV). Recommend only changes that should save at least about a fifth of the "
            "run's CPU time, or that correct the resource request; say for each what to change, where "
            "(file and function), and roughly what it saves. Verdict `ok` when no such change exists -- "
            "most simple methods need none -- and `optimise` otherwise.\n\n"
            f"<specification>\n{spec}\n</specification>\n\n<code_and_notes>\n{files}\n</code_and_notes>\n\n"
            f"<test_run_report>\n{toy_report}\n</test_run_report>")


def optimise_prompt(alg_id: str, perf: dict) -> str:
    r = perf.get("recommended_resources") or {}
    changes = "\n".join(f"- {c}" for c in perf.get("changes", []))
    return (f"You are an implementer. The code in `algorithms/{alg_id}/` passed review. Before it uses "
            "cluster CPU time, the performance engineer recommends these changes:\n\n"
            f"{changes}\n\nRecommended resource request for `algorithm.yaml`: {r}\n"
            f"Performance engineer's summary: {perf.get('summary', '')}\n\n"
            "First call `run_tests` and keep the reported numbers. Then make the changes. They must not "
            "change the method or its results: after each substantial change call `run_tests` again and "
            "compare with the numbers you kept; differences must be rounding only (far below 1 cm/s). If a "
            "recommended change would alter the results or the method, do not make it, and say why in "
            "`IMPLEMENTATION_NOTES.md` under a heading 'Performance'. List there what you changed. The code "
            "will be reviewed again after this pass. Reply with a three-line summary.")


def fix_prompt(alg_id: str, review: dict) -> str:
    issues = "\n".join(f"- [{k}] {i}" for k in ("integrity_issues", "fidelity_issues", "correctness_issues",
                                                "numerical_issues")
                       for i in review.get(k, []))
    return (f"You are an implementer. A code reviewer and a numerical analyst examined `algorithms/{alg_id}/` "
            f"against its `SPEC.md` and asked for changes:\n\n{issues}\n\nReviewer summary: {review.get('summary', '')}\n\n"
            "Read the current files, fix each issue (or explain in `IMPLEMENTATION_NOTES.md` why the reviewer "
            "is mistaken), keep fidelity to the specification, and confirm with `run_tests` that the algorithm "
            "still completes. Reply with a three-line summary.")


def postmortem_prompt(slot: dict | None, pid: str, spec: str, notes: str, reports: str, references: str) -> str:
    who = f"proposer {slot['id']}: {PERSONAS[slot['persona']]}" if slot else "the reference-implementation specialist"
    return (f"You are {who}. Your algorithm `{pid}` has been evaluated by the fixed cluster pipeline. Write "
            "its post-mortem.\n\n"
            f"<specification>\n{spec}\n</specification>\n\n<implementation_notes>\n{notes}\n</implementation_notes>\n\n"
            f"<result_reports>\n{reports}\n</result_reports>\n\n"
            f"<reference_points>\n{references}\n</reference_points>\n\n"
            "Cover, in under 700 words:\n"
            "1. **Outcome versus prediction** -- the pre-registered numbers next to the measured ones.\n"
            "2. **What the evidence says about the mechanism** -- use the timescale table, the per-band "
            "transfers and your diagnostics. Distinguish 'the idea is wrong', 'the idea is right but too "
            "weak', 'the implementation or data handling let it down' and 'cannot tell'; say what would tell.\n"
            "3. **What limits it now** -- the largest remaining term in the residual scatter, as specifically "
            "as the evidence allows.\n"
            "4. **Next step** -- the single most promising continuation (or a clear recommendation to park "
            "the line). If you want humans to add a diagnostic, a data product or a library to the "
            "harness, put each request on its own line starting with `REQUEST TO HUMANS:` (requests only; "
            "scoring cannot be changed).\n"
            "5. **Transferable insight** -- one or two sentences another family could use.\n\n"
            "Be candid and specific; a clear account of a failure is a contribution. Do not conclude that the "
            "goal is unreachable from one experiment.")


def seminar_round1(persona: str, topic: str, material: str) -> str:
    return (f"You are {PERSONAS[persona]}, taking part in the collaboration's seminar.\n\nQuestion: {topic}\n\n"
            f"<material>\n{material}\n</material>\n\nGive your position in under 300 words: your explanation, "
            "the evidence in the material for and against it, and the one measurement that would settle it.")


def seminar_round2(persona: str, topic: str, positions: str) -> str:
    return (f"You are {PERSONAS[persona]} in the second round of the seminar on: {topic}\n\n"
            f"<positions>\n{positions}\n</positions>\n\nIn under 250 words: where do you now agree or "
            "disagree and why; what did someone else see that you missed; and what new approach to the "
            "overall problem does this discussion suggest? Build on others' points rather than repeating yours.")


def minutes_prompt(topic: str, transcript: str) -> str:
    return ("Write the minutes of this seminar as Markdown: the question, the competing explanations, points "
            "of agreement, open disagreements, proposed decisive measurements, and a final list `## Ideas this "
            "suggests` (new algorithmic directions, each one or two sentences). Report faithfully; add "
            f"nothing of your own.\n\nQuestion: {topic}\n\n<transcript>\n{transcript}\n</transcript>")


def crosspollinate_prompt(material: str, n: int) -> str:
    return ("You are the collaboration's synthesist. Below are the algorithms that passed the standard "
            "tests best in each family, with their post-mortems, and recent seminar minutes.\n\n"
            f"<material>\n{material}\n</material>\n\n"
            f"Write {n} cross-pollination briefs. Each brief (under 250 words, separated by a line "
            "`=====`) names two or more specific results from different families whose strengths are "
            "complementary -- one removes a residual the other leaves, or one supplies a regressor, "
            "constraint or representation the other lacks -- and sketches the combined method and why the "
            "combination should beat both. Prefer combinations across the most different families, and "
            "include at least one that involves a family that is currently behind. Do not rank families.")


def digest_prompt(facts: str) -> str:
    return ("Write this cycle's digest for the human scientists who supervise the collaboration: one page "
            "of Markdown. Lead with what changed in the results. Then: what was proposed and selected (one "
            "line each, with file paths), what failed and why, how the portfolio is spread across families, "
            "spending, and anything that needs a human decision. Use only the facts given; do not invent "
            f"numbers.\n\n<facts>\n{facts}\n</facts>")


def literature_prompt(topics: list[str], existing: list[str], web: bool) -> str:
    src = ("Use web search and fetch to find and read the papers (abstract pages and full text on arXiv / "
           "ADS / journal sites)." if web else
           "Web access is unavailable in this run: write from what you reliably know, and begin every card "
           "with the line `> UNVERIFIED: written from model memory; citations need checking by a human.`")
    return ("You are the collaboration's librarian. Build the literature library on methods for mitigating "
            "stellar activity in extreme-precision radial velocities.\n\n" + src + "\n\n"
            "Ethics of sources: published journal articles, conference proceedings and public arXiv preprints "
            "only. Do not use, seek or reproduce unpublished or private material (drafts, proposals, private "
            "repositories, talks without proceedings). Attribute every method to its authors with a full "
            "citation (authors, year, journal or arXiv id, DOI or URL). Summarise in your own words; do not "
            "copy text.\n\n"
            "Topics to cover:\n" + "\n".join(f"- {t}" for t in topics) + "\n\n"
            + ("Cards that already exist (update only if you have something new):\n"
               + "\n".join(f"- {e}" for e in existing) + "\n\n" if existing else "")
            + "For each distinct method write one card with `write_file` to `literature/<short_name>.md` "
            "containing: full citation(s); the idea in three sentences; the algorithm step by step with "
            "enough detail to re-implement; which data products it needs; reported performance (numbers, on "
            "which star / instrument, how measured); known limitations and failure cases reported in the "
            "literature; and how it treats planet-signal preservation. Mark anything you could not verify. "
            "Aim for 20-30 cards over the topics. Finish with `literature/INDEX.md`: one line per card, "
            "grouped by kind of approach, plus a short list of open problems the literature itself identifies.")

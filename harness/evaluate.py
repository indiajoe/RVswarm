"""Evaluate one algorithm at one stage. This is what a SLURM job runs.

    python -m harness.evaluate --alg <alg_id> --stage screen|full|sealed
    python -m harness.evaluate --alg-dir path/to/dir --stage toy      # local: a few real frames, or synthetic

The algorithm is run once without hidden planets and then n times with them, each in a
fresh process. At the stages listed under `precheck` in config/harness.yaml the injected
runs are skipped (to save CPU time) when the un-injected run does not reduce the scatter
of the pipeline RVs. Results go to workspace/results/<alg_id>/<stage>_<NN>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

import numpy as np
import yaml

from . import audit, backends, config, injection, score

# sparse like the curated real set: about one epoch every three days
TOY_BACKEND = {"kind": "synth", "seed": 3, "n_days": 900, "obs_per_day": 1, "day_keep": 0.33}


def read_manifest(alg_dir: Path, cfg: dict) -> dict:
    """algorithm.yaml: metadata + resource request, clipped to the configured caps."""
    man = {}
    p = alg_dir / "algorithm.yaml"
    if p.exists():
        try:
            man = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError:
            man = {}
    r, caps = {**cfg["resources"]["default"], **(man.get("resources") or {})}, cfg["resources"]
    man["resources"] = {
        "cpus": int(np.clip(int(r.get("cpus", 8)), 1, caps["max_cpus"])),
        "mem_gb": int(np.clip(int(r.get("mem_gb", 64)), 4, caps["max_mem_gb"])),
        "gpus": int(np.clip(int(r.get("gpus", 0)), 0, caps["max_gpus"])),
        "hours": float(max(0.1, float(r.get("hours", 4)))),
        "parallel_runs": bool(r.get("parallel_runs", False)),
    }
    return man


def stage_hours(res: dict, cfg: dict, stage: str) -> float:
    """Wall-time allowance for all runs of one evaluation at this stage."""
    r = cfg["resources"]
    return float(min(res["hours"] * r["time_factor"][stage], r["max_hours"][stage]))


def partition(be, cfg: dict, stage: str):
    """-> (index into backend, days to score on or None)."""
    days = np.unique(be.day)
    n_sealed = int(round(cfg["partitions"]["sealed_fraction"] * days.size))
    sealed_days = days[days.size - n_sealed:] if n_sealed else days[:0]
    dev_days = days[: days.size - n_sealed]
    part = cfg["stages"][stage]["partition"] if stage in cfg["stages"] else "dev"
    if part == "all":
        use = days
    elif part == "screen":
        use = dev_days[dev_days < dev_days.min() + cfg["partitions"]["screen_days"]]
    else:
        use = dev_days
    index = np.nonzero(np.isin(be.day, use))[0]
    index = index[np.argsort(be.time[index], kind="stable")]
    score_days = sealed_days if cfg["stages"].get(stage, {}).get("score_on") == "sealed" else None
    return index, score_days


def _sha_dir(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _array_text(a: np.ndarray) -> str:
    a = np.asarray(a, dtype=float)
    flat = a.reshape(-1)
    head = f"shape={a.shape} min={np.nanmin(flat):.5g} max={np.nanmax(flat):.5g} " \
           f"mean={np.nanmean(flat):.5g} std={np.nanstd(flat):.5g}"
    if flat.size > 120:
        flat = flat[np.linspace(0, flat.size - 1, 120).astype(int)]
        head += " (120 evenly spaced samples shown)"
    return head + "\n  " + " ".join(f"{x:.4g}" for x in flat)


def write_report(out: Path, alg_id: str, stage: str, m: dict, man: dict, runs: list[dict],
                 publish_details: bool, cfg: dict) -> None:
    g = cfg["gates"]
    L = [f"# Result: `{alg_id}` at stage `{stage}`", "",
         f"- written by the harness on {time.strftime('%Y-%m-%d %H:%M')} (code hash `{m.get('code_sha')}`)",
         f"- status: **{m.get('status')}**"]
    if m.get("status") != "ok":
        L += ["", "## Failure", "```", str(m.get("error", ""))[-3500:], "```"]
    elif m.get("smoke"):
        L += ["", "## Smoke test on a few real NEID spectra", "",
              f"The algorithm ran on {m['n_obs']} real, hand-picked spectra, once as they are and once with "
              "small Doppler shifts injected. This only shows that the code runs on real data and what "
              "it returns. There are far too few spectra to measure scatter or planet preservation; "
              "those are measured on the cluster. A method that needs many spectra to train must still "
              "run here without crashing (reduce its rank, or fall back, when `n_obs` is tiny).", "",
              f"- finite outputs: {m['valid_fraction']:.2f} of observations", "",
              "| time (d) | pipeline RV (m/s) | your corrected RV (m/s) | injected shift (m/s) | change in your output (m/s) |",
              "|---|---|---|---|---|"]
        L += ["| " + " | ".join(f"{x:.4f}" for x in row) + " |" for row in m["smoke_rows"]]
        L += ["", "For a method that preserves planets, the last two columns agree, apart from whatever "
              "the method legitimately cannot determine from so few spectra (for example a constant offset)."]
    elif m.get("injection_skipped"):
        L += [f"- observations scored: {m['n_obs']} on {m['n_days']} days "
              f"(baseline {m['baseline_days']:.0f} d)", "",
              "## Pre-check failed: no planet injection was run", "",
              "Before spending CPU time on the injection-recovery runs, the harness compares the scatter of "
              "the corrected RVs with that of the pipeline RVs (`rv_pipeline`, the DRP's CCFRVMOD) on the "
              "un-injected data. The corrected scatter must be below "
              f"{m['precheck_max_ratio']:.2f} x the pipeline scatter. It was not, so this algorithm could "
              "not have been promoted whatever its planet transfer, and the injected runs were skipped. "
              "Planet transfer is therefore NOT measured in this report.", "",
              "| quantity | value |", "|---|---|",
              f"| pipeline RV, RMS per epoch (m/s) | {m['rms_raw_daily']:.4f} |",
              f"| **corrected RV, RMS per epoch (m/s)** | **{m['rms_null_daily']:.4f}** |",
              f"| ratio corrected / pipeline | {m['precheck_ratio']:.3f} (needed: below {m['precheck_max_ratio']:.2f}) |",
              f"| robust (MAD) daily RMS (m/s) | {m['robust_rms_null_daily']:.4f} |",
              f"| valid output fraction | {m['valid_fraction']:.3f} (per observation), {m['day_fraction']:.3f} (per day) |",
              "", "## Scatter versus timescale (m/s)", "",
              "| bin | pipeline | corrected |", "|---|---|---|",
              f"| single observation | {m['rms_raw_per_obs']:.4f} | {m['rms_null_per_obs']:.4f} |",
              f"| 1 day | {m['rms_raw_daily']:.4f} | {m['rms_null_daily']:.4f} |"]
        for n in ("2", "4", "8", "16", "32"):
            L.append(f"| {n} days | {m['rms_raw_multiday'][n]:.4f} | {m['rms_null_multiday'][n]:.4f} |")
    else:
        L += [f"- observations scored: {m['n_obs']} on {m['n_days']} days "
              f"(baseline {m['baseline_days']:.0f} d)", "",
              "## Headline", "",
              "| quantity | value |", "|---|---|",
              f"| pipeline RV, RMS per epoch (m/s) | {m['rms_raw_daily']:.4f} |",
              f"| **corrected RV, RMS per epoch (m/s)** | **{m['rms_null_daily']:.4f}** |",
              f"| smallest planet amplitude detectable at 5 sigma in this many epochs (m/s) | {m.get('k_detect_5sigma', float('nan')):.4f} |",
              f"| robust (MAD) daily RMS (m/s) | {m['robust_rms_null_daily']:.4f} |",
              f"| planet amplitude transfer, min / max over bands | {m['min_transfer']:.3f} / {m['max_transfer']:.3f} |",
              f"| **effective RMS = RMS / min(1, min transfer) (m/s)** | **{m['effective_rms']:.4f}** |",
              f"| improvement over pipeline RV (x) | {m['improvement_over_raw']:.2f} |",
              f"| gate (transfer in {g['transfer_range']}, valid outputs) | {'PASS' if m['gate_pass'] else 'FAIL: ' + '; '.join(m['gate_reasons'])} |",
              f"| meets {g['target_rms_ms']*100:.0f} cm/s target | {'YES' if m['meets_target'] else 'no'} |",
              "", "## Blind planet injection-recovery", "",
              "Hidden circular-orbit planets (semi-amplitudes of order 10-25 cm/s, one per period band) "
              "were Doppler-injected into the data before the algorithm saw it. `transfer` = recovered / "
              "injected amplitude from the difference between the injected and un-injected runs "
              "(1.0 = planet perfectly preserved, 0 = absorbed). `absolute` is the same fit on the "
              "injected run alone (noisier: includes residual stellar signal at that period).", "",
              "| period band (d) | transfer per injection | quadrature | absolute |", "|---|---|---|---|"]
        for b in sorted(m["bands"], key=int):
            v = m["bands"][b]
            f = lambda xs: ", ".join(f"{x:.3f}" for x in xs)
            L.append(f"| {v['band_days'][0]:.0f}-{v['band_days'][1]:.0f} | {f(v['transfer'])} | "
                     f"{f(v['quadrature'])} | {f(v['absolute'])} |")
        L += ["", f"- distortion of the injected signal (rms of difference - truth, relative): {m['distortion']:.3f}",
              f"- RMS of (corrected - true planets), daily: {m['truth_rms_daily']:.4f} m/s",
              f"- valid output fraction: {m['valid_fraction']:.3f} (per observation), {m['day_fraction']:.3f} (per day)",
              "", "## Scatter versus timescale (m/s)", "",
              "| bin | pipeline | corrected |", "|---|---|---|",
              f"| single observation | {m['rms_raw_per_obs']:.4f} | {m['rms_null_per_obs']:.4f} |",
              f"| 1 day | {m['rms_raw_daily']:.4f} | {m['rms_null_daily']:.4f} |"]
        for n in ("2", "4", "8", "16", "32"):
            L.append(f"| {n} days | {m['rms_raw_multiday'][n]:.4f} | {m['rms_null_multiday'][n]:.4f} |")
    L += ["", "## Run facts", "",
          f"- resources: {man['resources']}",
          "- wall time per run (s): " + ", ".join(f"{r['runtime_s']:.0f}" for r in runs),
          f"- CPU time of all runs together: {m.get('cpu_core_hours', float('nan')):.3f} core-hours "
          f"({m.get('cpu_utilisation', float('nan')):.0%} of the cores that were reserved for that long). "
          "Low utilisation means the request was larger than the code can use.",
          "- run statuses: " + ", ".join(r["status"] for r in runs)]
    if publish_details and runs:
        r0 = runs[0]
        if r0["metrics"]:
            L += ["", "## Intermediate checks reported by the algorithm (un-injected run)", ""]
            L += [f"- `{k}` = {v:.6g}" for k, v in r0["metrics"].items()]
        if r0["diags"]:
            L += ["", "## Diagnostics arrays reported by the algorithm (un-injected run)", ""]
            for k, v in r0["diags"].items():
                L += [f"### `{k}`", "```", _array_text(v), "```"]
        L += ["", "## Log tail (un-injected run)", "```", r0["log_tail"], "```"]
    elif not publish_details:
        L += ["", "_Sealed-data run: algorithm logs and diagnostics are deliberately not published._"]
    (out / "report.md").write_text("\n".join(L) + "\n")


def run_one(py: str, snap: Path, view_file: Path, out_npz: Path, scratch: Path, log: Path,
            res: dict, timeout_s: float, cpus: int) -> dict:
    cmd = [py, "-m", "harness.runner", "--alg-dir", str(snap), "--view", str(view_file),
           "--out", str(out_npz), "--scratch", str(scratch), "--cpus", str(cpus),
           "--gpu", str(res["gpus"]), "--time-budget-s", str(int(timeout_s * 0.9))]
    env = dict(os.environ, PYTHONPATH=str(config.ROOT), OMP_NUM_THREADS=str(cpus),
               MKL_NUM_THREADS=str(cpus), OPENBLAS_NUM_THREADS=str(cpus),
               RVSWARM_RUN_WORK=str(view_file.parent), RVSWARM_RUN_SNAP=str(snap),
               RVSWARM_HARNESS_DIR=str(config.ROOT / "harness"))
    sandbox = os.environ.get("RVSWARM_SANDBOX_CMD", "").strip()
    if sandbox:                                   # $RVSWARM_RUN_WORK etc. are expanded in the template
        cmd = shlex.split(Template(sandbox).safe_substitute(env)) + cmd
    t0 = time.time()
    status = "ok"
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                cwd=str(view_file.parent), timeout=timeout_s).returncode
            if rc != 0:
                status = "error"
        except subprocess.TimeoutExpired:
            status = "timeout"
    out = {"status": status, "runtime_s": time.time() - t0, "metrics": {}, "diags": {},
           "rv_clean": None, "error": ""}
    lines = log.read_text(errors="replace").splitlines()
    out["log_tail"] = "\n".join(l[:300] for l in lines[-60:])
    if out_npz.exists():
        z = np.load(out_npz, allow_pickle=False)
        out["rv_clean"] = z["rv_clean"]
        out["error"] = str(z["error"])
        out["metrics"] = dict(zip([str(k) for k in z["metric_names"]], z["metric_values"].tolist()))
        out["diags"] = {k[6:]: z[k] for k in z.files if k.startswith("diag__")}
        if str(z["status"]) != "ok":
            out["status"] = "error"
    elif status == "ok":
        out["status"] = "error"
    if out["status"] != "ok" and not out["error"]:
        out["error"] = f"{out['status']}; log tail:\n" + out["log_tail"]
    return out


def evaluate(alg_id: str, stage: str, alg_dir: Path | None = None, out_root: Path | None = None,
             backend_spec: dict | None = None, keep_scratch: bool = False) -> dict:
    cfg = config.load()
    alg_dir = Path(alg_dir or config.WORKSPACE / "algorithms" / alg_id)
    out_root = Path(out_root or config.WORKSPACE / "results" / alg_id)
    out_root.mkdir(parents=True, exist_ok=True)
    k = 1 + max([int(p.name.rsplit("_", 1)[1]) for p in out_root.glob(f"{stage}_*")
                 if p.name.rsplit("_", 1)[1].isdigit()] or [0])
    out = out_root / f"{stage}_{k:02d}"
    out.mkdir()
    man = read_manifest(alg_dir, cfg)
    res = man["resources"]
    meta = {"alg_id": alg_id, "stage": stage, "run_dir": out.name, "status": "ok",
            "family": man.get("family"), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    snap = out / "code_snapshot"
    shutil.copytree(alg_dir, snap, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    meta["code_sha"] = _sha_dir(snap)
    publish = stage != "sealed"
    runs: list[dict] = []

    def finish(extra: dict) -> dict:
        meta.update(extra)
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=float))
        write_report(out, alg_id, stage, meta, man, runs, publish, cfg)
        (out / "DONE").write_text(meta["status"])
        return meta

    problems = audit.audit_dir(snap, cfg["audit"])
    if problems:
        return finish({"status": "audit_failed", "error": "\n".join(problems), "gate_pass": False})
    meta["sandboxed"] = bool(os.environ.get("RVSWARM_SANDBOX_CMD", "").strip())
    if stage != "toy" and not meta["sandboxed"] and os.environ.get("RVSWARM_ALLOW_UNSANDBOXED") != "1":
        return finish({"status": "no_sandbox", "gate_pass": False, "error":
                       "HARNESS CONFIGURATION (not an algorithm fault): RVSWARM_SANDBOX_CMD is not set. "
                       "Algorithm code must run inside a container that cannot read the harness secrets. "
                       "See RUNBOOK section 10, or set RVSWARM_ALLOW_UNSANDBOXED=1 to accept the risk."})

    smoke = False
    if stage == "toy":
        toy_dir = cfg["data"].get("toy_l2_dir")
        if backend_spec is None and toy_dir and os.environ.get("RVSWARM_TOY") != "synth":
            # a handful of real, hand-picked frames: does the code run on real data?
            if not (Path(cfg["data"]["toy_store_dir"]) / "manifest.json").exists():
                return finish({"status": "no_toy_store", "gate_pass": False, "error":
                               "HARNESS CONFIGURATION (not an algorithm fault): data.toy_l2_dir is set "
                               "but its store is missing. Run: python -m harness.ingest toy"})
            backend_spec = {"kind": "store", "store_dir": cfg["data"]["toy_store_dir"], "l2_dir": toy_dir}
            smoke = True
        backend_spec = backend_spec or TOY_BACKEND
        stage_cfg = {"n_injections": 1}
        hours = min(res["hours"], 0.5)
    else:
        test = os.environ.get("RVSWARM_TEST_BACKEND")          # plumbing tests only (see RUNBOOK)
        backend_spec = backend_spec or (json.loads(test) if test else
                                        {"kind": "store", "store_dir": cfg["data"]["store_dir"],
                                         "l2_dir": cfg["data"]["l2_dir"]})
        stage_cfg = cfg["stages"][stage]
        hours = stage_hours(res, cfg, stage)
    be = backends.open_backend(backend_spec)
    if stage == "toy":
        index = np.argsort(be.time, kind="stable")
        score_days = None
    else:
        index, score_days = partition(be, cfg, stage)
        mf = Path(cfg["data"]["store_dir"]) / "manifest.json"
        if not os.environ.get("RVSWARM_TEST_BACKEND") and mf.exists():
            meta["data_draw"] = json.loads(mf.read_text()).get("draw", {}).get("label")
    t = be.time[index]
    t0 = float(np.floor(t.min()))
    work = Path(os.environ.get("RVSWARM_WORKDIR") or os.environ.get("TMPDIR") or out) / f"rvswarm_{alg_id}_{out.name}_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    np.save(work / "index.npy", index)
    n_runs = 1 + int(stage_cfg["n_injections"])
    ep = f"{cfg['injection']['epoch']}" + (f"/draw{meta['data_draw']}" if meta.get("data_draw") else "")
    tags = [None] + [f"{ep}|{stage}|{r}|{alg_id}|{meta['code_sha']}"
                     for r in range(1, n_runs)]
    plans = []
    for r, tag in enumerate(tags):
        planets = [] if tag is None else injection.draw_planets(
            config.derive_seed("inj", tag), float(t.max() - t.min()), cfg["injection"])
        s = injection.signal(planets, t, t0)
        shift_file = work / f"shift_{r}.npy"
        np.save(shift_file, s)
        view = {"backend": backend_spec, "index_file": str(work / "index.npy"),
                "shift_file": str(shift_file), "alg_seed": 12345,
                "max_diag_bytes": cfg["audit"]["max_diagnostic_bytes"]}
        (work / f"view_{r}.json").write_text(json.dumps(view))
        plans.append({"planets": planets, "truth": s})
    # The un-injected run goes first, alone, on all granted cores. The injected runs follow,
    # concurrently if the algorithm asked for that (then each of the two phases gets half
    # of the wall time).
    par = (n_runs - 1) if (res["parallel_runs"] and res["gpus"] == 0) else 1
    timeout = hours * 3600 / (2 if par > 1 else n_runs)
    py = sys.executable

    def go(r: int, cpus: int) -> dict:
        return run_one(py, snap, work / f"view_{r}.json", work / f"out_{r}.npz", work / f"scratch_{r}",
                       out / f"run_{r}.log" if publish and r == 0 else work / f"run_{r}.log",
                       res, timeout, cpus)

    def cpu_seconds() -> float:
        u = resource.getrusage(resource.RUSAGE_CHILDREN)
        return u.ru_utime + u.ru_stime

    cpu0, wall0 = cpu_seconds(), time.time()
    runs = [go(0, res["cpus"])]
    # Pre-check (fixed rule, config `precheck`): no reduction of the pipeline scatter -> the
    # algorithm cannot be promoted whatever its planet transfer, so the injections are skipped.
    pre = cfg.get("precheck") or {}
    skipped = None
    if (stage in (pre.get("stages") or []) and runs[0]["status"] == "ok"
            and not alg_id.startswith(("baseline_", "canary_"))):
        m0 = score.score(t, be.day[index], be.rv_err[index], be.rv[index], runs[0]["rv_clean"],
                         [], t0, cfg["gates"], cfg["data"]["min_obs_per_day"], score_days)
        if m0.get("status") == "ok":
            ratio = m0["rms_null_daily"] / m0["rms_raw_daily"] if m0["rms_raw_daily"] > 0 else float("inf")
            if not ratio < float(pre["max_rms_ratio"]):          # also catches NaN
                skipped = dict(m0, injection_skipped=True, precheck_ratio=float(ratio),
                               precheck_max_ratio=float(pre["max_rms_ratio"]), gate_pass=False,
                               meets_target=False, min_transfer=None, max_transfer=None,
                               effective_rms=None, improvement_over_raw=None, k_detect_5sigma=None,
                               gate_reasons=[f"pre-check: corrected scatter is {ratio:.3f} x the pipeline "
                                             f"scatter (needed: below {pre['max_rms_ratio']}); "
                                             "injection-recovery not run"])
    if runs[0]["status"] == "ok" and skipped is None:
        with ThreadPoolExecutor(max_workers=par) as ex:
            runs += list(ex.map(lambda r: go(r, max(1, res["cpus"] // par)), range(1, n_runs)))
    cpu_s, wall_s = cpu_seconds() - cpu0, time.time() - wall0
    meta["cpu_core_hours"] = cpu_s / 3600
    meta["wall_hours"] = wall_s / 3600
    meta["cpu_utilisation"] = cpu_s / max(wall_s * res["cpus"], 1e-9)
    truth_dir = config.STATE / "truth" / alg_id
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / f"{out.name}.json").write_text(json.dumps(
        {"tags": tags, "planets": [p["planets"] for p in plans], "t0": t0}, indent=1))

    if runs[0]["status"] != "ok":
        result = finish({"status": "run_failed", "error": runs[0]["error"], "gate_pass": False})
    elif skipped is not None:
        skipped["algorithm_metrics"] = runs[0]["metrics"] if publish else {}
        if publish and runs[0]["diags"]:
            np.savez_compressed(out / "diagnostics.npz", **runs[0]["diags"])
        result = finish(skipped)
    elif smoke:
        # Too few spectra for any statistic: report only whether the code ran and what it returned.
        c0, r1 = runs[0]["rv_clean"], runs[1]
        m = {"smoke": True, "n_obs": int(index.size), "gate_pass": True, "gate_reasons": [],
             "valid_fraction": float(np.isfinite(c0).mean()),
             "smoke_rows": [[float(x) for x in row] for row in zip(
                 t - t0, be.rv[index], c0,
                 plans[1]["truth"], (r1["rv_clean"] - c0) if r1["status"] == "ok" else np.full(index.size, np.nan))]}
        if r1["status"] != "ok":
            m.update(status="run_failed", gate_pass=False,
                     error="the run with injected planets failed:\n" + r1["error"][-3000:])
        elif not np.isfinite(c0).any():
            m.update(gate_pass=False, gate_reasons=["no finite output"])
        m["algorithm_metrics"] = runs[0]["metrics"]
        result = finish(m)
    else:
        injected = [{"rv_clean": r["rv_clean"], "planets": p["planets"], "truth": p["truth"]}
                    for r, p in zip(runs[1:], plans[1:]) if r["status"] == "ok"]
        m = score.score(t, be.day[index], be.rv_err[index], be.rv[index], runs[0]["rv_clean"],
                        injected, t0, cfg["gates"], cfg["data"]["min_obs_per_day"], score_days)
        if len(injected) < n_runs - 1:
            m["gate_pass"] = False
            m.setdefault("gate_reasons", []).append("an injected run crashed or timed out")
            m["meets_target"] = False
            bad = next(r for r in runs[1:] if r["status"] != "ok")
            m["injected_run_error"] = bad["error"][-1500:]
        if publish and runs[0]["diags"]:
            np.savez_compressed(out / "diagnostics.npz", **runs[0]["diags"])
        m["algorithm_metrics"] = runs[0]["metrics"] if publish else {}
        result = finish(m)
    if not keep_scratch:
        shutil.rmtree(work, ignore_errors=True)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alg")
    ap.add_argument("--alg-dir")
    ap.add_argument("--stage", required=True, choices=["toy", "screen", "full", "sealed"])
    ap.add_argument("--out-root")
    ap.add_argument("--ticket", help="ticket file to mark as finished")
    a = ap.parse_args(argv)
    alg_id = a.alg or Path(a.alg_dir).resolve().name
    m = evaluate(alg_id, a.stage, a.alg_dir, a.out_root)
    if a.ticket and Path(a.ticket).exists():
        done = config.STATE / "eval_done"
        done.mkdir(parents=True, exist_ok=True)
        Path(a.ticket).rename(done / Path(a.ticket).name)
    keys = ("status", "rms_raw_daily", "rms_null_daily", "min_transfer", "effective_rms", "gate_pass")
    print(json.dumps({k: m.get(k) for k in keys}, default=float))


if __name__ == "__main__":
    main()

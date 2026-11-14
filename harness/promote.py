"""Fixed, agent-independent rules for what gets evaluated at which stage.

    python -m harness.promote enqueue --alg <id> [--stage screen]   # new algorithm -> queue
    python -m harness.promote advance                                # apply promotion rules
    python -m harness.promote fields <ticket.json>                   # for bin/submit_eval.sh
    python -m harness.promote leaderboard                            # print all results

screen -> full   : gate passed and effective_rms / raw_rms below a threshold
full   -> sealed : same with a stricter threshold, rationed per month and per family
`baseline_*` algorithms always go through every stage; `canary_*` stop after full.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from . import config, evaluate

ORDER = ["screen", "full", "sealed"]


def _results() -> list[dict]:
    out = []
    for p in sorted((config.WORKSPACE / "results").glob("*/*/metrics.json")):
        try:
            m = json.loads(p.read_text())
            if m.get("status") != "no_sandbox":           # harness misconfiguration, not a result
                out.append(m)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _pending() -> set[tuple[str, str]]:
    s = set()
    for d in ("eval_queue", "eval_submitted"):
        for p in (config.STATE / d).glob("*.json"):
            t = json.loads(p.read_text())
            s.add((t["alg_id"], t["stage"]))
    return s


def enqueue(alg_id: str, stage: str = "screen") -> Path | None:
    cfg = config.load()
    alg_dir = config.WORKSPACE / "algorithms" / alg_id
    if not (alg_dir / "algorithm.py").exists():
        raise FileNotFoundError(alg_dir / "algorithm.py")
    if (alg_id, stage) in _pending():
        return None
    man = evaluate.read_manifest(alg_dir, cfg)
    r = man["resources"]
    hours = evaluate.stage_hours(r, cfg, stage)
    ticket = {"alg_id": alg_id, "stage": stage, "cpus": r["cpus"], "mem_gb": r["mem_gb"],
              "gpus": r["gpus"], "minutes": int(hours * 60) + 30, "family": man.get("family"),
              "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    config.QUEUE.mkdir(parents=True, exist_ok=True)
    p = config.QUEUE / f"{alg_id}__{stage}.json"
    p.write_text(json.dumps(ticket))
    return p


def _close_lost_tickets() -> None:
    """A submitted job that never reported back (killed, node failure) becomes a failed result."""
    for p in (config.STATE / "eval_submitted").glob("*.json"):
        t = json.loads(p.read_text())
        if time.time() - p.stat().st_mtime < t["minutes"] * 60 + 3 * 86400:
            continue
        root = config.WORKSPACE / "results" / t["alg_id"]
        if not list(root.glob(f"{t['stage']}_*/DONE")):
            out = root / f"{t['stage']}_lost"
            out.mkdir(parents=True, exist_ok=True)
            m = {"alg_id": t["alg_id"], "stage": t["stage"], "run_dir": out.name, "status": "job_lost",
                 "family": t.get("family"), "gate_pass": False,
                 "error": "The cluster job never reported a result (time limit, memory or node failure)."}
            (out / "metrics.json").write_text(json.dumps(m))
            (out / "report.md").write_text(f"# Result: `{t['alg_id']}` at stage `{t['stage']}`\n\n"
                                           f"- status: **job_lost**\n\n{m['error']}\n")
            (out / "DONE").write_text("job_lost")
        (config.STATE / "eval_done").mkdir(exist_ok=True)
        p.rename(config.STATE / "eval_done" / p.name)


def advance() -> list[str]:
    cfg = config.load()
    pr = cfg["promotion"]
    res = [r for r in _results() if r.get("stage") in ORDER]
    _close_lost_tickets()
    res = [r for r in _results() if r.get("stage") in ORDER]
    pending = _pending()
    done = {(r["alg_id"], r["stage"]) for r in res}
    best: dict[tuple[str, str], dict] = {}
    for r in res:                                             # latest result per (alg, stage)
        best[(r["alg_id"], r["stage"])] = r
    actions = []

    def ratio(r):
        raw, eff = r.get("rms_raw_daily"), r.get("effective_rms")
        return eff / raw if raw and eff is not None and eff == eff else float("inf")

    month = time.strftime("%Y-%m")
    log_file = config.STATE / "sealed_log.jsonl"
    log = [json.loads(l) for l in log_file.read_text().splitlines()] if log_file.exists() else []
    used = [l for l in log if l["month"] == month and not l["alg_id"].startswith("baseline_")]
    fam_used = Counter(l.get("family") for l in used)
    sealed_candidates = []
    for (alg, stage), r in best.items():
        nxt = ORDER[ORDER.index(stage) + 1] if stage != "sealed" else None
        if nxt is None or (alg, nxt) in done or (alg, nxt) in pending:
            continue
        if not (config.WORKSPACE / "algorithms" / alg / "algorithm.py").exists():
            continue
        if alg.startswith("baseline_"):
            if r.get("status") == "ok":
                enqueue(alg, nxt)
                actions.append(f"{alg}: {stage} -> {nxt} (baseline)")
            continue
        if alg.startswith("canary_") and nxt == "sealed":
            continue
        if r.get("status") != "ok" or not (r.get("gate_pass") or alg.startswith("canary_")):
            continue
        if nxt == "full" and (ratio(r) <= pr["screen_to_full_max_ratio"] or alg.startswith("canary_")):
            enqueue(alg, nxt)
            actions.append(f"{alg}: screen -> full (ratio {ratio(r):.3f})")
        elif nxt == "sealed" and (ratio(r) <= pr["full_to_sealed_max_ratio"] or r.get("meets_target")):
            sealed_candidates.append((r["effective_rms"], alg, r.get("family")))
    for eff, alg, fam in sorted(sealed_candidates):
        if len(used) >= pr["sealed_runs_per_month"]:
            actions.append(f"{alg}: sealed run deferred (monthly ration used up)")
            continue
        if fam_used[fam] >= pr["sealed_max_per_family_per_month"]:
            actions.append(f"{alg}: sealed run deferred (family ration used up)")
            continue
        enqueue(alg, "sealed")
        entry = {"month": month, "alg_id": alg, "family": fam, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        used.append(entry)
        fam_used[fam] += 1
        with open(log_file, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        actions.append(f"{alg}: full -> sealed (effective rms {eff:.4f})")

    winners = [r for (a, s), r in best.items() if s == "sealed" and r.get("meets_target")
               and not a.startswith(("baseline_", "canary_"))]
    fams = {w.get("family") for w in winners}
    success = len(winners) >= pr["success_needed"] and len(fams) >= pr["success_min_families"]
    (config.STATE / "success.json").write_text(json.dumps(
        {"success": success, "winners": [w["alg_id"] for w in winners], "families": sorted(map(str, fams))}))
    return actions


def leaderboard() -> list[dict]:
    rows = {}
    for r in _results():
        if r.get("stage") in ORDER:
            rows[(r["alg_id"], r["stage"])] = r
    out = [{"alg_id": a, "stage": s, "status": r.get("status"), "family": r.get("family"),
            "rms_raw_daily": r.get("rms_raw_daily"), "rms_null_daily": r.get("rms_null_daily"),
            "min_transfer": r.get("min_transfer"), "effective_rms": r.get("effective_rms"),
            "gate_pass": r.get("gate_pass"), "meets_target": r.get("meets_target"),
            "run_dir": r.get("run_dir")} for (a, s), r in rows.items()]
    inf = float("inf")
    out.sort(key=lambda x: (ORDER.index(x["stage"]) * -1, not x["gate_pass"],
                            x["effective_rms"] if x["effective_rms"] is not None else inf))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["enqueue", "advance", "fields", "leaderboard"])
    ap.add_argument("ticket", nargs="?")
    ap.add_argument("--alg")
    ap.add_argument("--stage", default="screen")
    a = ap.parse_args(argv)
    if a.cmd == "enqueue":
        print(enqueue(a.alg, a.stage) or "already queued")
    elif a.cmd == "advance":
        print("\n".join(advance()) or "nothing to promote")
    elif a.cmd == "fields":
        t = json.loads(Path(a.ticket).read_text())
        print(t["alg_id"], t["stage"], t["cpus"], t["mem_gb"], t["gpus"], t["minutes"])
    else:
        for r in leaderboard():
            f = lambda x: "   -   " if x is None else f"{x:7.4f}"
            print(f"{r['stage']:7s} {'PASS' if r['gate_pass'] else 'fail'} raw={f(r['rms_raw_daily'])} "
                  f"rms={f(r['rms_null_daily'])} transfer={f(r['min_transfer'])} "
                  f"eff={f(r['effective_rms'])} {r['alg_id']}")


if __name__ == "__main__":
    main()

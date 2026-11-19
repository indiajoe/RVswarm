"""CPU-time ranking, and moving proposers or families of algorithms between swarms.

    bin/swarm.sh timing [--slow-minutes M]
        Rank this swarm's evaluated algorithms, proposers and families by the time their code
        takes on the cluster (also written to state/TIMING.md).

    bin/swarm.sh move --to ../rvswarm_B [--slot P05 ...] [--family F05 ...] [--alg c003_P07 ...]
                      [--copy] [--reevaluate passed|all|none] [--apply]
        Move proposers (configuration, notebook, every proposal, specification, code and
        post-mortem), every proposal of a family, and/or single proposals to another swarm. Without --apply it only
        shows what would happen. Both swarms must be between cycles, with their loops stopped.

    bin/swarm.sh import <bundle>
        Run by `move` inside the destination swarm; not normally used by hand.

What moves, and what does not:
* Moved: the proposer's entry in the panel, its notebook, its proposals with all debate files
  and specifications, algorithm code, review and implementation notes, post-mortems, and the
  result reports measured here (as history, clearly labelled).
* Not moved: evaluation results as results (the destination evaluates the code again on its
  own data and hidden planets), the harness secret, the data, the ledger.
* Ids change, because every swarm has its own P01...P24 and c001_P01...: proposer P05 of swarm
  A becomes P05A in B, proposal c003_P05 becomes A_c003_P05. Mentions of moved ids inside the
  moved documents are rewritten; mentions of ids that stay behind become "A:c002_P07".
* In the source swarm the moved code is removed from workspace/algorithms/ (so it is no longer
  evaluated there), queued evaluations are cancelled and the registry marks the proposals as
  moved. Everything moved is kept in state/moved/<date>_move_to_B/ as well.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import yaml

from harness import cache as hcache
from harness import config as hconfig
from harness import promote

from . import registry
from .cycle import (STATE, front_matter, load_cfg, load_state, log, pdir, refresh_results, spec_text,
                    swarm_lock, write_local_cfg)
from .tools import ROOT, WORKSPACE

TEXT_SUFFIXES = {".md", ".yaml", ".yml", ".txt"}
STAGES = ("screen", "full", "sealed")


# ======================================================================================= timing
def _all_metrics() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in sorted((WORKSPACE / "results").glob("*/*/metrics.json")):
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("stage") in STAGES and m.get("status") != "no_sandbox":
            out.setdefault(m.get("alg_id") or p.parent.parent.name, []).append(m)
    return out


def _wall_hours(m: dict) -> float | None:
    if m.get("wall_hours") is not None:
        return float(m["wall_hours"])
    try:
        a = time.mktime(time.strptime(m["started"], "%Y-%m-%dT%H:%M:%S"))
        b = time.mktime(time.strptime(m["finished"], "%Y-%m-%dT%H:%M:%S"))
        return (b - a) / 3600
    except (KeyError, TypeError, ValueError):
        return None


def _minutes_per_run(m: dict) -> float | None:
    """Wall time of the un-injected run (the same work for every algorithm at a stage), counting a
    reduction step taken from the cache at the time it took to compute."""
    if m.get("uninjected_run_cost_s") is not None:
        return float(m["uninjected_run_cost_s"]) / 60
    if m.get("runtimes_s"):
        return float(m["runtimes_s"][0]) / 60
    wh = _wall_hours(m)
    if wh is None:
        return None
    n = 1 if m.get("injection_skipped") else (2 if m.get("stage") == "screen" else 3)
    return wh * 60 / n


def _uses(pid: str, reg: dict) -> str:
    if pid not in reg["proposals"]:
        return "-"
    fm, _ = front_matter(spec_text(pid, reg))
    u = fm.get("uses") or []
    return ",".join(map(str, u)) if isinstance(u, list) else str(u)


def _f(x, nd=1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _min(x) -> str:
    """Minutes: whole above 10, else two significant digits."""
    return "-" if x is None else (f"{x:.0f}" if x >= 10 else f"{x:.2g}")


def timing(slow_minutes: float | None = None) -> None:
    reg = registry.load()
    cfg = load_cfg(env_overlay=False)
    slots = {s["id"]: s for s in cfg["proposer_slots"]}
    rows = []
    for alg, ms in _all_metrics().items():
        p = reg["proposals"].get(alg, {})
        ref = None
        for stage in STAGES:                     # screen first: the same data for every algorithm
            cand = [m for m in ms if m.get("stage") == stage and _minutes_per_run(m) is not None]
            if cand:
                ref = sorted(cand, key=lambda m: m.get("run_dir", ""))[-1]
                break
        if ref is None:
            continue
        mpr = _minutes_per_run(ref)
        n_in = ref.get("n_obs_input") or ref.get("n_obs")
        used = sum(float(m.get("cpu_core_hours") or 0.0) for m in ms)
        reserved = sum((_wall_hours(m) or 0.0) * float((m.get("resources") or {}).get("cpus") or 0) for m in ms)
        rows.append({"alg": alg, "slot": p.get("slot") or "-", "family": p.get("family") or ref.get("family") or "-",
                     "uses": _uses(alg, reg), "stage": ref["stage"], "mpr": mpr,
                     "sps": (mpr * 60 / n_in) if n_in else None, "used": used, "reserved": reserved,
                     "util": ref.get("cpu_utilisation"), "cpus": (ref.get("resources") or {}).get("cpus"),
                     "status": ref.get("status")})
    L = [f"# CPU time of the algorithms of this swarm ({os.environ.get('RVSWARM_NAME') or ROOT.name})", "",
         f"_Written by `bin/swarm.sh timing` on {time.strftime('%Y-%m-%d %H:%M')}._", ""]
    if not rows:
        L.append("No evaluation results with timing information yet.")
        _emit(L)
        return
    tot_res = sum(r["reserved"] for r in rows) or 1e-9
    L += ["'min/run' is the wall time of the un-injected run at the earliest stage reached (screen: the "
          "same data for every algorithm), with a preliminary reduction step counted at the time it took to "
          "compute even where its products came from the cache. 'core-h reserved' is what the allocation is "
          "charged (CPUs requested x wall time) over all stages; 'core-h used' is the CPU time the code "
          "actually used.", "",
          "## Algorithms, slowest first", "",
          "| algorithm | proposer | family | uses | stage | min/run | s per spectrum | core-h reserved | core-h used | CPU busy | CPUs | status |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["mpr"]):
        busy = "-" if r["util"] is None else f"{100 * r['util']:.0f} %"
        L.append(f"| {r['alg']} | {r['slot']} | {r['family']} | {r['uses']} | {r['stage']} | {_min(r['mpr'])} | "
                 f"{_f(r['sps'], 2)} | {_f(r['reserved'], 2)} | {_f(r['used'], 2)} | {busy} | "
                 f"{r['cpus'] or '-'} | {r['status']} |")

    def group(key: str, title: str, describe) -> dict[str, float]:
        g: dict[str, list[dict]] = {}
        for r in rows:
            g.setdefault(r[key], []).append(r)
        med = {k: statistics.median(x["mpr"] for x in v) for k, v in g.items()}
        L.extend(["", f"## By {title}, slowest first", "",
                  f"| {title} | | algorithms | median min/run | max min/run | core-h reserved | share of this swarm |",
                  "|---|---|---|---|---|---|---|"])
        for k in sorted(g, key=lambda k: -med[k]):
            v = g[k]
            res = sum(x["reserved"] for x in v)
            L.append(f"| {k} | {describe(k)} | {len(v)} | {_min(med[k])} | {_min(max(x['mpr'] for x in v))} | "
                     f"{_f(res, 2)} | {100 * res / tot_res:.0f} % |")
        return med

    slot_med = group("slot", "proposer",
                     lambda s: f"{slots[s]['persona']}, view {slots[s]['view']}" if s in slots else
                     ("reference / human request" if s == "-" else "(no longer in this swarm)"))
    fam_med = group("family", "family", lambda f: reg["families"].get(f, {}).get("name", ""))
    if slow_minutes is not None:
        slow_s = [s for s, m in slot_med.items() if s in slots and m >= slow_minutes]
        fast_s = [s for s, m in slot_med.items() if s in slots and m < slow_minutes]
        slow_f = [f for f, m in fam_med.items() if f in reg["families"] and m >= slow_minutes]
        L += ["", f"## Split at {slow_minutes:g} min per run", "",
              f"- slow proposers: {', '.join(sorted(slow_s)) or 'none'}",
              f"- fast proposers: {', '.join(sorted(fast_s)) or 'none'}",
              f"- proposers without a timed algorithm yet: {', '.join(sorted(set(slots) - set(slot_med))) or 'none'}",
              f"- slow families: {', '.join(sorted(slow_f)) or 'none'}"]
        if slow_s:
            L += ["", "To move the slow proposers to another swarm (a dry run; add --apply to do it):", "", "```",
                  "bin/swarm.sh move --to ../rvswarm_<NAME> " + " ".join(f"--slot {s}" for s in sorted(slow_s)), "```"]
        if slow_f:
            L += ["", "To move the slow families with all their proposals instead:", "", "```",
                  "bin/swarm.sh move --to ../rvswarm_<NAME> " + " ".join(f"--family {f}" for f in sorted(slow_f)), "```"]
    _emit(L)


def _emit(L: list[str]) -> None:
    text = "\n".join(L) + "\n"
    print(text)
    STATE.mkdir(exist_ok=True)
    (STATE / "TIMING.md").write_text(text)


# ========================================================================================= move
def _swarm_name(root: Path) -> str:
    env = root / "config" / "local.env"
    if env.exists():
        m = re.search(r'^\s*export\s+RVSWARM_NAME="?([A-Za-z0-9]*)"?', env.read_text(), re.M)
        if m and m.group(1):
            return m.group(1)
    return root.name


def _lock_busy(root: Path) -> str | None:
    """Who holds the orchestrator lock of the swarm at `root` (None: nobody). Never call this
    while this process holds that lock: closing the file would release it (POSIX locks)."""
    p = root / "state" / "orchestrator.lock"
    if not p.exists():
        return None
    with open(p, "a+") as fh:
        try:
            fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                fh.seek(0)
                return fh.read().strip() or "unknown process"
            return None
        fcntl.lockf(fh, fcntl.LOCK_UN)
    return None


def _has_code(pid: str) -> bool:
    return (WORKSPACE / "algorithms" / pid / "algorithm.py").exists()


def _will_reevaluate(p: dict, has_code: bool, policy: str) -> bool:
    if not has_code or policy == "none" or p.get("status") == "implementation_failed":
        return False
    if policy == "all":
        return True
    best = registry.best_result(p)
    return best is None or bool(best.get("gate_pass"))       # not yet evaluated, or passed the gate


def move(dest: str, slots: list[str], families: list[str], copy: bool, reevaluate: str, apply: bool,
         name: str | None = None, algs: list[str] | None = None) -> None:
    algs = list(algs or [])
    src = (name or os.environ.get("RVSWARM_NAME") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]+", src):
        raise SystemExit("This swarm has no name (RVSWARM_NAME in config/local.env, written by bin/new_swarm.sh). "
                         "Give one with --name (letters and digits only).")
    dpath = Path(dest).expanduser().resolve()
    if dpath == ROOT.resolve():
        raise SystemExit("The destination is this swarm.")
    if not (dpath / "bin" / "swarm.sh").exists():
        raise SystemExit(f"{dpath} is not a swarm directory (no bin/swarm.sh).")
    if not (dpath / "workspace" / "registry" / "registry.json").exists():
        raise SystemExit(f"Swarm {dpath} is not initialised: run bin/swarm.sh init there first.")
    dname = _swarm_name(dpath)
    if dname == src:
        raise SystemExit(f"Both swarms are called {src}; give them different names in config/local.env.")
    if not slots and not families and not algs:
        raise SystemExit("Name at least one --slot, --family or --alg.")
    st = load_state()
    if st.get("done"):
        raise SystemExit(f"This swarm is in the middle of cycle {st['cycle']} (phases done: {', '.join(st['done'])}). "
                         "Move between cycles: let the cycle finish first.")
    cfg = load_cfg(env_overlay=False)
    reg = registry.load()
    slot_cfg = {s["id"]: s for s in cfg["proposer_slots"]}
    bad = [s for s in slots if s not in slot_cfg]
    if bad:
        raise SystemExit(f"Unknown proposer(s) {bad}. This swarm has: {', '.join(sorted(slot_cfg))}")
    badf = [f for f in families if f not in reg["families"]]
    if badf:
        raise SystemExit(f"Unknown family id(s) {badf}. This swarm has: {', '.join(sorted(reg['families']))}")
    bada = [x for x in algs if x not in reg["proposals"] or x.startswith(("baseline_", "canary_"))
            or reg["proposals"][x].get("moved_to")]
    if bada:
        raise SystemExit(f"Not a movable proposal of this swarm: {bada} (baselines and canaries exist in every swarm; "
                         "a moved proposal has already gone)")
    chosen, refs_kept = [], []
    for pid, p in sorted(reg["proposals"].items()):
        if pid.startswith(("baseline_", "canary_")) or p.get("moved_to"):
            continue
        by_slot, by_fam, by_id = p.get("slot") in slots, p.get("family") in families, pid in algs
        if by_fam and not (by_slot or by_id) and p.get("kind") == "reference":
            refs_kept.append(pid)                     # every swarm has its own reproductions
            continue
        if by_slot or by_fam or by_id:
            chosen.append(pid)
            refresh_results(pid, p)                   # results that arrived since the last cycle
    queued = sorted(p.name for pid in chosen for p in (STATE / "eval_queue").glob(f"{pid}__*.json"))
    running = sorted(p.name for pid in chosen for p in (STATE / "eval_submitted").glob(f"{pid}__*.json"))

    verb = "COPY" if copy else "MOVE"
    L = [f"{verb} from swarm {src} ({ROOT}) to swarm {dname} ({dpath})", ""]
    for s in slots:
        c = slot_cfg[s]
        L.append(f"proposer {s} ({c['persona']}, knowledge {c['knowledge']}, view {c['view']}) "
                 f"-> {s}{src} in {dname}, with its notebook and all {sum(reg['proposals'][x].get('slot') == s for x in chosen)} of its proposals")
    for f in families:
        L.append(f"family {f} ({reg['families'][f]['name']}): every proposal in it; in {dname} it keeps its id "
                 "if that family exists there under the same name, otherwise it is added")
    if algs:
        L.append(f"single proposals: {', '.join(algs)}")
    L += ["", f"{len(chosen)} proposal(s):", "",
          f"  {'id here':<16} {'proposer':<9} {'family':<7} {'status':<24} code  {'best result here':<26} -> {dname}"]
    for pid in chosen:
        p = reg["proposals"][pid]
        b = registry.best_result(p)
        eff = b.get("effective_rms") if b else None
        res = "-" if not b else (f"{b['stage']} eff={eff:.3f} " if isinstance(eff, (int, float)) else f"{b['stage']} ") \
            + ("PASS" if b and b.get("gate_pass") else ("fail" if b else ""))
        todo = ("evaluated again (screen)" if _will_reevaluate(p, _has_code(pid), reevaluate)
                else ("implemented next cycle" if p.get("status") in ("selected", "waiting_for_environment") else "history only"))
        L.append(f"  {pid:<16} {str(p.get('slot') or '-'):<9} {str(p.get('family') or '-'):<7} "
                 f"{str(p.get('status'))[:24]:<24} {'yes ' if _has_code(pid) else 'no  '}  {res[:26]:<26} -> {src}_{pid}: {todo}")
    if refs_kept:
        L += ["", f"Reference reproductions in these families stay here (every swarm has its own): {', '.join(refs_kept)}"]
    if not copy:
        if queued:
            L += ["", f"Queued evaluations here that will be cancelled: {', '.join(queued)}"]
        if running:
            L += ["", f"Evaluations already running on the cluster: {', '.join(running)}. Their results will still "
                  "arrive here and are kept as a record, but nothing here promotes them any more."]
        L += ["", f"In {src}: the moved code leaves workspace/algorithms/, the proposers leave the panel "
              "(config/local_swarm.yaml), and the registry marks the proposals as moved."]
    else:
        L += ["", f"--copy: nothing changes in {src}."]
    L += ["", f"Results measured here are NOT carried over as results: {dname} evaluates the code again on its own "
          "data and hidden planets (--reevaluate decides which)."]
    print("\n".join(L), flush=True)
    if not chosen and not slots:
        raise SystemExit("\nNothing to move.")
    busy = [f"{n} (held by: {h})" for n, r in ((src, ROOT), (dname, dpath)) if (h := _lock_busy(r))]
    if busy:
        print(f"\nNOTE: an orchestrator is running in swarm {' and '.join(busy)}. Stop it (between cycles) "
              "before --apply; the move refuses otherwise.")
    if not apply:
        print("\nThis was a dry run; nothing was changed. Add --apply to do it. Both swarms must run the same "
              "code version (git pull in both first).")
        return

    with swarm_lock():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bdir = STATE / "moved" / f"{stamp}_{'copy' if copy else 'move'}_to_{dname}"
        files = bdir / "files"
        files.mkdir(parents=True)
        ign = shutil.ignore_patterns("__pycache__", "*.pyc")
        for pid in chosen:
            p = reg["proposals"][pid]
            if pdir(pid, reg).exists():
                shutil.copytree(pdir(pid, reg), files / "proposals" / pid, ignore=ign)
            if (WORKSPACE / "algorithms" / pid).exists():
                shutil.copytree(WORKSPACE / "algorithms" / pid, files / "algorithms" / pid, ignore=ign)
            for pm in (WORKSPACE / "postmortems").glob(f"{pid}__*.md"):
                (files / "postmortems").mkdir(exist_ok=True)
                shutil.copy2(pm, files / "postmortems" / pm.name)
            for stage, r in (p.get("results") or {}).items():
                rep = WORKSPACE / "results" / pid / str(r.get("run_dir")) / "report.md"
                if rep.exists():
                    (files / "results" / pid).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(rep, files / "results" / pid / f"{stage}_report.md")
        for s in slots:
            if (WORKSPACE / "agents" / s).exists():
                shutil.copytree(WORKSPACE / "agents" / s, files / "agents" / s)
        fams = {reg["proposals"][x].get("family") for x in chosen} | set(families)
        man = {"source": src, "source_root": str(ROOT), "dest": dname, "copy": copy, "reevaluate": reevaluate,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "slots": [slot_cfg[s] for s in slots],
               "proposals": {x: reg["proposals"][x] for x in chosen},
               "families": {f: reg["families"][f] for f in fams if f in reg["families"]},
               "source_ids": sorted(reg["proposals"])}
        (bdir / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("RVSWARM_") and k not in ("PYTHONPATH", "NEID_DATA_DIR")}
        log(f"importing into swarm {dname} ...")
        pr = subprocess.run([str(dpath / "bin" / "swarm.sh"), "import", str(bdir)], cwd=str(dpath), env=env)
        if pr.returncode != 0 or not (bdir / "IMPORTED.json").exists():
            raise SystemExit(f"The import into {dpath} failed (output above). Nothing was changed in this swarm; "
                             f"the bundle is kept in {bdir}.")
        imp = json.loads((bdir / "IMPORTED.json").read_text())
        if not copy:
            when = time.strftime("%Y-%m-%d")
            for pid in chosen:
                p = reg["proposals"][pid]
                p["moved_to"] = {"swarm": dname, "id": imp["id_map"][pid], "date": when}
                p["status"] = f"moved_to_{dname}"
                if (WORKSPACE / "algorithms" / pid).exists():
                    shutil.rmtree(WORKSPACE / "algorithms" / pid)          # a copy is in the bundle
                hcache.RunCache(hconfig.load()).drop(pid)                  # stored runs on this swarm's data
            for name_ in queued:
                (bdir / "cancelled_tickets").mkdir(exist_ok=True)
                try:
                    (STATE / "eval_queue" / name_).rename(bdir / "cancelled_tickets" / name_)
                except FileNotFoundError:                          # submitted in the meantime
                    log(f"note: {name_} was submitted to the cluster while moving; its result will be ignored here")
            for s in slots:
                if (WORKSPACE / "agents" / s).exists():
                    shutil.rmtree(WORKSPACE / "agents" / s)               # a copy is in the bundle
            if slots:
                write_local_cfg("proposer_slots", [c for c in cfg["proposer_slots"] if c["id"] not in slots])
            registry.save(reg)
        with open(STATE / "moves.log", "a") as fh:
            fh.write(json.dumps({"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "to": dname, "copy": copy,
                                 "slots": imp.get("slot_map"), "proposals": imp["id_map"],
                                 "bundle": str(bdir)}) + "\n")
    print(f"\nDone. {len(chosen)} proposal(s) {'copied' if copy else 'moved'} to swarm {dname}"
          + (f"; proposers {', '.join(f'{a} -> {b}' for a, b in imp.get('slot_map', {}).items())}" if slots else "")
          + f". Queued for evaluation there: {', '.join(imp.get('queued', [])) or 'none'}.\nBundle and backup: {bdir}")


# ======================================================================================= import
def _unique(base: str, used: set, sep: str) -> str:
    out, k = base, 2
    while out in used:
        out, k = f"{base}{sep}{k}", k + 1
    used.add(out)
    return out


def _rewriter(id_map: dict, source_ids: list, src: str):
    repl = dict(id_map)
    for i in source_ids:
        if i not in repl and not i.startswith(("baseline_", "canary_")):
            repl[i] = f"{src}:{i}"
    if not repl:
        return lambda s: s
    pat = re.compile(r"(?<![A-Za-z0-9_:])(" + "|".join(re.escape(k) for k in sorted(repl, key=len, reverse=True))
                     + r")(?![A-Za-z0-9])(?!_[A-Za-z0-9])")
    return lambda s: pat.sub(lambda m: repl[m.group(1)], s)


def _copy_rewrite(src: Path, dst: Path, rw) -> None:
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = Path(dirpath).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            a, b = Path(dirpath) / fn, dst / rel / fn
            if a.suffix == ".pyc":
                continue
            if a.suffix in TEXT_SUFFIXES and "reduction" not in rel.parts:   # a reduction step stays byte-identical
                b.write_text(rw(a.read_text(errors="replace")))
            else:
                shutil.copy2(a, b)


def import_bundle(bundle: str) -> None:
    bdir = Path(bundle).resolve()
    man = json.loads((bdir / "manifest.json").read_text())
    files, src = bdir / "files", man["source"]
    with swarm_lock():
        st = load_state()
        if st.get("done"):
            raise SystemExit(f"Swarm {ROOT} is in the middle of cycle {st['cycle']} (phases done: "
                             f"{', '.join(st['done'])}). Import between cycles: let the cycle finish first.")
        cfg = load_cfg(env_overlay=False)
        reg = registry.load()
        used_slots = {s["id"] for s in cfg["proposer_slots"]} | {p.get("slot") for p in reg["proposals"].values() if p.get("slot")}
        slot_map = {s["id"]: _unique(f"{s['id']}{src}", used_slots, "") for s in man["slots"]}
        taken = set(reg["proposals"]) | {d.name for d in (WORKSPACE / "algorithms").glob("*")}
        id_map = {old: _unique(f"{src}_{old}", taken, "_") for old in man["proposals"]}
        fam_map = {}
        for fid, f in man["families"].items():
            nm = f["name"].strip().lower()
            if fid in reg["families"] and reg["families"][fid]["name"].strip().lower() == nm:
                fam_map[fid] = fid
                continue
            same = [k for k, v in reg["families"].items() if v["name"].strip().lower() == nm]
            if same:
                fam_map[fid] = same[0]
                continue
            new = registry.add_family(reg, f["name"], f["description"], 0)
            reg["families"][new]["origin"] = f"imported from swarm {src} (its {fid})"
            fam_map[fid] = new
        rw = _rewriter(id_map, man["source_ids"], src)
        when = time.strftime("%Y-%m-%d")
        created: list[Path] = []
        queue = []
        try:
            for old, e0 in man["proposals"].items():
                new = id_map[old]
                e = json.loads(json.dumps(e0))
                e.pop("moved_to", None)
                e["origin"] = {"swarm": src, "id": old, "cycle": e0.get("cycle"), "kind": e0.get("kind"),
                               "slot": e0.get("slot"), "status": e0.get("status"), "date": when}
                e["cycle"], e["kind"] = 0, "imported"
                e["slot"] = slot_map.get(e0.get("slot"))
                if e0.get("family"):
                    e["family"] = fam_map.get(e0["family"], e0["family"])
                e["parents"] = [id_map.get(x, f"{src}:{x}") for x in e0.get("parents") or []]
                e["origin_results"], e["results"], e["postmortems"] = e0.get("results") or {}, {}, []
                for k in ("title", "decision", "summary"):
                    if isinstance(e.get(k), str):
                        e[k] = rw(e[k])
                e["moderator_notes"] = [rw(str(x)) for x in e.get("moderator_notes") or []]
                pd = WORKSPACE / "proposals" / "c000" / new
                if (files / "proposals" / old).exists():
                    created.append(pd)
                    _copy_rewrite(files / "proposals" / old, pd, rw)
                if (files / "results" / old).exists():
                    created.append(pd / f"results_in_swarm_{src}")
                    _copy_rewrite(files / "results" / old, pd / f"results_in_swarm_{src}", rw)
                ad = WORKSPACE / "algorithms" / new
                has_code = (files / "algorithms" / old / "algorithm.py").exists()
                if (files / "algorithms" / old).exists():
                    created.append(ad)
                    _copy_rewrite(files / "algorithms" / old, ad, rw)
                    mf = ad / "algorithm.yaml"
                    try:
                        y = yaml.safe_load(mf.read_text()) if mf.exists() else {}
                    except yaml.YAMLError:
                        y = {}
                    y = y if isinstance(y, dict) else {}
                    y["family"] = e.get("family")
                    mf.write_text(yaml.safe_dump(y, sort_keys=False))
                for pm in sorted((files / "postmortems").glob(f"{old}__*.md")):
                    dst = WORKSPACE / "postmortems" / f"{new}__{pm.stem.split('__', 1)[1]}-in-swarm-{src}.md"
                    dst.parent.mkdir(exist_ok=True)
                    created.append(dst)
                    dst.write_text(rw(pm.read_text(errors="replace")))
                if _will_reevaluate(e0, has_code, man["reevaluate"]):
                    queue.append(new)
                    e["status"] = "queued:screen"
                elif e0.get("status") in ("selected", "waiting_for_environment"):
                    e["status"] = e0["status"]
                else:
                    e["status"] = f"imported ({e0.get('status')})"
                reg["proposals"][new] = e
            for s in man["slots"]:
                nid = slot_map[s["id"]]
                nd = WORKSPACE / "agents" / nid
                created.append(nd)
                nd.mkdir(parents=True, exist_ok=True)
                nb = files / "agents" / s["id"] / "notebook.md"
                mine = [id_map[o] for o, e0 in man["proposals"].items() if e0.get("slot") == s["id"]]
                (nd / "notebook.md").write_text(
                    (rw(nb.read_text(errors="replace")) if nb.exists() else "")
                    + f"\n## Moved on {when} from swarm {src}, where I was proposer {s['id']}\n"
                    f"My earlier proposals continue here as {', '.join(mine) or 'none'}. Results measured in swarm "
                    f"{src} were on a different curated set of spectra: this swarm evaluates the code again on its own.\n")
            if man["slots"]:
                write_local_cfg("proposer_slots", list(cfg["proposer_slots"])
                                + [dict(s, id=slot_map[s["id"]]) for s in man["slots"]])
            registry.save(reg)
        except BaseException:
            for pth in reversed(created):                        # undo, so a failed import leaves no trace
                shutil.rmtree(pth, ignore_errors=True) if pth.is_dir() else pth.unlink(missing_ok=True)
            raise
        for new in queue:
            promote.enqueue(new, "screen")
        (bdir / "IMPORTED.json").write_text(json.dumps(
            {"id_map": id_map, "slot_map": slot_map, "family_map": fam_map, "queued": queue,
             "into": str(ROOT), "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
        with open(STATE / "moves.log", "a") as fh:
            fh.write(json.dumps({"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "from": src, "bundle": str(bdir),
                                 "slots": slot_map, "proposals": id_map, "queued": queue}) + "\n")
    log(f"import: {len(id_map)} proposal(s) from swarm {src}"
        + (f", proposers {', '.join(f'{a} -> {b}' for a, b in slot_map.items())}" if slot_map else "")
        + f"; queued for screen: {', '.join(queue) or 'none'}")

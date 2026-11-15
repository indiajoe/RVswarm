"""The research cycle. Stateless agents, all state on disk, every phase resumable.

    python -m swarm.cycle init            install reference algorithms, create folders
    python -m swarm.cycle run             run the current cycle (resumes where it stopped)
    python -m swarm.cycle run --only propose
    python -m swarm.cycle loop            cycle after cycle, each starting when the previous cycle's
                                          evaluation results are in, until the budget ends
    python -m swarm.cycle status
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import yaml

from harness import config as hconfig
from harness import promote

from . import prompts, registry
from .llm import LLM, BudgetExceeded, Refused
from .tools import ROOT, WORKSPACE, Toolset, run_toy

STATE = ROOT / "state"
STATE_FILE = STATE / "swarm_state.json"
PHASES = ["sync_results", "literature", "postmortem", "seminar", "propose", "classify", "debate",
          "moderate", "implement", "crosspollinate", "digest", "submit"]
RESULT_KEYS = ("status", "rms_raw_daily", "rms_null_daily", "min_transfer", "max_transfer",
               "effective_rms", "gate_pass", "gate_reasons", "meets_target", "run_dir", "improvement_over_raw")
_reg_lock = threading.Lock()


def log(*a) -> None:
    print(time.strftime("[%m-%d %H:%M:%S]"), *a, flush=True)


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v                                   # lists and scalars are replaced whole
    return base


def load_cfg() -> dict:
    """config/swarm.yaml, optionally overlaid by the file named in $RVSWARM_SWARM_OVERLAY
    (for example config/trial.yaml: a small, capped trial run)."""
    cfg = yaml.safe_load((ROOT / "config" / "swarm.yaml").read_text())
    over = os.environ.get("RVSWARM_SWARM_OVERLAY")
    if over:
        p = Path(over)
        _merge(cfg, yaml.safe_load((p if p.is_absolute() else ROOT / p).read_text()) or {})
    return cfg


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"cycle": 1, "done": [], "seminar_topics": [], "last_finished": 0}


def save_state(st: dict) -> None:
    STATE.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, indent=1))


def front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"\s*(?:```\w*\s*)?---\s*\n(.*?)\n---\s*\n(.*)", text, re.S)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), m.group(2)


def read(p: Path, cap: int | None = None) -> str:
    if not p.exists():
        return ""
    s = p.read_text(errors="replace")
    return s if cap is None or len(s) <= cap else s[:cap] + "\n... (truncated)"


def last_critiques(d: Path) -> list[Path]:
    """The reviews of the last debate round a proposal went through."""
    files = sorted(d.glob("CRITIQUE_r*_*.md"))
    if not files:
        return []
    rnd = max(int(f.name.split("_")[1][1:]) for f in files)
    return [f for f in files if f.name.startswith(f"CRITIQUE_r{rnd}_")]


def _verdict(text: str) -> str:
    m = re.search(r"VERDICT:\s*\**\s*(SETTLED|OPEN)", text.upper())
    return m.group(1) if m else "OPEN"


def pdir(pid: str, reg: dict) -> Path:
    return WORKSPACE / "proposals" / f"c{reg['proposals'][pid]['cycle']:03d}" / pid


def spec_text(pid: str, reg: dict) -> str:
    d = pdir(pid, reg)
    return read(d / "SPEC.md") or read(d / "PROPOSAL.md")


def fmt(x, nd=4) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def result_line(pid: str, stage: str, r: dict) -> str:
    return (f"{pid} [{stage}] status={r.get('status')} raw={fmt(r.get('rms_raw_daily'))} "
            f"rms={fmt(r.get('rms_null_daily'))} min_transfer={fmt(r.get('min_transfer'), 3)} "
            f"effective_rms={fmt(r.get('effective_rms'))} gate={'PASS' if r.get('gate_pass') else 'FAIL'}")


# ================================================================================================
class Cycle:
    def __init__(self):
        self.cfg = load_cfg()
        self.c = self.cfg["cycle"]
        self.st = load_state()
        self.n = self.st["cycle"]
        self.llm = LLM(self.cfg, STATE)
        self.llm.cycle = self.n
        self.reg = registry.load()
        self.slots = {s["id"]: s for s in self.cfg["proposer_slots"]}
        self._system = None

    @property
    def system(self) -> str:
        if self._system is None:
            self._system = prompts.system_prompt()
        return self._system

    def save(self) -> None:
        with _reg_lock:
            registry.save(self.reg)

    def tag(self, phase: str) -> str:
        return f"c{self.n:03d}_{phase}"

    def this_cycle(self, kinds=("idea", "human", "reference")) -> list[str]:
        return sorted(p for p, v in self.reg["proposals"].items()
                      if v["cycle"] == self.n and v.get("kind") in kinds and not p.startswith(("baseline_", "canary_")))

    # ------------------------------------------------------------------------------ helpers
    def library(self, knowledge: str) -> str:
        cards = sorted(p for p in (WORKSPACE / "literature").glob("*.md") if p.name != "INDEX.md")
        if knowledge == "none" or not cards:
            return ""
        if knowledge == "A":
            cards = cards[0::2]
        elif knowledge == "B":
            cards = cards[1::2]
        body = "\n\n".join(f"### card: {p.name}\n{read(p, 9000)}" for p in cards)
        if knowledge == "full":
            body = read(WORKSPACE / "literature" / "INDEX.md", 12000) + "\n\n" + body
        return body[:220000]

    def informed_slots(self) -> set[str]:
        if self.n <= self.c["independence_cycles"]:
            return set()
        ids = sorted(self.slots)
        k = max(1, round(self.c["informed_fraction"] * len(ids)))
        return {ids[(self.n * k + j) % len(ids)] for j in range(k)}

    def human_suggestions(self, priority: bool | None = None) -> dict[str, str]:
        out = {}
        for p in sorted((WORKSPACE / "human_suggestions").glob("*.md")):
            if p.name == "README.md":
                continue
            is_prio = p.name.upper().startswith("PRIORITY")
            if priority is None or priority == is_prio:
                out[p.name] = read(p, 8000)
        return out

    def references(self) -> str:
        rows = [result_line(pid, s, r) for pid, p in sorted(self.reg["proposals"].items())
                if pid.startswith("baseline_") for s, r in (p.get("results") or {}).items()]
        return "\n".join(rows) or "(reference baselines have not been evaluated yet)"

    def history(self, slot: str) -> str:
        mine = sorted((v["cycle"], pid) for pid, v in self.reg["proposals"].items() if v.get("slot") == slot)
        out = []
        for i, (cyc, pid) in enumerate(mine):
            p = self.reg["proposals"][pid]
            head = (f"#### {pid} (cycle {cyc}) \"{p.get('title', '')}\" -- family {p.get('family')}, "
                    f"{'selected' if p.get('selected') else 'not selected this time'}, status: {p.get('status')}")
            lines = [head, p.get("summary", "")]
            lines += [result_line(pid, s, r) for s, r in (p.get("results") or {}).items()]
            if i >= len(mine) - 4:
                if p.get("decision"):
                    lines.append(f"Selection decision: {p['decision']}")
                lines += [f"Moderator {w}"[:1500] for w in p.get("moderator_notes", [])]
                if i >= len(mine) - 2:                     # reviewers' last word on the final draft's predecessor
                    for cf in last_critiques(pdir(pid, self.reg)):
                        lines.append(f"Reviewer's final-round critique ({cf.stem.split('_', 2)[-1]}):\n" + read(cf, 3000))
                for pm in sorted((WORKSPACE / "postmortems").glob(f"{pid}__*.md")):
                    lines.append(f"Post-mortem ({pm.stem.split('__')[1]}):\n" + read(pm, 6000))
            out.append("\n".join(l for l in lines if l))
        return "\n\n".join(out)

    def leaderboard_md(self) -> str:
        L = ["| algorithm | family | stage | raw RMS | RMS | min transfer | effective RMS | gate |", "|---|---|---|---|---|---|---|---|"]
        rows = []
        for pid, p in self.reg["proposals"].items():
            b = registry.best_result(p)
            if b:
                e = b.get("effective_rms")
                rows.append(((not b.get("gate_pass")), e if e is not None and math.isfinite(e) else 1e9, pid, p, b))
        for _, _, pid, p, b in sorted(rows, key=lambda r: r[:3]):
            L.append(f"| {pid} | {p.get('family')} | {b['stage']} | {fmt(b.get('rms_raw_daily'))} | "
                     f"{fmt(b.get('rms_null_daily'))} | {fmt(b.get('min_transfer'), 3)} | "
                     f"{fmt(b.get('effective_rms'))} | {'PASS' if b.get('gate_pass') else 'fail'} |")
        return "\n".join(L)

    # ------------------------------------------------------------------------------ phases
    def sync_results(self) -> None:
        new = 0
        for pid, p in self.reg["proposals"].items():
            root = WORKSPACE / "results" / pid
            if not root.exists():
                continue
            for mf in sorted(root.glob("*/metrics.json")):
                if not (mf.parent / "DONE").exists():
                    continue
                m = json.loads(mf.read_text())
                stage = m.get("stage")
                if stage not in ("screen", "full", "sealed") or m.get("status") == "no_sandbox":
                    continue
                res = p.setdefault("results", {})
                if res.get(stage, {}).get("run_dir") != m.get("run_dir"):
                    res[stage] = {k: m.get(k) for k in RESULT_KEYS}
                    p["status"] = f"evaluated:{stage}" if m.get("status") == "ok" else f"failed:{stage}"
                    new += 1
        (WORKSPACE / "registry").mkdir(exist_ok=True)
        (WORKSPACE / "registry" / "LEADERBOARD.md").write_text(
            "# Leaderboard (best stage reached per algorithm)\n\n" + self.leaderboard_md() + "\n")
        self.save()
        log(f"sync_results: {new} new result(s)")

    def literature(self) -> None:
        lit = WORKSPACE / "literature"
        lit.mkdir(exist_ok=True)
        due = not (lit / "INDEX.md").exists() or (self.n > 1 and (self.n - 1) % self.c["literature_refresh_every"] == 0)
        if not due:
            return
        lc = self.cfg["literature"]
        existing = [p.name for p in sorted(lit.glob("*.md")) if p.name != "INDEX.md"]
        ts = Toolset(read_roots=["docs"], write_roots=["literature"])
        server = []
        if lc["web"]:
            server = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 60,
                       "allowed_domains": lc["allowed_domains"]},
                      {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 80,
                       "allowed_domains": lc["allowed_domains"]}]
        try:
            self.llm.agent("literature", self.system, prompts.literature_prompt(lc["topics"], existing, bool(server)),
                           ts, lc["max_tool_turns"], tag=self.tag("literature"), server_tools=server)
        except Exception as e:                       # web tools unavailable for this account etc.
            if isinstance(e, BudgetExceeded) or not server:
                raise
            log(f"literature: web tools failed ({type(e).__name__}: {e}); falling back to model memory")
            self.llm.agent("literature", self.system, prompts.literature_prompt(lc["topics"], existing, False),
                           ts, lc["max_tool_turns"], tag=self.tag("literature_noweb"))
        log(f"literature: {len(list(lit.glob('*.md')))} files in the library")

    def postmortem(self) -> None:
        todo = []
        for pid, p in self.reg["proposals"].items():
            if pid.startswith(("baseline_", "canary_")):
                continue
            for stage in ("sealed", "full", "screen"):
                if stage in (p.get("results") or {}) and stage not in p.get("postmortems", []):
                    todo.append((pid, stage))
                    break
        todo = todo[: self.c["max_postmortems_per_cycle"]]
        items = {}
        for pid, stage in todo:
            p = self.reg["proposals"][pid]
            reports = "\n\n".join(
                f"--- stage {s} ---\n" + read(WORKSPACE / "results" / pid / r["run_dir"] / "report.md", 14000)
                for s, r in p["results"].items() if s not in p.get("postmortems", []))
            items[f"{pid}__{stage}"] = prompts.postmortem_prompt(
                self.slots.get(p.get("slot")), pid, spec_text(pid, self.reg),
                read(WORKSPACE / "algorithms" / pid / "IMPLEMENTATION_NOTES.md", 8000), reports, self.references())
        out = self.llm.call_many("postmortem", self.system, items, phase_tag=self.tag("postmortem"))
        (WORKSPACE / "postmortems").mkdir(exist_ok=True)
        requests = []
        for key, text in out.items():
            pid, stage = key.split("__")
            (WORKSPACE / "postmortems" / f"{key}.md").write_text(f"# Post-mortem: {pid} ({stage})\n\n{text}\n")
            p = self.reg["proposals"][pid]
            p.setdefault("postmortems", [])
            p["postmortems"] += [s for s in p["results"] if s not in p["postmortems"]]
            requests += [f"- from `{pid}`: {l.strip()}" for l in text.splitlines() if "REQUEST TO HUMANS" in l.upper()]
        if requests:
            d = WORKSPACE / "diagnostic_requests"
            d.mkdir(exist_ok=True)
            (d / f"c{self.n:03d}_requests.md").write_text(
                "# Requests from post-mortems (for the human harness maintainer)\n\n" + "\n".join(requests) + "\n")
        self.save()
        log(f"postmortem: {len(out)} written")

    def seminar(self) -> None:
        topics = self.st.get("seminar_topics", [])[: self.c["seminar_topics"]]
        evaluated = [pid for pid, p in self.reg["proposals"].items() if p.get("results")]
        if self.n == 1 or not topics or len(evaluated) < 3:
            return
        pms = sorted((WORKSPACE / "postmortems").glob("*.md"), key=lambda p: p.stat().st_mtime)[-8:]
        material = ("## Registry\n" + registry.render(self.reg) + "\n## Recent post-mortems\n"
                    + "\n\n".join(read(p, 5000) for p in pms))[:150000]
        personas = [k for k in prompts.PERSONAS if not k.startswith("skeptic")]
        out_dir = WORKSPACE / "seminars"
        out_dir.mkdir(exist_ok=True)
        for k, topic in enumerate(topics, 1):
            dest = out_dir / f"c{self.n:03d}_seminar_{k}.md"
            if dest.exists():
                continue
            panel = [personas[(self.n + k + j * 3) % len(personas)] for j in range(self.c["seminar_panel"])]
            panel = list(dict.fromkeys(panel))
            r1 = self.llm.call_many("seminar", self.system,
                                    {p: prompts.seminar_round1(p, topic, material) for p in panel},
                                    phase_tag=self.tag(f"seminar{k}_r1"))
            pos = "\n\n".join(f"[{p}]\n{t}" for p, t in r1.items())
            r2 = self.llm.call_many("seminar", self.system,
                                    {p: prompts.seminar_round2(p, topic, pos) for p in r1},
                                    phase_tag=self.tag(f"seminar{k}_r2"))
            transcript = pos + "\n\n--- round 2 ---\n\n" + "\n\n".join(f"[{p}]\n{t}" for p, t in r2.items())
            minutes = self.llm.call("minutes", self.system, prompts.minutes_prompt(topic, transcript),
                                    tag=self.tag(f"seminar{k}_minutes"))
            dest.write_text(f"# Seminar, cycle {self.n}: {topic}\n\n{minutes}\n\n---\n\n"
                            f"<details><summary>Full transcript</summary>\n\n{transcript}\n\n</details>\n")
        log(f"seminar: {len(topics)} topic(s)")

    def propose(self) -> None:
        items, meta = {}, {}
        informed = self.informed_slots()
        ids = sorted(self.slots)
        human_all = self.human_suggestions(priority=False)
        for i, sid in enumerate(ids):
            slot = self.slots[sid]
            for j in range(self.c["proposals_per_slot"]):
                pid = f"c{self.n:03d}_{sid}" + ("" if j == 0 else "abcdefgh"[j])
                if pid in self.reg["proposals"]:
                    continue
                directive = ""
                rd = self.reg.get("redirects", {}).get(sid)
                if rd and rd["cycle"] == self.n:
                    f = self.reg["families"][rd["family"]]
                    directive = (f"Several proposers have converged on similar approaches. For this cycle, "
                                 f"take your expertise to a direction the collaboration has barely explored: "
                                 f"**{f['name']}** -- {f['description']} Bring your own angle to it; you will "
                                 f"be free to return to your earlier line afterwards.")
                shared = []
                asg = self.reg.get("assignments", {}).get(sid)
                if asg and asg["cycle"] == self.n:
                    shared.append("Cross-pollination brief addressed to you:\n" + read(WORKSPACE / asg["path"]))
                if sid in informed:
                    shared.append("Results so far across all families (you are one of the few proposers shown "
                                  "this picture this cycle; use it to combine or contrast, not to imitate):\n"
                                  + self.leaderboard_md())
                    sems = sorted((WORKSPACE / "seminars").glob("*.md"))[-2:]
                    shared += [read(s, 7000).split("<details>")[0] for s in sems]
                human = "\n\n".join(human_all.values()) if (i + self.n) % 3 == 0 else ""
                nb = read(WORKSPACE / "agents" / sid / "notebook.md")[-6000:]
                items[pid] = prompts.proposer_prompt(slot, self.n, self.library(slot["knowledge"]), nb,
                                                     self.history(sid), directive, "\n\n".join(shared), human)
                meta[pid] = {"slot": sid, "kind": "idea", "redirected": bool(directive)}
        out = self.llm.call_many("propose", self.system, items, phase_tag=self.tag("propose"))

        ref_items = {}
        if self.n == 1:
            for k, method in enumerate(self.cfg.get("bootstrap_reproductions", []), 1):
                pid = f"ref_{k:02d}"
                if pid not in self.reg["proposals"]:
                    ref_items[pid] = prompts.reproduce_prompt(method, self.library("full"))
                    meta[pid] = {"slot": None, "kind": "reference"}
        done_h = self.reg.setdefault("human_done", [])
        for name, text in self.human_suggestions(priority=True).items():
            if name in done_h:
                continue
            pid = f"c{self.n:03d}_H{len(done_h) + 1:02d}"
            ref_items[pid] = prompts.reproduce_prompt(
                "A method requested by the human scientists (turn their note into a full specification, "
                "keeping their intent):\n" + text, self.library("full"))
            meta[pid] = {"slot": None, "kind": "human", "source": name}
            done_h.append(name)
        out.update(self.llm.call_many("reproduce", self.system, ref_items, phase_tag=self.tag("reproduce")))

        for pid, text in out.items():
            body, _, note = text.partition("## Notebook")
            fm, _ = front_matter(body)
            m = meta[pid]
            self.reg["proposals"][pid] = {"cycle": self.n, "slot": m["slot"], "kind": m["kind"],
                                          "title": str(fm.get("title", pid))[:160], "status": "proposed",
                                          "family": None, "redirected": m.get("redirected", False),
                                          "predicted": fm.get("predicted"), "source": m.get("source")}
            d = pdir(pid, self.reg)
            d.mkdir(parents=True, exist_ok=True)
            (d / "PROPOSAL.md").write_text(body.strip() + "\n")
            if note.strip() and m["slot"]:
                nb = WORKSPACE / "agents" / m["slot"] / "notebook.md"
                nb.parent.mkdir(parents=True, exist_ok=True)
                with open(nb, "a") as fh:
                    fh.write(f"\n## Cycle {self.n}\n{note.strip()}\n")
        self.save()
        log(f"propose: {len(out)} proposal(s)")

    def classify(self) -> None:
        todo = [p for p in self.this_cycle() if not self.reg["proposals"][p].get("family")]
        fams = "\n".join(f"{f}: {v['name']} -- {v['description']}" for f, v in self.reg["families"].items())
        existing = "\n".join(f"{pid} [{p.get('family')}]: {p.get('summary') or p.get('title')}"
                             for pid, p in self.reg["proposals"].items() if p.get("family"))[-30000:]
        items = {pid: prompts.classify_prompt(pid, read(pdir(pid, self.reg) / "PROPOSAL.md", 30000), fams, existing)
                 for pid in todo}
        out = self.llm.call_many("classify", self.system, items, prompts.CLASSIFY_SCHEMA, self.tag("classify"))
        for pid in todo:
            r = out.get(pid) or {"family": "F10", "novelty": 0.5, "one_line_summary": ""}
            fam = str(r.get("family", "")).strip()
            if fam not in self.reg["families"]:
                name = (r.get("new_family_name") or "").strip()
                same = [f for f, v in self.reg["families"].items() if v["name"].lower() == name.lower()]
                fam = same[0] if same else (registry.add_family(
                    self.reg, name, r.get("new_family_description", ""), self.n) if name else "F10")
            p = self.reg["proposals"][pid]
            p["family"], p["novelty"] = fam, float(r.get("novelty", 0.5))
            p["summary"] = r.get("one_line_summary", "")
            p["closest"] = r.get("closest_existing", [])
        self.save()
        log(f"classify: {len(todo)} classified; families now {len(self.reg['families'])}")

    # ---- debate: skeptics <-> proposer, `debate_rounds` times ------------------------------------
    def _finalize(self, pid: str, text: str) -> None:
        d = pdir(pid, self.reg)
        (d / "SPEC.md").write_text(text.strip() + "\n")
        fm, _ = front_matter(text)
        p = self.reg["proposals"][pid]
        p["title"] = str(fm.get("title", p["title"]))[:160]
        p["predicted"] = fm.get("predicted", p.get("predicted"))
        par = fm.get("parents") or []
        p["parents"] = [str(x) for x in par if str(x) in self.reg["proposals"]] if isinstance(par, list) else []
        req = fm.get("requires") or []
        p["requires"] = [str(x) for x in (req if isinstance(req, list) else [req])]
        p["status"] = "specified"

    def debate(self) -> None:
        R = max(1, int(self.c.get("debate_rounds", 1)))
        sks = self.cfg["skeptics"]
        for r in range(1, R + 1):
            active = [pid for pid in self.this_cycle() if not (pdir(pid, self.reg) / "SPEC.md").exists()]
            if not active:
                break
            cur = {pid: read(pdir(pid, self.reg) / ("PROPOSAL.md" if r == 1 else f"DRAFT_r{r - 1}.md"))
                   for pid in active}
            items = {}
            for pid in active:
                d = pdir(pid, self.reg)
                for sk in sks:
                    if not (d / f"CRITIQUE_r{r}_{sk}.md").exists():
                        items[f"{pid}@{sk}"] = prompts.critique_prompt(
                            sk, pid, cur[pid], r, R, read(d / f"CRITIQUE_r{r - 1}_{sk}.md") if r > 1 else "")
            out = self.llm.call_many("critique", self.system, items, phase_tag=self.tag(f"critique_r{r}"))
            for key, text in out.items():
                pid, sk = key.split("@")
                (pdir(pid, self.reg) / f"CRITIQUE_r{r}_{sk}.md").write_text(
                    f"# Review of {pid} by {sk} (round {r})\n\n{text}\n")
            items, settled = {}, 0
            for pid in active:
                d = pdir(pid, self.reg)
                crits = [read(d / f"CRITIQUE_r{r}_{sk}.md") for sk in sks]
                if all(_verdict(c) == "SETTLED" for c in crits):      # nothing left to argue about
                    self._finalize(pid, cur[pid])
                    settled += 1
                elif not (d / f"DRAFT_r{r}.md").exists():
                    items[pid] = prompts.revise_prompt(self.slots.get(self.reg["proposals"][pid].get("slot")),
                                                       pid, cur[pid], "\n\n".join(c for c in crits if c), r, R)
            out = self.llm.call_many("revise", self.system, items, phase_tag=self.tag(f"revise_r{r}"))
            for pid in items:
                (pdir(pid, self.reg) / f"DRAFT_r{r}.md").write_text((out.get(pid) or cur[pid]).strip() + "\n")
            if r == R:
                for pid in active:
                    d = pdir(pid, self.reg)
                    if not (d / "SPEC.md").exists():
                        self._finalize(pid, read(d / f"DRAFT_r{r}.md"))
            self.save()
            log(f"debate round {r}/{R}: {len(active)} proposal(s), {settled} settled, {len(items)} revised")
        self._debate_summary()

    def _debate_summary(self) -> None:
        """How much each round still changed: the evidence for tuning cycle.debate_rounds."""
        root = WORKSPACE / "proposals" / f"c{self.n:03d}"
        rows, per = defaultdict(lambda: {"n": 0, "open": 0, "viol": 0, "chg": []}), []
        for pid in self.this_cycle():
            d = pdir(pid, self.reg)
            prev, line, r = read(d / "PROPOSAL.md"), [], 1
            while list(d.glob(f"CRITIQUE_r{r}_*.md")):
                crits = [read(c) for c in sorted(d.glob(f"CRITIQUE_r{r}_*.md"))]
                n_open = sum(_verdict(c) != "SETTLED" for c in crits)
                row = rows[r]
                row["n"] += 1
                row["open"] += n_open
                row["viol"] += sum("RULES: VIOLATION" in c.upper() for c in crits)
                draft = read(d / f"DRAFT_r{r}.md")
                if draft:
                    chg = 1.0 - difflib.SequenceMatcher(None, prev.split(), draft.split(), autojunk=False).quick_ratio()
                    row["chg"].append(chg)
                    line.append(f"r{r}: {n_open} open, {100 * chg:.0f}% changed")
                    prev = draft
                else:
                    line.append(f"r{r}: {n_open} open, settled")
                r += 1
            per.append(f"| `{pid}` | " + " / ".join(line) + " |")
        L = [f"# Debate summary, cycle {self.n}", "",
             f"`cycle.debate_rounds` = {self.c.get('debate_rounds', 1)}. A reviewer verdict is OPEN while it still "
             "has a substantive point; a proposal is settled early when all reviewers say SETTLED. 'Text changed' "
             "is the share of words that differ between consecutive drafts. Rounds in which few verdicts are "
             "open and little text changes are not buying anything.", "",
             "| round | proposals debated | open reviewer verdicts | rule violations flagged | mean text changed | max text changed |",
             "|---|---|---|---|---|---|"]
        for r in sorted(rows):
            v = rows[r]
            mean = f"{100 * sum(v['chg']) / len(v['chg']):.0f}%" if v["chg"] else "-"
            mx = f"{100 * max(v['chg']):.0f}%" if v["chg"] else "-"
            L.append(f"| {r} | {v['n']} | {v['open']} of {v['n'] * len(self.cfg['skeptics'])} | {v['viol']} | {mean} | {mx} |")
        L += ["", "## Per proposal", "", "| proposal | rounds |", "|---|---|"] + per
        if root.exists():
            (root / "DEBATE_SUMMARY.md").write_text("\n".join(L) + "\n")

    def moderate(self) -> None:
        sel_file = WORKSPACE / "selected" / f"c{self.n:03d}_SELECTION.md"
        if sel_file.exists():
            return
        cands = self.this_cycle(kinds=("idea",))
        forced = self.this_cycle(kinds=("reference", "human"))
        blocked, rules = set(), []
        for pid in cands:
            viol = [c.name for c in last_critiques(pdir(pid, self.reg)) if "RULES: VIOLATION" in read(c).upper()]
            rules.append(f"{pid}: " + (f"VIOLATION flagged by {len(viol)} reviewer(s)" if viol else "OK"))
            if len(viol) >= 2:
                blocked.add(pid)
        specs = "\n\n".join(f"<proposal id=\"{pid}\" family=\"{self.reg['proposals'][pid]['family']}\" "
                            f"novelty=\"{self.reg['proposals'][pid].get('novelty', 0.5):.2f}\">\n"
                            f"{spec_text(pid, self.reg)[:26000]}\n</proposal>" for pid in cands)
        human = "\n\n".join(f"[{k}]\n{v}" for k, v in self.human_suggestions().items())
        early = self.n <= self.c["independence_cycles"]
        K = self.c["select_per_cycle"]
        items = {f"M{k}": prompts.moderate_prompt(k, K, specs, registry.render(self.reg), "\n".join(rules), human, early)
                 for k in range(1, self.cfg["moderators"] + 1)} if cands else {}
        out = self.llm.call_many("moderate", self.system, items, prompts.MODERATE_SCHEMA, self.tag("moderate"))

        borda, why = Counter(), defaultdict(list)
        for mk, r in out.items():
            order = [x["id"] for x in r.get("ranking", []) if x["id"] in cands]
            order += [c for c in cands if c not in order]
            for rank, pid in enumerate(order):
                borda[pid] += len(cands) - rank
            for x in r.get("ranking", []):
                why[x["id"]].append(f"{mk}: {x['rationale']}")
        ranked = sorted((c for c in cands if c not in blocked), key=lambda c: (-borda[c], c))
        stats = registry.family_stats(self.reg)
        fam = lambda pid: self.reg["proposals"][pid]["family"]
        cap = max(1, math.ceil(self.c["max_family_share"] * K))
        chosen, reason, per_fam = [], {}, Counter()
        untried = [p for p in ranked if stats.get(fam(p), {}).get("evaluated", 0) == 0
                   and stats.get(fam(p), {}).get("selected", 0) == 0]
        for pid in sorted(untried, key=lambda p: (-self.reg["proposals"][p].get("novelty", 0.5), -borda[p])):
            if len(chosen) < self.c["wildcard_slots"] and per_fam[fam(pid)] < 1:
                chosen.append(pid)
                per_fam[fam(pid)] += 1
                reason[pid] = "wildcard: novel idea in a family with no selected or evaluated member"
        for pid in ranked:
            if len(chosen) >= K:
                break
            if pid in chosen:
                continue
            if per_fam[fam(pid)] >= cap:
                reason[pid] = f"not selected: family {fam(pid)} already holds its maximum of {cap} places"
                continue
            chosen.append(pid)
            per_fam[fam(pid)] += 1
            reason[pid] = "selected on moderators' combined ranking"
        for pid in ranked:                                   # caps left places unfilled: relax them
            if len(chosen) >= K:
                break
            if pid not in chosen:
                chosen.append(pid)
                reason[pid] = "selected (family cap relaxed: too few proposals in other families)"
        for pid in cands:
            reason.setdefault(pid, "blocked: rule violation flagged by two or more reviewers" if pid in blocked
                              else "not selected this cycle (ranked below the cut)")
        for pid in forced:
            chosen.append(pid)
            reason[pid] = "selected automatically (reference reproduction or human priority request)"
        for pid in cands + forced:
            p = self.reg["proposals"][pid]
            p["selected"] = pid in chosen
            p["status"] = "selected" if pid in chosen else "not_selected"
            p["decision"] = reason[pid]
            p["moderator_notes"] = why.get(pid, [])
        self.reg["redirects"] = registry.compute_redirects(
            self.reg, self.n, self.cfg["proposer_slots"], self.c["redirect_share"], self.c["redirect_window_cycles"])
        topics = [t for r in out.values() for t in r.get("seminar_topics", [])]
        self.st["seminar_topics"] = topics[0::2][:2] + topics[1::2][:2]

        L = [f"# Selection for cycle {self.n}", "",
             f"{len(chosen)} specification(s) go to implementers. Places per family are capped at {cap}; "
             f"{self.c['wildcard_slots']} place(s) are reserved for novel ideas in untried families.", "",
             "| proposal | family | title | combined rank score | decision |", "|---|---|---|---|---|"]
        for pid in sorted(cands + forced, key=lambda c: (c not in chosen, -borda[c], c)):
            p = self.reg["proposals"][pid]
            L.append(f"| `{pid}` | {p['family']} | {p['title'][:60]} | {borda[pid] or '-'} | {reason[pid]} |")
        L += ["", "## Moderators' rationales", ""]
        for pid in cands:
            L += [f"**{pid}**"] + [f"- {w}" for w in why.get(pid, [])] + [""]
        for mk, r in out.items():
            L += [f"## {mk}: portfolio comment", "", r.get("portfolio_comment", ""), "",
                  f"## {mk}: response to human suggestions", "", r.get("human_suggestions_response", ""), ""]
        if self.reg["redirects"]:
            L += ["## Redirects for next cycle (computed from family shares)", ""]
            L += [f"- slot {s}: asked to explore {r['family']} ({self.reg['families'][r['family']]['name']}), "
                  f"away from over-represented {r['from_family']}" for s, r in self.reg["redirects"].items()]
        sel_file.parent.mkdir(exist_ok=True)
        sel_file.write_text("\n".join(L) + "\n")
        self.save()
        save_state(self.st)
        log(f"moderate: selected {len(chosen)} of {len(cands) + len(forced)}")

    # ---- implementation -------------------------------------------------------------------------
    def _alg_files(self, alg_dir: Path) -> str:
        parts = []
        for p in sorted(alg_dir.iterdir()):
            if p.suffix in (".py", ".yaml", ".md") and p.name not in ("SPEC.md", "REVIEW.md"):
                parts.append(f"===== {p.name} =====\n{read(p, 60000)}")
        if (alg_dir / "native").is_dir():
            for p in sorted((alg_dir / "native").iterdir()):
                if p.is_file():
                    parts.append(f"===== native/{p.name} =====\n{read(p, 60000)}")
        return "\n\n".join(parts)

    @staticmethod
    def _missing_requirements(p: dict) -> list[str]:
        """Packages / languages a specification needs that the cluster environment does not offer yet."""
        a = hconfig.load()["audit"]
        nat = a.get("native") or {}
        have = {str(x).lower() for x in a["allowed_imports"]} | {str(x).lower() for x in nat.get("languages", [])} \
            | {str(x).lower() for x in nat.get("extra_imports", [])} | {"python", "c++"}
        if "cpp" not in have:
            have.discard("c++")
        out = []
        for r in p.get("requires") or []:
            name = re.split(r"[\s<>=!~\[(]", str(r).strip().lower(), 1)[0].replace("-", "_")
            if name and name not in have and name.split(".")[0] not in have:
                out.append(str(r))
        return out

    def _implement_one(self, pid: str) -> str:
        p = self.reg["proposals"][pid]
        alg_dir = WORKSPACE / "algorithms" / pid
        alg_dir.mkdir(parents=True, exist_ok=True)
        (alg_dir / "SPEC.md").write_text(spec_text(pid, self.reg))
        parents = [x for x in p.get("parents", []) if (WORKSPACE / "algorithms" / x).exists()]
        refs = [f"algorithms/{d.name}" for d in (WORKSPACE / "algorithms").glob("baseline_*")]
        ts = Toolset(read_roots=["docs", "literature"] + refs + [f"algorithms/{x}" for x in parents],
                     write_roots=[f"algorithms/{pid}"], alg_dir=alg_dir, allow_tests=True)
        review, toy_ok, toy_report = {}, False, ""
        try:
            self.llm.agent("implement", self.system, prompts.implement_prompt(pid, parents), ts,
                           self.c["implement_max_turns"], tag=f"{self.tag('implement')}:{pid}")
            for rnd in range(self.c["implement_fix_rounds"] + 1):
                if not (alg_dir / "algorithm.py").exists():
                    toy_ok, toy_report = False, "algorithm.py was never written"
                else:
                    self._fix_manifest(alg_dir, p)
                    toy_report, toy_ok = run_toy(alg_dir)
                review = self.llm.call("review", self.system, prompts.review_prompt(
                    pid, read(alg_dir / "SPEC.md", 40000), self._alg_files(alg_dir), toy_report),
                    prompts.REVIEW_SCHEMA, tag=f"{self.tag('review')}:{pid}#{rnd}")
                (alg_dir / "REVIEW.md").write_text(
                    f"# Code review of {pid} (round {rnd + 1})\n\nVerdict: **{review['verdict']}**; toy run "
                    f"{'completed' if toy_ok else 'FAILED'}\n\n{review['summary']}\n\n"
                    + "\n".join(f"- [{k.split('_')[0]}] {i}" for k in ("integrity_issues", "fidelity_issues",
                                                                       "correctness_issues") for i in review.get(k, []))
                    + "\n")
                if (review["verdict"] == "pass" and toy_ok) or review["verdict"] == "reject" \
                        or rnd == self.c["implement_fix_rounds"]:
                    break
                if not toy_ok:
                    review.setdefault("correctness_issues", []).append(
                        "The toy run does not complete. Report:\n" + toy_report[-3000:])
                self.llm.agent("implement", self.system, prompts.fix_prompt(pid, review), ts,
                               self.c["implement_max_turns"] // 2, tag=f"{self.tag('fix')}:{pid}#{rnd}")
        except Refused as e:
            review = {"verdict": "reject", "summary": f"model declined: {e}"}
        except (BudgetExceeded, anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            raise
        except Exception as e:                      # one broken implementation must not stop the cycle
            log(f"implement {pid}: {type(e).__name__}: {e}")
            review = {"verdict": "reject", "summary": f"orchestrator error: {type(e).__name__}: {e}"}
        if review.get("verdict") == "pass" and toy_ok:
            promote.enqueue(pid, "screen")
            status = "queued:screen"
        else:
            status = "implementation_failed"
        with _reg_lock:
            p["status"] = status
        self.save()
        return status

    @staticmethod
    def _fix_manifest(alg_dir: Path, p: dict) -> None:
        """Make sure algorithm.yaml exists, parses and carries the registry's family id."""
        mf = alg_dir / "algorithm.yaml"
        try:
            man = yaml.safe_load(mf.read_text()) if mf.exists() else {}
        except yaml.YAMLError:
            man = {}
        man = man if isinstance(man, dict) else {}
        man.setdefault("name", p.get("title", alg_dir.name))
        man["family"] = p.get("family")
        man.setdefault("resources", {})
        mf.write_text(yaml.safe_dump(man, sort_keys=False))

    def implement(self) -> None:
        todo, waiting = [], []
        for pid, p in self.reg["proposals"].items():      # includes earlier cycles' specs that were waiting
            if not p.get("selected") or p.get("status") not in ("selected", "waiting_for_environment"):
                continue
            miss = self._missing_requirements(p)
            if miss:
                p["status"] = "waiting_for_environment"
                waiting.append((pid, miss))
            else:
                p["status"] = "selected"
                todo.append(pid)
        d = WORKSPACE / "environment_requests"
        d.mkdir(exist_ok=True)
        L = ["# Environment requests (for the human who maintains the cluster environment)", "",
             "_Generated each cycle. A specification listed here is selected but NOT implemented until what "
             "it needs is available: install it in the cluster environment (and the sandbox container, and "
             "the orchestrator host for the toy self-test), then add the import name to "
             "`audit.allowed_imports`, or the language to `audit.native.languages`, in `config/harness.yaml`. "
             "It is then implemented in the next cycle automatically._", ""]
        L += [f"- `{pid}` ({self.reg['proposals'][pid].get('title', '')[:70]}) needs: **{', '.join(m)}**"
              for pid, m in waiting] or ["Nothing is waiting."]
        (d / "PENDING.md").write_text("\n".join(L) + "\n")
        self.save()
        if waiting:
            log(f"implement: {len(waiting)} specification(s) waiting for environment: "
                + ", ".join(f"{pid} [{', '.join(m)}]" for pid, m in waiting))
        if not todo:
            return
        with ThreadPoolExecutor(self.cfg["api"]["max_concurrency"]) as ex:
            results = list(ex.map(self._implement_one, todo))
        log("implement: " + ", ".join(f"{k}={v}" for k, v in Counter(results).items()))

    def crosspollinate(self) -> None:
        if self.n <= self.c["independence_cycles"] or self.n % self.c["crosspollinate_every"] != 0:
            return
        stats = registry.family_stats(self.reg)
        tops = [(f, s["best_alg"]) for f, s in stats.items() if s["best_alg"]]
        if len(tops) < 2:
            return
        parts = []
        for f, pid in tops:
            p = self.reg["proposals"][pid]
            b = registry.best_result(p)
            pms = sorted((WORKSPACE / "postmortems").glob(f"{pid}__*.md"))
            parts.append(f"### Family {f} ({self.reg['families'][f]['name']}): {pid} \"{p['title']}\"\n"
                         f"{result_line(pid, b['stage'], b)}\n\nSpecification (start):\n{spec_text(pid, self.reg)[:5000]}"
                         f"\n\nPost-mortem:\n{read(pms[-1], 5000) if pms else '(none yet)'}")
        sems = sorted((WORKSPACE / "seminars").glob("*.md"))[-2:]
        parts += [read(s, 6000).split("<details>")[0] for s in sems]
        ids = sorted(self.slots)
        k = max(1, round(self.c["informed_fraction"] * len(ids)))
        text = self.llm.call("crosspollinate", self.system,
                             prompts.crosspollinate_prompt("\n\n".join(parts)[:160000], k), tag=self.tag("crosspollinate"))
        briefs = [b.strip() for b in text.split("=====") if b.strip()]
        d = WORKSPACE / "crosspollination"
        d.mkdir(exist_ok=True)
        targets = [ids[((self.n + 1) * k + j + 1) % len(ids)] for j in range(len(briefs))]
        self.reg["assignments"] = {}
        for j, (b, sid) in enumerate(zip(briefs, targets), 1):
            path = d / f"c{self.n:03d}_brief_{j}.md"
            path.write_text(f"# Cross-pollination brief {j} (cycle {self.n}, addressed to {sid})\n\n{b}\n")
            self.reg["assignments"][sid] = {"cycle": self.n + 1, "path": str(path.relative_to(WORKSPACE))}
        self.save()
        log(f"crosspollinate: {len(briefs)} brief(s)")

    def digest(self) -> None:
        dest = WORKSPACE / "digests" / f"c{self.n:03d}_DIGEST.md"
        spent = self.llm.ledger.spent(self.n)
        stats = registry.family_stats(self.reg)
        success = read(STATE / "success.json") or "{}"
        rows = [f"- {pid} | {p.get('kind')} | slot {p.get('slot')} | family {p.get('family')} | {p.get('title')} "
                f"| status {p.get('status')} | spec: proposals/c{self.n:03d}/{pid}/SPEC.md"
                for pid, p in sorted(self.reg["proposals"].items()) if p["cycle"] == self.n]
        newres = [result_line(pid, s, r) for pid, p in sorted(self.reg["proposals"].items())
                  for s, r in (p.get("results") or {}).items()]
        reqs = sorted((WORKSPACE / "diagnostic_requests").glob(f"c{self.n:03d}_*.md"))
        facts = "\n".join([
            f"cycle: {self.n}", f"date: {time.strftime('%Y-%m-%d')}",
            f"spending USD: this cycle {spent['cycle']:.2f}, this month {spent['month']:.2f}, total {spent['total']:.2f}"
            f" (limits: {self.cfg['budget']})",
            f"goal status (from harness): {success}",
            "proposals this cycle:", *rows,
            "all evaluation results so far:", *(newres or ["(none yet)"]),
            "family portfolio (proposed/selected/evaluated/passed gate/best effective rms):",
            *[f"- {f} {self.reg['families'][f]['name']}: {s['proposed']}/{s['selected']}/{s['evaluated']}/"
              f"{s['gate_pass']}/{fmt(s['best_eff'])}" for f, s in stats.items()],
            "redirects for next cycle: " + json.dumps(self.reg.get("redirects", {})),
            "requests to humans: " + (", ".join(str(r.relative_to(WORKSPACE)) for r in reqs) or "none"),
            f"selection document: selected/c{self.n:03d}_SELECTION.md",
            "pending evaluation tickets: " + str(len(list((STATE / 'eval_queue').glob('*.json'))))
            + " queued, " + str(len(list((STATE / 'eval_submitted').glob('*.json')))) + " running",
        ])
        try:
            text = self.llm.call("digest", "You write concise, accurate status reports for scientists.",
                                 prompts.digest_prompt(facts), tag=self.tag("digest"))
        except (Refused, ValueError):
            text = "```\n" + facts + "\n```"
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(f"# Digest, cycle {self.n}\n\n{text}\n")
        self.sync_results()
        log(f"digest: {dest.relative_to(ROOT)}")

    def submit(self) -> None:
        cmd = (self.c.get("submit_command") or "").strip()
        if not cmd:
            log("submit: no submit_command configured -- run bin/submit_eval.sh on a node with sbatch")
            return
        pr = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True, text=True)
        log("submit:", (pr.stdout + pr.stderr).strip()[-800:])

    # ------------------------------------------------------------------------------ driver
    def run(self, only: str | None = None) -> None:
        log(f"=== cycle {self.n} ===  spent so far: ${self.llm.ledger.spent(self.n)['total']:.2f}")
        for ph in PHASES:
            if only and ph != only:
                continue
            if not only and ph in self.st["done"]:
                continue
            log(f"--- phase {ph}")
            getattr(self, ph)()
            if not only:
                self.st["done"].append(ph)
                save_state(self.st)
        if not only:
            self.st.update(cycle=self.n + 1, done=[], last_finished=time.time())
            self.st.pop("idle_since", None)
            save_state(self.st)
            log(f"=== cycle {self.n} complete ===")


# ================================================================================================
def pending_tickets(stages=None) -> list[str]:
    """Evaluations queued or running on the cluster, as 'alg/stage' strings."""
    out = []
    for d in ("eval_queue", "eval_submitted"):
        for p in (STATE / d).glob("*.json"):
            try:
                t = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):               # ticket being moved right now
                continue
            if stages is None or t.get("stage") in stages:
                out.append(f"{t.get('alg_id')}/{t.get('stage')}")
    return sorted(out)


def hold_reason(cfg: dict, st: dict) -> str | None:
    """Why the next cycle must not start yet (None: start it). Purely mechanical: tickets and clocks."""
    c, now = cfg["cycle"], time.time()
    since = st.get("last_finished", 0)
    left = since + c.get("min_days_between_cycles", 0) * 86400 - now
    if left > 0:
        return f"next cycle in {left / 3600:.1f} h (min_days_between_cycles)"
    mode = c.get("wait_for_results", "all")
    if mode == "none":
        return None
    pend = pending_tickets(("screen",) if mode == "screen" else None)
    if now - since > c.get("max_days_waiting", 10) * 86400:
        if pend:
            log(f"max_days_waiting reached with {len(pend)} evaluation(s) still out; starting the next cycle anyway")
        return None
    if pend:
        if st.pop("idle_since", None) is not None:
            save_state(st)
        queued = len(list((STATE / "eval_queue").glob("*.json")))
        hint = (f" ({queued} not yet submitted: is bin/submit_eval.sh being run?)"
                if queued and not (c.get("submit_command") or "").strip() else "")
        return f"waiting for {len(pend)} evaluation(s){hint}: " + ", ".join(pend[:6]) + (" ..." if len(pend) > 6 else "")
    if mode == "all":                                             # a finished stage may still be promoted
        if "idle_since" not in st:
            st["idle_since"] = now
            save_state(st)
        settle = c.get("settle_minutes", 45) * 60 - (now - st["idle_since"])
        if settle > 0:
            return f"all evaluations reported; allowing {settle / 60:.0f} more min for promotions to be queued"
    return None


def init() -> None:
    for d in ("docs", "literature", "proposals", "selected", "algorithms", "results", "postmortems", "registry",
              "human_suggestions", "diagnostic_requests", "environment_requests", "digests", "agents", "crosspollination", "seminars"):
        (WORKSPACE / d).mkdir(parents=True, exist_ok=True)
    readme = WORKSPACE / "human_suggestions" / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Human suggestions\n\nDrop a Markdown file here to steer the collaboration.\n\n"
            "- `anything.md` -- shown to the moderators every cycle (they must respond in the selection "
            "document) and to a rotating third of the proposers.\n"
            "- `PRIORITY_anything.md` -- turned into a full specification and implemented in the next cycle, "
            "outside the normal quota.\n\nMove a file to `archive/` when it has served its purpose.\n")
    reg = registry.load()
    for src in sorted((ROOT / "baselines").iterdir()):
        dst = WORKSPACE / "algorithms" / src.name
        if not dst.exists():
            shutil.copytree(src, dst)
        if src.name not in reg["proposals"]:
            reg["proposals"][src.name] = {"cycle": 0, "slot": None, "kind": "reference", "family": None,
                                          "title": (src / "algorithm.py").read_text().split('"""')[1].split("\n")[0],
                                          "status": "queued:screen", "selected": True}
        if not (WORKSPACE / "results" / src.name).exists():
            promote.enqueue(src.name, "screen")
    registry.save(reg)
    if not STATE_FILE.exists():
        save_state(load_state())
    print("initialised. Reference algorithms queued for evaluation: run bin/submit_eval.sh on the cluster.")


def status() -> None:
    st, cfg = load_state(), load_cfg()
    from .llm import Ledger
    led = Ledger(STATE / "ledger.jsonl", cfg["pricing"], cfg["budget"])
    s = led.spent(st["cycle"])
    print(f"cycle {st['cycle']}; phases done: {st['done'] or 'none'}")
    print(f"spent: cycle ${s['cycle']:.2f} / month ${s['month']:.2f} / total ${s['total']:.2f}  limits {cfg['budget']}")
    print(f"tickets: {len(list((STATE / 'eval_queue').glob('*.json')))} queued, "
          f"{len(list((STATE / 'eval_submitted').glob('*.json')))} submitted")
    print(read(STATE / "success.json"))
    lb = WORKSPACE / "registry" / "LEADERBOARD.md"
    print(read(lb, 6000))


def costs() -> None:
    """Spending by kind of call, from state/ledger.jsonl: the basis for scaling up."""
    rows = [json.loads(l) for l in read(STATE / "ledger.jsonl", 10**9).splitlines() if l.strip()]
    if not rows:
        print("no API calls recorded yet")
        return
    agg: dict = {}
    for r in rows:
        a = agg.setdefault((r["cycle"], r["role"], r["model"]), {"calls": 0, "tags": set(), "in": 0, "out": 0,
                                                                 "cr": 0, "cw": 0, "ws": 0, "usd": 0.0})
        a["calls"] += 1
        a["tags"].add(r["tag"])
        for k, f in (("in", "in"), ("out", "out"), ("cr", "cache_read"), ("cw", "cache_write"), ("ws", "web_searches")):
            a[k] += r.get(f, 0)
        a["usd"] += r["usd"]
    print("'units' = distinct jobs (one proposal, one critique, one implementation ...); 'USD/unit' is what to scale.")
    print(f"{'cycle':>5} {'role':<15} {'model':<19} {'calls':>6} {'units':>6} {'in kTok':>9} {'cache-rd kTok':>13} "
          f"{'cache-wr kTok':>13} {'out kTok':>9} {'searches':>8} {'USD':>9} {'USD/unit':>9}")
    for (cyc, role, model), a in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1]["usd"])):
        n = len(a["tags"])
        print(f"{cyc:>5} {role:<15} {model:<19} {a['calls']:>6} {n:>6} {a['in']/1e3:>9.1f} {a['cr']/1e3:>13.1f} "
              f"{a['cw']/1e3:>13.1f} {a['out']/1e3:>9.1f} {a['ws']:>8} {a['usd']:>9.2f} {a['usd']/n:>9.2f}")
    print(f"total: ${sum(a['usd'] for a in agg.values()):.2f} in {len(rows)} calls "
          "(local estimate from config/swarm.yaml prices; the Console usage page is authoritative)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "run", "loop", "status", "costs"])
    ap.add_argument("--only", choices=PHASES)
    a = ap.parse_args(argv)
    if a.cmd == "init":
        return init()
    if a.cmd == "status":
        return status()
    if a.cmd == "costs":
        return costs()
    try:
        if a.cmd == "run":
            Cycle().run(a.only)
            return
        while True:
            cfg, st = load_cfg(), load_state()
            if st.get("last_finished") and not st["done"]:        # between cycles
                c = Cycle()
                c.sync_results()
                c.submit()
                why = hold_reason(cfg, st)
                if why:
                    log(why)
                    time.sleep(max(0.01, cfg["cycle"].get("poll_minutes", 15)) * 60)
                    continue
            try:
                Cycle().run()
            except BudgetExceeded as e:
                if "usd_per_month" not in str(e):
                    raise
                log(f"PAUSED: {e}. Resuming when the calendar month changes (or raise the limit).")
                time.sleep(6 * 3600)
    except BudgetExceeded as e:
        log(f"STOPPED: {e}. State is saved; raise the limit in config/swarm.yaml and re-run to continue.")
        sys.exit(2)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        log(f"STOPPED: the API rejected the credentials or the account limit was reached ({e}). "
            "State is saved; fix the key / limit and re-run to continue.")
        sys.exit(3)


if __name__ == "__main__":
    main()

"""Canned replies for offline tests of the orchestration (RVSWARM_MOCK_LLM=1). No science here."""
from __future__ import annotations

import hashlib
import re

PROPOSAL = """---
title: Mock proposal {tag}
family_hint: indicator decorrelation
parents: []
uses: [indices]
resources: {{cpus: 2, mem_gb: 8, gpus: 0, hours: 1}}
predicted: {{rms_null_daily_ms: 0.5, min_transfer: 0.97, confidence: 0.3}}
---
# Mock proposal {tag}

## 1. The idea
Decorrelate the pipeline RV against FWHM, BIS and Ca II H&K.

## 3. Algorithm specification
1. Standardise the three indicators. 2. Least-squares fit on daily means. 3. Subtract per observation.

## Notebook
Remember this was a mock run.
"""

ALGO = '''"""Mock implementation: linear decorrelation against three indicators."""
import numpy as np


def run(ctx):
    ds = ctx.target
    X = np.column_stack([ds.scalar("FWHMMOD"), ds.scalar("BISMOD"), ds.activity("CaIIHK")[0]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    A = np.column_stack([X, np.ones(ds.n_obs)])
    coef, *_ = np.linalg.lstsq(A, ds.rv_pipeline, rcond=None)
    ctx.metric("coef_fwhm", coef[0])
    return {"rv_clean": ds.rv_pipeline - X @ coef[:3]}
'''


def _h(s: str, n: int) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % n


def respond(role: str, user: str, schema, tag: str):
    if role == "classify":
        return {"family": f"F{1 + _h(tag, 9):02d}", "new_family_name": "", "new_family_description": "",
                "novelty": _h(tag, 100) / 100, "closest_existing": [], "one_line_summary": f"mock summary {tag}"}
    if role == "moderate":
        ids = re.findall(r'<proposal id="([^"]+)"', user)
        ids = sorted(ids, key=lambda i: _h(i + tag, 1000))
        return {"ranking": [{"id": i, "rationale": "mock rationale"} for i in ids],
                "portfolio_comment": "mock portfolio comment", "human_suggestions_response": "mock response",
                "seminar_topics": ["Why does decorrelation plateau?"]}
    if role == "review":
        return {"verdict": "pass", "fidelity_issues": [], "correctness_issues": [], "integrity_issues": [],
                "summary": "mock review: fine"}
    if role == "numerics":
        return {"verdict": "pass", "numerical_issues": [], "summary": "mock numerics review: fine"}
    if role == "performance":
        opt = _h(tag, 2) == 0
        return {"verdict": "optimise" if opt else "ok",
                "changes": ["mock: read each spectrum once"] if opt else [],
                "recommended_resources": {"cpus": 1, "mem_gb": 4, "hours": 0.5, "parallel_runs": False},
                "estimated_core_hours_screen": 0.1, "summary": "mock performance advice"}
    if role in ("propose", "reproduce", "revise"):
        extra = f"\n\n## Response to reviewers\nmock revision {_h(user, 10**6)}\n" if role == "revise" else ""
        return PROPOSAL.format(tag=tag.split(":")[-1]) + extra
    if role == "critique":
        settled = "review round 1 " not in user and _h(tag, 3) != 0
        return f"VERDICT: {'SETTLED' if settled else 'OPEN'}\n1. Failure modes: mock.\n5. RULES: OK"
    return f"(mock {role} text for {tag})"


def agent(role: str, user: str, toolset, tag: str) -> str:
    if role == "literature":
        for name in ("scalpels", "fiesta", "lbl"):
            toolset.execute("write_file", {"path": f"literature/{name}.md", "content": f"# {name}\nmock card\n"})
        toolset.execute("write_file", {"path": "literature/INDEX.md", "content": "# Index\n- scalpels\n- fiesta\n- lbl\n"})
        return "mock library written"
    m = re.search(r"algorithms/([A-Za-z0-9_]+)/", user)
    alg = m.group(1)
    if "asked for changes" not in user and "performance engineer recommends" not in user:
        toolset.execute("write_file", {"path": f"algorithms/{alg}/algorithm.py", "content": ALGO})
        toolset.execute("write_file", {"path": f"algorithms/{alg}/algorithm.yaml",
                                       "content": "name: mock\nfamily: F03\nresources: {cpus: 2, mem_gb: 8, hours: 1, gpus: 0}\n"})
        toolset.execute("write_file", {"path": f"algorithms/{alg}/IMPLEMENTATION_NOTES.md", "content": "mock notes\n"})
        toolset.execute("write_file", {"path": "docs/HACK.md", "content": "should be refused"})
    if toolset.allow_tests:
        toolset.execute("run_tests", {})
    return "mock implementation done"

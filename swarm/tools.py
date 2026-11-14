"""File tools handed to tool-using agents. Every path is confined to explicit roots
inside `workspace/`; the harness, its configuration and the data are unreachable."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RVSWARM_ROOT") or Path(__file__).resolve().parents[1])
WORKSPACE = ROOT / "workspace"


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


class Toolset:
    """read_roots / write_roots are directories relative to workspace/."""

    def __init__(self, read_roots: list[str], write_roots: list[str], alg_dir: Path | None = None,
                 allow_tests: bool = False):
        self.read_roots = [(WORKSPACE / r).resolve() for r in read_roots + write_roots]
        self.write_roots = [(WORKSPACE / r).resolve() for r in write_roots]
        self.alg_dir = alg_dir
        self.allow_tests = allow_tests
        self.tests_run = 0
        self.last_test_ok = False

    # -- definitions ---------------------------------------------------------------------
    def definitions(self) -> list[dict]:
        s = {"type": "string"}
        d = [
            {"name": "list_files", "description": "List files under a directory of the shared workspace "
             "(path relative to workspace/).", "input_schema": _schema({"path": s}, ["path"])},
            {"name": "read_file", "description": "Read a text file from the shared workspace (path relative "
             "to workspace/). Optional 1-based line range.",
             "input_schema": _schema({"path": s, "start_line": {"type": "integer"},
                                      "end_line": {"type": "integer"}}, ["path"])},
            {"name": "write_file", "description": "Create or overwrite a text file inside your own writable "
             "directory (path relative to workspace/). Keep each call under ~400 lines; use append_file for "
             "the rest of a long file.", "input_schema": _schema({"path": s, "content": s}, ["path", "content"])},
            {"name": "append_file", "description": "Append text to a file you created.",
             "input_schema": _schema({"path": s, "content": s}, ["path", "content"])},
            {"name": "edit_file", "description": "Replace one exact, unique occurrence of old_text by new_text "
             "in a file you may write.", "input_schema": _schema({"path": s, "old_text": s, "new_text": s},
                                                                 ["path", "old_text", "new_text"])},
        ]
        if self.allow_tests:
            d.append({"name": "run_tests", "description": "Run the static code audit and then your algorithm "
                      "end-to-end on the toy synthetic Sun (un-injected + one hidden-planet run), exactly as "
                      "the real harness would. Returns the audit result, the result report and the log tail. "
                      "Takes a few minutes for heavy algorithms.", "input_schema": _schema({}, [])})
        return d

    # -- execution -----------------------------------------------------------------------
    def _resolve(self, path: str, write: bool) -> Path:
        p = (WORKSPACE / str(path).lstrip("/")).resolve()
        roots = self.write_roots if write else self.read_roots
        if not any(p == r or r in p.parents for r in roots):
            allowed = ", ".join(str(r.relative_to(WORKSPACE)) for r in roots)
            raise PermissionError(f"'{path}' is outside what you may {'write' if write else 'read'}: {allowed}")
        return p

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        try:
            return getattr(self, "_t_" + name)(**(args or {})), False
        except AttributeError:
            return f"unknown tool {name}", True
        except Exception as e:                          # tool errors go back to the model as text
            return f"{type(e).__name__}: {e}", True

    def _t_list_files(self, path: str) -> str:
        p = self._resolve(path, False)
        if not p.is_dir():
            return f"{path} is not a directory"
        rows = [f"{'d' if q.is_dir() else '-'} {q.relative_to(WORKSPACE)}"
                + ("" if q.is_dir() else f"  ({q.stat().st_size} bytes)")
                for q in sorted(p.iterdir()) if not q.name.startswith(".")]
        return "\n".join(rows[:400]) or "(empty)"

    def _t_read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        p = self._resolve(path, False)
        lines = p.read_text(errors="replace").splitlines()
        end = min(len(lines), end_line or len(lines))
        body = "\n".join(lines[max(0, start_line - 1):end])
        if len(body) > 50000:
            body = body[:50000] + "\n... (truncated; read a line range)"
        return body

    def _t_write_file(self, path: str, content: str) -> str:
        p = self._resolve(path, True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} characters to {path}"

    def _t_append_file(self, path: str, content: str) -> str:
        p = self._resolve(path, True)
        with open(p, "a") as fh:
            fh.write(content)
        return f"appended {len(content)} characters to {path}"

    def _t_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        p = self._resolve(path, True)
        s = p.read_text()
        n = s.count(old_text)
        if n != 1:
            raise ValueError(f"old_text occurs {n} times; it must occur exactly once")
        p.write_text(s.replace(old_text, new_text))
        return "edited"

    def _t_run_tests(self) -> str:
        self.tests_run += 1
        text, ok = run_toy(self.alg_dir)
        self.last_test_ok = ok
        return text


def run_toy(alg_dir: Path, timeout_s: int = 2400) -> tuple[str, bool]:
    """Audit + toy evaluation in a subprocess. -> (report text, ran without error)."""
    out_root = ROOT / "state" / "toy_runs" / alg_dir.name
    shutil.rmtree(out_root, ignore_errors=True)
    work = ROOT / "state" / "toy_runs" / "_work"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), RVSWARM_ROOT=str(ROOT), RVSWARM_WORKDIR=str(work))
    env.pop("ANTHROPIC_API_KEY", None)
    try:
        pr = subprocess.run([sys.executable, "-m", "harness.evaluate", "--alg-dir", str(alg_dir),
                             "--stage", "toy", "--out-root", str(out_root)], cwd=str(ROOT), env=env,
                            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"run_tests timed out after {timeout_s} s: the algorithm is too slow on the toy data.", False
    runs = sorted(out_root.glob("toy_*"))
    if not runs:
        return "harness crashed before producing a report:\n" + (pr.stderr or pr.stdout)[-3000:], False
    report = (runs[-1] / "report.md").read_text()
    metrics = json.loads((runs[-1] / "metrics.json").read_text())
    ok = metrics.get("status") == "ok"
    note = ""
    if ok and not metrics.get("gate_pass"):
        note = ("\n\nNOTE: the run completed but the planet-preservation gate FAILED on the toy data ("
                + "; ".join(metrics.get("gate_reasons", [])) + "). An algorithm that absorbs injected "
                "planets on the toy will almost certainly do so on the real data.")
    return report[:14000] + note, ok

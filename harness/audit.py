"""Static audit of submitted algorithm code.

A guard-rail against accidental or drifting reward hacking, not a security
boundary: it rejects code that could reach outside the data API (files, network,
processes, introspection). For hard isolation also set RVSWARM_SANDBOX_CMD.

    python -m harness.audit path/to/algorithm_dir
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
    "vars", "globals", "locals", "breakpoint",
}
ALLOWED_DUNDER_ATTRS = {"__init__", "__call__", "__len__", "__getitem__", "__iter__", "__next__",
                        "__enter__", "__exit__", "__post_init__", "__version__"}
FORBIDDEN_ATTRS = {"attrgetter", "methodcaller", "system", "popen", "fromfile", "tofile", "loadtxt", "genfromtxt",
                   "savetxt", "read_csv", "read_pickle", "to_pickle", "environ", "getenv"}
BAD_STRING_BITS = ("../", "harness_secret", "/etc/", "/proc/", "/home/", "/tmp/", "~/")


def _root(name: str) -> str:
    return name.split(".")[0]


def audit_source(src: str, filename: str, allowed: set[str], forbidden_sub: list[str],
                 local_modules: set[str]) -> list[str]:
    problems: list[str] = []
    try:
        tree = ast.parse(src, filename)
    except SyntaxError as e:
        return [f"{filename}:{e.lineno}: syntax error: {e.msg}"]

    def bad(node, msg):
        problems.append(f"{filename}:{getattr(node, 'lineno', '?')}: {msg}")

    def check_import(node, mod: str):
        if any(mod == f or mod.startswith(f + ".") for f in forbidden_sub):
            bad(node, f"import of '{mod}' is not allowed")
        elif _root(mod) not in allowed and _root(mod) not in local_modules:
            bad(node, f"import of '{mod}' is not allowed (not on the package allow-list)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                check_import(node, a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                check_import(node, node.module or "")
                for a in node.names:
                    check_import(node, f"{node.module}.{a.name}") if any(
                        f"{node.module}.{a.name}" == f for f in forbidden_sub) else None
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            bad(node, f"use of builtin '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            at = node.attr
            if at.startswith("__") and at.endswith("__") and at not in ALLOWED_DUNDER_ATTRS:
                bad(node, f"dunder attribute '{at}' is not allowed")
            elif at.startswith("_") and not at.startswith("__"):
                if not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")):
                    bad(node, f"private attribute access '.{at}' on a foreign object is not allowed")
            elif at in FORBIDDEN_ATTRS:
                bad(node, f"attribute '.{at}' is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            pathlike = (s.startswith("/") and len(s) > 1 and " " not in s and s[1:2].isalnum())
            if pathlike or any(b in s.lower() for b in BAD_STRING_BITS):
                bad(node, f"path-like string literal {s[:40]!r} is not allowed (use ctx.scratch_file)")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            pass
    return problems


def audit_dir(alg_dir, cfg_audit: dict) -> list[str]:
    alg_dir = Path(alg_dir)
    files = sorted(alg_dir.glob("*.py"))
    if not (alg_dir / "algorithm.py").exists():
        return ["algorithm.py is missing"]
    local = {p.stem for p in files}
    problems: list[str] = []
    for p in files:
        problems += audit_source(p.read_text(), p.name, set(cfg_audit["allowed_imports"]),
                                 list(cfg_audit["forbidden_submodules"]), local)
    extra = [p.name for p in alg_dir.iterdir()
             if p.suffix in (".so", ".pyc", ".pyd", ".sh", ".pkl", ".pt", ".npy", ".npz")]
    if extra:
        problems.append(f"binary / executable files are not allowed in an algorithm dir: {extra}")
    return problems


if __name__ == "__main__":
    from . import config
    probs = audit_dir(sys.argv[1], config.load()["audit"])
    print("\n".join(probs) if probs else "AUDIT OK")
    sys.exit(1 if probs else 0)

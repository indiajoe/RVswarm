"""Static audit of submitted algorithm code.

A guard-rail against accidental or drifting reward hacking, not a security
boundary: it rejects code that could reach outside the data API (files, network,
processes, introspection). For hard isolation also set RVSWARM_SANDBOX_CMD.

    python -m harness.audit path/to/algorithm_dir
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Source files allowed in <alg>/native/, by language (config: audit.native.languages).
NATIVE_EXT = {"c": (".c", ".h"), "cpp": (".cpp", ".cc", ".cxx", ".hpp", ".hh"),
              "fortran": (".f90", ".f95", ".f03", ".f08", ".f", ".for"), "julia": (".jl",)}
# Coarse text screen of native sources. Compiled code cannot be audited statically; this only
# catches the obvious. The container (RVSWARM_SANDBOX_CMD) is what actually confines it.
_C_BAD = re.compile(r"\b(fopen|freopen|fdopen|open|openat|creat|popen|system|exec[lv]p?e?|fork|vfork|socket|"
                    r"connect|dlopen|getenv|opendir|readlink|remove|rename|unlink|mmap)\s*\(|"
                    r"#\s*include\s*[<\"](sys/socket|netinet/|arpa/|netdb|dirent|dlfcn|fstream|filesystem|curl/)")
_F_BAD = re.compile(r"(?i)\b(open|inquire)\s*\(|execute_command_line|call\s+system|get_environment_variable|\bgetenv\b")
_J_BAD = re.compile(r"\b(open|run|download|read|readlines|readdir|write|rm|mv|cp|mkdir|cd|pwd|ccall|eval|"
                    r"include_string|pipeline|Cmd|connect|listen)\s*\(|@ccall|@cmd|\bENV\b|\bLibc\b|\bSockets\b|"
                    r"\bDownloads\b|\bPkg\b|`")


def _lang_of(path: Path) -> str | None:
    for lang, exts in NATIVE_EXT.items():
        if path.suffix.lower() in exts:
            return lang
    return None


def audit_native(native_dir: Path, cfg_native: dict, label: str = "native/") -> list[str]:
    problems = []
    langs = set(cfg_native.get("languages", []))
    total = 0
    for p in sorted(native_dir.iterdir()):
        if p.is_dir():
            problems.append(f"{label}{p.name}: sub-directories are not allowed in {label}")
            continue
        lang = _lang_of(p)
        if lang is None:
            problems.append(f"{label}{p.name}: only source files are allowed "
                            f"({', '.join(e for l in sorted(langs) for e in NATIVE_EXT[l])})")
            continue
        if lang not in langs:
            problems.append(f"{label}{p.name}: language '{lang}' is not enabled on this cluster "
                            f"(enabled: {sorted(langs)}); request it in workspace/environment_requests/")
            continue
        src = p.read_text(errors="replace")
        total += len(src)
        rx = {"c": _C_BAD, "cpp": _C_BAD, "fortran": _F_BAD, "julia": _J_BAD}[lang]
        for n, line in enumerate(src.splitlines(), 1):
            m = rx.search(line)
            if m:
                problems.append(f"{label}{p.name}:{n}: '{m.group(0).strip()}' is not allowed "
                                "(no file, process, network or environment access from native code)")
    if total > int(cfg_native.get("max_source_bytes", 2_000_000)):
        problems.append(f"{label}: sources exceed the size limit")
    return problems

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


def _audit_code(d: Path, cfg_audit: dict, label: str, subdirs: tuple) -> list[str]:
    """One code directory: its *.py files (which may import each other), native/, nothing else."""
    files = sorted(d.glob("*.py"))
    local = {p.stem for p in files}
    problems: list[str] = []
    allowed = set(cfg_audit["allowed_imports"])
    forbidden = list(cfg_audit["forbidden_submodules"])
    nat_cfg = cfg_audit.get("native") or {}
    native = d / "native"
    if native.is_dir():
        if not nat_cfg.get("languages"):
            problems.append(f"{label}native/: native code is not enabled in config/harness.yaml (audit.native.languages)")
        else:
            problems += audit_native(native, nat_cfg, f"{label}native/")
            allowed |= set(nat_cfg.get("extra_imports", []))      # ctypes / juliacall, to call the native code
            forbidden = [f for f in forbidden if f != "numpy.ctypeslib"]
    for p in files:
        problems += audit_source(p.read_text(), f"{label}{p.name}", allowed, forbidden, local)
    for p in d.iterdir():
        if p.is_dir() and p.name not in subdirs + ("native", "__pycache__"):
            problems.append(f"sub-directory '{label}{p.name}/' is not allowed")
    extra = [p.name for p in d.iterdir()
             if p.suffix in (".so", ".pyc", ".pyd", ".sh", ".pkl", ".pt", ".npy", ".npz", ".o", ".a", ".dylib")]
    if extra:
        problems.append(f"binary / executable files are not allowed in an algorithm dir: {[label + x for x in extra]}")
    return problems


def audit_dir(alg_dir, cfg_audit: dict) -> list[str]:
    """algorithm.py and its helpers; optionally native/ and a preliminary reduction step in reduction/
    (its own code, which algorithm.py cannot import: the two only share the products)."""
    alg_dir = Path(alg_dir)
    if not (alg_dir / "algorithm.py").exists():
        return ["algorithm.py is missing"]
    problems = _audit_code(alg_dir, cfg_audit, "", ("reduction",))
    red = alg_dir / "reduction"
    if red.is_dir():
        entry = red / "reduce.py"
        if not entry.exists():
            problems.append("reduction/reduce.py is missing: a reduction/ directory must define reduce(ctx) there")
        else:
            try:
                tree = ast.parse(entry.read_text(), "reduction/reduce.py")
                if not any(isinstance(n, ast.FunctionDef) and n.name == "reduce" for n in tree.body):
                    problems.append("reduction/reduce.py must define a top-level function reduce(ctx)")
            except SyntaxError:
                pass                                           # reported by the source audit below
        problems += _audit_code(red, cfg_audit, "reduction/", ())
    return problems


if __name__ == "__main__":
    from . import config
    probs = audit_dir(sys.argv[1], config.load()["audit"])
    print("\n".join(probs) if probs else "AUDIT OK")
    sys.exit(1 if probs else 0)

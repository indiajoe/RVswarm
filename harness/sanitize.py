"""What an algorithm's own output may publish in reports on real data (RUNBOOK section 9).

Algorithm code decides what it prints, the text of its exceptions, its ctx.metric values and its
ctx.diagnostic arrays. Published unfiltered, these could show the agents per-epoch values of the
real data, and they could then tune methods to those values. So in screen and full reports:

* text (log tail and error messages): every digit is shown as '#', except the elapsed-time stamp
  of ctx.log lines and the line numbers of traceback locations; only the last lines are shown;
* ctx.metric: at most `max_metrics` values, rounded to `significant_digits`;
* ctx.diagnostic: shape and summary statistics, rounded; arrays with at most `max_array_values`
  values are shown in full; at most `max_arrays` arrays.

Sealed reports show only the masked error text. The complete logs and arrays are kept for the
humans in state/algorithm_outputs/. The local toy test (synthetic data or a few hand-picked
frames) shows everything.
"""
from __future__ import annotations

import re

import numpy as np

DEFAULTS = {"max_metrics": 20, "max_arrays": 8, "max_array_values": 16, "max_log_lines": 40,
            "max_line_chars": 200, "significant_digits": 3}
_STAMP = re.compile(r"^(\[alg \+\s*\d+s\])(.*)$")
_TB = re.compile(r'^(\s*)File "([^"]*)", line (\d+)(.*)$')
_DIGITS = re.compile(r"\d+")


def limits(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("algorithm_outputs") or {})}


def _mask(s: str) -> str:
    return _DIGITS.sub("#", s)


def mask_line(line: str, max_chars: int = 200) -> str:
    line = line.rstrip()[:max_chars]
    m = _TB.match(line)
    if m:                                    # traceback location: keep the line number, shorten the path
        name = m.group(2).replace("\\", "/")
        short = "/".join(name.split("/")[-2:]) if "/reduction/" in name else name.split("/")[-1]
        return f'{m.group(1)}File "{_mask(short)}", line {m.group(3)}{_mask(m.group(4))}'
    m = _STAMP.match(line)
    if m:                                    # ctx.log line: keep the harness's time stamp
        return m.group(1) + _mask(m.group(2))
    return _mask(line)


def mask_text(text: str, lim: dict | None = None) -> str:
    lim = {**DEFAULTS, **(lim or {})}
    lines = str(text or "").splitlines()
    n = int(lim["max_log_lines"])
    out = [mask_line(l, int(lim["max_line_chars"])) for l in lines[-n:]]
    if len(lines) > n:
        out.insert(0, f"... ({len(lines) - n} earlier lines not shown)")
    return "\n".join(out)


def sig(x, digits: int = 3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or x == 0.0:
        return x
    return float(f"{x:.{digits}g}")


def public_metrics(metrics: dict, lim: dict | None = None) -> tuple[list[list], int]:
    """-> ([[name, value], ...], number not shown)."""
    lim = {**DEFAULTS, **(lim or {})}
    items = list(metrics.items())
    n = int(lim["max_metrics"])
    rows = [[_mask(str(k))[:60], sig(v, int(lim["significant_digits"]))] for k, v in items[:n]]
    return rows, max(0, len(items) - n)


def public_diagnostics(diags: dict, lim: dict | None = None) -> tuple[list[dict], int]:
    """-> (one summary per array, number of arrays not shown)."""
    lim = {**DEFAULTS, **(lim or {})}
    sd = int(lim["significant_digits"])
    out = []
    items = list(diags.items())
    for name, a in items[: int(lim["max_arrays"])]:
        a = np.asarray(a, dtype=float)
        flat = a.reshape(-1)
        fin = flat[np.isfinite(flat)]
        e = {"name": _mask(str(name))[:60], "shape": list(a.shape), "non_finite": int(flat.size - fin.size)}
        if fin.size:
            e.update(min=sig(fin.min(), sd), median=sig(np.median(fin), sd), max=sig(fin.max(), sd),
                     mean=sig(fin.mean(), sd), std=sig(fin.std(), sd))
        if flat.size <= int(lim["max_array_values"]):
            e["values"] = [sig(x, sd) for x in flat]
        out.append(e)
    return out, max(0, len(items) - int(lim["max_arrays"]))


def _cell(x) -> str:
    return "-" if x is None else (f"{x:.6g}" if isinstance(x, float) else str(x))


def to_markdown(rows: list[list], n_more_metrics: int, diags: list[dict], n_more_diags: int,
                log_tail: str, lim: dict | None = None) -> str:
    lim = {**DEFAULTS, **(lim or {})}
    L = ["## Reported by the algorithm (un-injected run)", "",
         "_On real data a report shows only reduced forms of what the algorithm prints and records: every "
         "digit in its log and error messages is shown as `#` (time stamps of `ctx.log` lines and traceback "
         f"line numbers excepted), `ctx.metric` values are rounded to {lim['significant_digits']} significant "
         f"digits (at most {lim['max_metrics']}), and `ctx.diagnostic` arrays are summarised (arrays of at most "
         f"{lim['max_array_values']} values are shown in full). Put the numbers you need into `ctx.metric` and "
         "small `ctx.diagnostic` arrays; the local toy test shows everything._", ""]
    if rows:
        L += ["### `ctx.metric`", "", "| name | value |", "|---|---|"]
        L += [f"| `{k}` | {_cell(v)} |" for k, v in rows]
        if n_more_metrics:
            L.append(f"\n{n_more_metrics} more metric(s) not shown (limit {lim['max_metrics']}).")
        L.append("")
    if diags:
        L += ["### `ctx.diagnostic`", "", "| name | shape | non-finite | min | median | max | mean | std | values |",
              "|---|---|---|---|---|---|---|---|---|"]
        for e in diags:
            vals = ", ".join(_cell(x) for x in e["values"]) if "values" in e else "(summary only)"
            L.append(f"| `{e['name']}` | {tuple(e['shape'])} | {e['non_finite']} | "
                     + " | ".join(_cell(e.get(k)) for k in ("min", "median", "max", "mean", "std")) + f" | {vals} |")
        if n_more_diags:
            L.append(f"\n{n_more_diags} more array(s) not shown (limit {lim['max_arrays']}).")
        L.append("")
    L += ["### Log tail, digits masked", "", "```", log_tail or "(empty)", "```"]
    return "\n".join(L)

#!/usr/bin/env bash
# Harness self-test on the toy Sun: honest methods pass, cheaters are caught.
# Usage: tests/selftest.sh     (takes a few minutes; needs no data and no API key)
set -u
export RVSWARM_TOY=synth        # always the synthetic Sun here, even if real toy frames are configured
source "$(dirname "$0")/../bin/_common.sh"
OUT="${TMPDIR:-/tmp}/rvswarm_selftest_$$"
# harness bookkeeping of the self-test (truth files, caches, kept algorithm outputs) stays out of state/
export RVSWARM_STATE_DIR="$OUT/state" RVSWARM_REDUCTION_CACHE="$OUT/state/reduction_cache"
fail=0
check() {  # name  expected_gate(True/False/audit)  alg_dir
  local res; res=$("$PY" -m harness.evaluate --alg-dir "$3" --stage toy --out-root "$OUT/$1" | tail -1)
  local got; got=$("$PY" -c "import json,sys; d=json.loads(sys.argv[1]); print('audit' if d['status']=='audit_failed' else d['gate_pass'])" "$res")
  if [ "$got" = "$2" ]; then echo "ok    $1 (gate=$got)"; else echo "FAIL  $1: expected $2, got $got :: $res"; fail=1; fi
}
check honest_pipeline      True  baselines/baseline_pipeline
check indicator_decorr     True  baselines/baseline_indicator_decorr
check ccf_acf_pca          True  baselines/baseline_ccf_acf_pca
check canary_constant      False baselines/canary_constant
check canary_time_smoother False baselines/canary_time_smoother
check canary_drop_half     False baselines/canary_drop_half
check forbidden_code       audit tests/algs/bad_audit
check native_c_code        True  tests/algs/native_c
check forbidden_native     audit tests/algs/bad_native
# runs of one evaluation cannot pass information to each other through files the harness controls
check crossrun_cheat       False tests/algs/crossrun_cheat
# pre-check: an algorithm that does not reduce the pipeline scatter gets no injection runs at screen
res=$(RVSWARM_ALLOW_UNSANDBOXED=1 RVSWARM_TEST_BACKEND='{"kind": "synth", "seed": 3, "n_days": 900, "obs_per_day": 1, "day_keep": 0.33}' \
      "$PY" -m harness.evaluate --alg-dir tests/algs/no_gain --stage screen --out-root "$OUT/no_gain" | tail -1)
if [ -f "$OUT/no_gain/screen_01/metrics.json" ] && grep -q '"injection_skipped": true' "$OUT/no_gain/screen_01/metrics.json"; then
  echo "ok    precheck_skips_injection"; else echo "FAIL  precheck_skips_injection :: $res"; fail=1; fi
mkdir -p "$OUT"
# development-data runs (screen / full) on the synthetic Sun
dev() {  # alg_dir stage
  RVSWARM_ALLOW_UNSANDBOXED=1 RVSWARM_TEST_BACKEND='{"kind": "synth", "seed": 3, "n_days": 900, "obs_per_day": 1, "day_keep": 0.33}' \
    "$PY" -m harness.evaluate --alg-dir "$1" --stage "$2" --out-root "$OUT/$(basename "$1")" > /dev/null 2>&1
}
# re-use: reduce_b has the same reduction step as reduce_a, so its un-injected run takes the cached
# products; the full stage of reduce_a takes the un-injected run of its screen stage
dev tests/algs/reduce_a screen; dev tests/algs/reduce_b screen; dev tests/algs/reduce_a full
if "$PY" - "$OUT" <<'PYCHECK'
import json, sys
o = sys.argv[1]
a = json.load(open(f"{o}/reduce_a/screen_01/metrics.json"))
b = json.load(open(f"{o}/reduce_b/screen_01/metrics.json"))
f = json.load(open(f"{o}/reduce_a/full_01/metrics.json"))
ok = (a["gate_pass"] and a["reduction"]["runs"][0].get("stored") == "stored"
      and b["reduction"]["runs"][0].get("cached") is True and b["reduction"]["runs"][1].get("cached") is False
      and f.get("uninjected_run_reused_from") == "screen_01" and len(f["runtimes_s"]) == 3 and f["gate_pass"])
sys.exit(0 if ok else 1)
PYCHECK
then echo "ok    reduction_cache_and_run_reuse"; else echo "FAIL  reduction_cache_and_run_reuse (see $OUT/reduce_*)"; fail=1; fi
# what an algorithm prints and records is published only in reduced form on real data
dev tests/algs/leaky screen
if "$PY" - "$OUT" <<'PYCHECK'
import json, re, sys
from pathlib import Path
o = Path(sys.argv[1])
d = o / "leaky" / "screen_01"
rep = (d / "report.md").read_text()
m = json.loads((d / "metrics.json").read_text())
leak_lines = [l for l in rep.splitlines() if "LEAK" in l]
ok = (leak_lines and not any(re.search(r"\d", l.split("]", 1)[-1] if l.startswith("[alg") else l) for l in leak_lines)
      and len(m["algorithm_outputs"]["metrics"]) == 20 and m["algorithm_outputs"]["more_metrics"] == 10
      and "values" not in m["algorithm_outputs"]["diagnostics"][0]
      and not (d / "diagnostics.npz").exists() and not (d / "run_0.log").exists()
      and (o / "state" / "algorithm_outputs" / "leaky" / "screen_01" / "run_0.log").exists())
sys.exit(0 if ok else 1)
PYCHECK
then echo "ok    algorithm_output_filter"; else echo "FAIL  algorithm_output_filter (see $OUT/leaky)"; fail=1; fi
# residual diagnostics: every file in harness/residual_diagnostics/ runs and respects the limits
if "$PY" -m harness.residual_diagnostics check > "$OUT/residual_diagnostics.txt" 2>&1; then
  echo "ok    residual_diagnostics"; else echo "FAIL  residual_diagnostics (see: bin/py.sh -m harness.residual_diagnostics check)"; fail=1; fi
rm -rf "$OUT"
[ $fail = 0 ] && echo "ALL SELF-TESTS PASSED" || { echo "SELF-TEST FAILURES"; exit 1; }

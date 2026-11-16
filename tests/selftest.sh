#!/usr/bin/env bash
# Harness self-test on the toy Sun: honest methods pass, cheaters are caught.
# Usage: tests/selftest.sh     (takes a few minutes; needs no data and no API key)
set -u
export RVSWARM_TOY=synth        # always the synthetic Sun here, even if real toy frames are configured
source "$(dirname "$0")/../bin/_common.sh"
OUT="${TMPDIR:-/tmp}/rvswarm_selftest_$$"
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
# pre-check: an algorithm that does not reduce the pipeline scatter gets no injection runs at screen
res=$(RVSWARM_ALLOW_UNSANDBOXED=1 RVSWARM_TEST_BACKEND='{"kind": "synth", "seed": 3, "n_days": 900, "obs_per_day": 1, "day_keep": 0.33}' \
      "$PY" -m harness.evaluate --alg-dir tests/algs/no_gain --stage screen --out-root "$OUT/no_gain" | tail -1)
if [ -f "$OUT/no_gain/screen_01/metrics.json" ] && grep -q '"injection_skipped": true' "$OUT/no_gain/screen_01/metrics.json"; then
  echo "ok    precheck_skips_injection"; else echo "FAIL  precheck_skips_injection :: $res"; fail=1; fi
rm -rf "$OUT"
[ $fail = 0 ] && echo "ALL SELF-TESTS PASSED" || { echo "SELF-TEST FAILURES"; exit 1; }

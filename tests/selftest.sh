#!/usr/bin/env bash
# Harness self-test on the toy Sun: honest methods pass, cheaters are caught.
# Usage: tests/selftest.sh     (takes a few minutes; needs no data and no API key)
set -u
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
rm -rf "$OUT"
[ $fail = 0 ] && echo "ALL SELF-TESTS PASSED" || { echo "SELF-TEST FAILURES"; exit 1; }

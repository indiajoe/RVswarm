#!/usr/bin/env bash
# One evaluation: algorithm $1 at stage $2 (ticket $3).  Submitted by bin/submit_eval.sh.
set -u
ROOT="${RVSWARM_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$ROOT/bin/_common.sh"
export RVSWARM_WORKDIR="${RVSWARM_SCRATCH:-$ROOT/state/scratch}"
mkdir -p "$RVSWARM_WORKDIR"
echo "host=$(hostname) alg=$1 stage=$2 start=$(date)"
"$PY" -m harness.evaluate --alg "$1" --stage "$2" --ticket "$3"
echo "exit=$? end=$(date)"

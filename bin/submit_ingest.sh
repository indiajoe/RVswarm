#!/usr/bin/env bash
# Build the compact store from the full NEID L2 data set (run once; re-runnable).
#   bin/submit_ingest.sh [N_CHUNKS]     default 200 array tasks
#   bin/submit_ingest.sh --local        do everything in this shell (small data sets)
set -eu
source "$(dirname "$0")/_common.sh"
if [ "${1:-}" = "--local" ]; then
  "$PY" -m harness.ingest plan --n-chunks 1
  "$PY" -m harness.ingest shard --chunk 0
  "$PY" -m harness.ingest finalize
  exit 0
fi
N="${1:-200}"
"$PY" -m harness.ingest plan --n-chunks "$N"
STORE="$("$PY" -c 'from harness import config; print(config.load()["data"]["store_dir"])')"
N="$(cat "$STORE/plan/n_chunks")"
COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --output="state/slurm_logs/ingest_%A_%a.log")
[ -n "${RVSWARM_SBATCH_ACCOUNT}" ] && COMMON+=(--account="${RVSWARM_SBATCH_ACCOUNT}")
[ -n "${RVSWARM_SBATCH_PARTITION}" ] && COMMON+=(--partition="${RVSWARM_SBATCH_PARTITION}")
JID=$(sbatch --parsable "${COMMON[@]}" --job-name=rv_ingest --time=08:00:00 --array="0-$((N-1))%64" \
      --export=ALL,RVSWARM_ROOT="$ROOT" --wrap \
      "source '$ROOT/bin/_common.sh' && \"\$PY\" -m harness.ingest shard --chunk \$SLURM_ARRAY_TASK_ID")
echo "ingest array job $JID"
sbatch "${COMMON[@]}" --job-name=rv_ingest_fin --time=02:00:00 --mem=64G --dependency="afterok:${JID%%;*}" \
      --export=ALL,RVSWARM_ROOT="$ROOT" --wrap \
      "source '$ROOT/bin/_common.sh' && \"\$PY\" -m harness.ingest finalize"

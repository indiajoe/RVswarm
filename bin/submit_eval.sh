#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# The ONLY path by which an algorithm reaches the hidden data.  Deliberately dumb:
#   1. apply the fixed promotion rules (screen -> full -> sealed)
#   2. turn every ticket in state/eval_queue/ into one SLURM job
# Run it by hand or from cron on a node that has `sbatch`.
#   bin/submit_eval.sh            submit with sbatch
#   bin/submit_eval.sh --local    run the jobs right here, one after another (testing)
# -----------------------------------------------------------------------------
set -u
source "$(dirname "$0")/_common.sh"
MODE="${1:-slurm}"

if [ -z "${RVSWARM_SANDBOX_CMD}" ] && [ "${RVSWARM_ALLOW_UNSANDBOXED}" != "1" ]; then
  echo "Refusing to submit: set RVSWARM_SANDBOX_CMD in config/cluster.env (RUNBOOK section 10)," >&2
  echo "or RVSWARM_ALLOW_UNSANDBOXED=1 to knowingly run algorithm code un-contained." >&2
  exit 1
fi

"$PY" -m harness.promote advance

shopt -s nullglob
for T in state/eval_queue/*.json; do
  read -r ALG STAGE CPUS MEM GPUS MINUTES < <("$PY" -m harness.promote fields "$T")
  NAME="$(basename "$T")"
  mv "$T" "state/eval_submitted/$NAME"
  if [ "$MODE" = "--local" ]; then
    echo "running locally: $ALG / $STAGE"
    bash bin/eval_job.sbatch "$ALG" "$STAGE" "state/eval_submitted/$NAME"
    continue
  fi
  ARGS=(--job-name="rv${RVSWARM_NAME:+_$RVSWARM_NAME}_${STAGE}_${ALG}" --cpus-per-task="$CPUS" --mem="${MEM}G"
        --time="$MINUTES" --nodes=1 --ntasks=1
        --output="state/slurm_logs/${ALG}__${STAGE}_%j.log")
  [ -n "${RVSWARM_SBATCH_ACCOUNT}" ] && ARGS+=(--account="${RVSWARM_SBATCH_ACCOUNT}")
  if [ "$GPUS" -gt 0 ]; then
    [ -n "${RVSWARM_SBATCH_GPU_PARTITION}" ] && ARGS+=(--partition="${RVSWARM_SBATCH_GPU_PARTITION}")
    ARGS+=(${RVSWARM_SBATCH_GPU_FLAG})
  else
    [ -n "${RVSWARM_SBATCH_PARTITION}" ] && ARGS+=(--partition="${RVSWARM_SBATCH_PARTITION}")
  fi
  if sbatch "${ARGS[@]}" ${RVSWARM_SBATCH_EXTRA} bin/eval_job.sbatch "$ALG" "$STAGE" "state/eval_submitted/$NAME"; then
    echo "submitted: $ALG / $STAGE"
  else
    echo "sbatch failed for $ALG / $STAGE; ticket returned to the queue" >&2
    mv "state/eval_submitted/$NAME" "$T"
  fi
done

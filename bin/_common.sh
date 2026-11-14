# Shared setup for all bin/ scripts (sourced, not executed).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RVSWARM_ROOT="$ROOT"
cd "$ROOT"
source "$ROOT/config/cluster.env"
PY="${RVSWARM_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/state/eval_queue" "$ROOT/state/eval_submitted" "$ROOT/state/eval_done" "$ROOT/state/slurm_logs"

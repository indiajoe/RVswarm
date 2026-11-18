#!/usr/bin/env bash
# Create one fully independent swarm: its own copy of the code, workspace, state, harness
# secret (so its own hidden planets) and data set. Swarms never read each other's files.
#
#   bin/new_swarm.sh NAME DATA_DIR [TOY_DATA_DIR] [DEST]
#
#   NAME          short label, letters/digits only (A, B, C ...): used in SLURM job names
#   DATA_DIR      the curated L2 spectra of THIS swarm, as seen from the compute nodes
#   TOY_DATA_DIR  about 5 hand-picked frames on the orchestrator host for the implementers'
#                 self-test ("" = synthetic Sun; slow and costly, see RUNBOOK 4.0)
#   DEST          where to put it (default: next to this repository, rvswarm_NAME)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?usage: bin/new_swarm.sh NAME DATA_DIR [TOY_DATA_DIR] [DEST]}"
DATA="${2:?give the directory with the curated spectra of this swarm}"
TOY="${3:-}"
DEST="${4:-$(dirname "$HERE")/rvswarm_$NAME}"
[[ "$NAME" =~ ^[A-Za-z0-9]+$ ]] || { echo "NAME must be letters and digits only"; exit 1; }
[ -e "$DEST" ] && { echo "$DEST already exists"; exit 1; }
[ -d "$DATA" ] || echo "WARNING: $DATA is not visible from this host (fine if it exists on the compute nodes)"
if [ -n "$(git -C "$HERE" status --porcelain --untracked-files=no)" ]; then
  echo "WARNING: uncommitted changes in $HERE are NOT copied (git clone takes committed files only)"
fi
git clone --quiet "$HERE" "$DEST"
{
  echo "# Settings of swarm $NAME (not tracked by git). Sourced first by config/cluster.env."
  echo "export RVSWARM_NAME=\"$NAME\""
  echo "export NEID_DATA_DIR=\"$DATA\""
  echo "export RVSWARM_TOY_DATA_DIR=\"$TOY\""
} > "$DEST/config/local.env"
cat > "$DEST/config/local_swarm.yaml" <<EOT
# Settings of swarm $NAME that differ from config/swarm.yaml. Not tracked by git.
# Keys here replace those in swarm.yaml; a list replaces the whole list. RUNBOOK section 3.6.
# Examples (remove the leading '# ' to use):
# scope:
#   note: "This swarm develops fast methods built on the CCFs, per-order RVs, indices and telemetry."
#   allowed_uses: [ccf, order_rv, indices, telemetry, time]
# cycle:
#   wait_for_results: screen
EOT
if [ -d "$HERE/.venv" ] && [ ! -e "$DEST/.venv" ] && [ ! -L "$DEST/.venv" ]; then
  ln -s "$HERE/.venv" "$DEST/.venv"                                      # share the Python environment
fi
echo "swarm $NAME created in $DEST"
echo "  data: $DATA"
echo "next, inside $DEST: follow RUNBOOK section 3.5"

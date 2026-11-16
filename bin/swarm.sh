#!/usr/bin/env bash
# Run the orchestrator of THIS swarm with its environment (config/cluster.env and
# config/local.env) loaded:   bin/swarm.sh init | run | loop | status | costs  [--only PHASE]
source "$(dirname "$0")/_common.sh"
exec "$PY" -m swarm.cycle "$@"

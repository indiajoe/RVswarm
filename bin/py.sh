#!/usr/bin/env bash
# Run the project Python with the environment of THIS swarm loaded (config/cluster.env and
# config/local.env):   bin/py.sh -m harness.ingest toy      bin/py.sh -m harness.promote leaderboard
source "$(dirname "$0")/_common.sh"
exec "$PY" "$@"

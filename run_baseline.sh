#!/bin/bash
# Start baseline (Word2Vec + MLP) training under nohup (exit immediately; training keeps running).
#
# Usage:
#   bash run_baseline.sh
#   SAMPLE=/path/to/sample.json bash run_baseline.sh
#
# Env: SAMPLE (default data/sample.json).
# Logs: logs/baseline.log — tail -f logs/baseline.log

set -e
MODEL=baseline

cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

: "${SAMPLE:=data/sample.json}"
if [[ ! -f "$SAMPLE" ]]; then
  echo "Missing data file: $SAMPLE (run: bash create_sample.sh)"
  exit 1
fi

mkdir -p logs
rm -rf "output/${MODEL}"

nohup python -u baseline.py --config "config/${MODEL}.yaml" --data "$SAMPLE" \
  >"logs/${MODEL}.log" 2>&1 &
echo $! >"logs/${MODEL}.pid"
echo "Started ${MODEL} PID $(cat "logs/${MODEL}.pid")"
echo "  log: logs/${MODEL}.log"
echo "  tail -f logs/${MODEL}.log"

#!/bin/bash
# Start SciBERT training under nohup (exit immediately; training keeps running).
#
# Usage:
#   bash run_scibert.sh
#   SAMPLE=/path/to/sample.json FAST=1 bash run_scibert.sh
#
# Env: SAMPLE (default data/sample.json), FAST=1 → config/scibert_fast.yaml,
#      NUM_WORKERS (DataLoader; default 4, use 0 on NFS hangs).
# Logs: logs/scibert.log — tail -f logs/scibert.log

set -e
MODEL=scibert

cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

: "${SAMPLE:=data/sample.json}"
: "${FAST:=0}"
if [[ ! -f "$SAMPLE" ]]; then
  echo "Missing data file: $SAMPLE (run: bash create_sample.sh)"
  exit 1
fi

if [[ "$FAST" == "1" ]]; then
  CFG="config/${MODEL}_fast.yaml"
  OUTDIR="output/${MODEL}_fast"
  echo "=== FAST=1 → $CFG ==="
else
  CFG="config/${MODEL}.yaml"
  OUTDIR="output/${MODEL}"
fi

mkdir -p logs
rm -rf "$OUTDIR"

nohup python -u train.py --config "$CFG" --data "$SAMPLE" >"logs/${MODEL}.log" 2>&1 &
echo $! >"logs/${MODEL}.pid"
echo "Started ${MODEL} PID $(cat "logs/${MODEL}.pid")"
echo "  log: logs/${MODEL}.log"
echo "  tail -f logs/${MODEL}.log"

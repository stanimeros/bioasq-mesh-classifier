#!/bin/bash
# Start PubMedBERT training under nohup (exit immediately; training keeps running).
#
# Usage:
#   bash run_pubmedbert.sh
#   SAMPLE=/path/to/sample.json bash run_pubmedbert.sh
#
# Env: SAMPLE (default data/sample.json),
#      NUM_WORKERS (DataLoader; default 4, use 0 on NFS hangs).
# Logs: logs/pubmedbert.log — tail -f logs/pubmedbert.log

set -e
MODEL=pubmedbert

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

nohup python -u train.py --config "config/${MODEL}.yaml" --data "$SAMPLE" \
  >"logs/${MODEL}.log" 2>&1 &
echo $! >"logs/${MODEL}.pid"
echo "Started ${MODEL} PID $(cat "logs/${MODEL}.pid")"
echo "  log: logs/${MODEL}.log"
echo "  tail -f logs/${MODEL}.log"

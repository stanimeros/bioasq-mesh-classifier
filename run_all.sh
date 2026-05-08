#!/bin/bash
# Run all experiments. Execute this on the AIDA server.
# Usage: bash run_all.sh [data_path]
# Tail logs: tail -f logs/baseline.log logs/biobert.log logs/scibert.log

set -e

source .venv/bin/activate

DATA="${1:-data/allMeSH_2022.json}"
SAMPLE="${SAMPLE:-data/sample.json}"
MAX_ARTICLES="${MAX_ARTICLES:-800000}"
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "=== Sampling (max_articles=$MAX_ARTICLES) ==="
python -u sample.py --data "$DATA" --out "$SAMPLE" --config config/biobert.yaml --max_articles "$MAX_ARTICLES"

echo "=== Launching 3 training jobs in background ==="
nohup python -u baseline.py --config config/baseline.yaml --data "$SAMPLE" > logs/baseline.log 2>&1 &
echo "  baseline PID $!"
nohup python -u train.py    --config config/biobert.yaml  --data "$SAMPLE" > logs/biobert.log  2>&1 &
echo "  biobert  PID $!"
nohup python -u train.py    --config config/scibert.yaml  --data "$SAMPLE" > logs/scibert.log  2>&1 &
echo "  scibert  PID $!"

echo "=== Jobs launched. Tail with: tail -f logs/baseline.log logs/biobert.log logs/scibert.log ==="

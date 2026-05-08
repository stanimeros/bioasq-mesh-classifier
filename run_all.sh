#!/bin/bash
# Run all experiments. Execute this on the AIDA server.
# Usage: bash run_all.sh [data_path]
# Detached: chmod +x run_all.sh && nohup ./run_all.sh >> run_all.log 2>&1 & disown
# Tail logs: tail -f run_all.log logs/baseline.log logs/biobert.log logs/scibert.log

set -e

source .venv/bin/activate

DATA="${1:-data/allMeSH_2022.json}"
SAMPLE="data/sample.json"
SMOKE="data/smoke.json"
MAX_ARTICLES="${MAX_ARTICLES:-800000}"
export PYTHONUNBUFFERED=1

mkdir -p logs

# ── Smoke test (1 000 articles) ──────────────────────────────────────────────
echo "=== [1/3] Smoke test (1 000 articles) ==="
python -u sample.py --data "$DATA" --out "$SMOKE" --config config/biobert.yaml --max_articles 1000

nohup python -u baseline.py --config config/baseline.yaml --data "$SMOKE" > logs/smoke_baseline.log 2>&1 &
PID1=$!
nohup python -u train.py    --config config/biobert.yaml  --data "$SMOKE" > logs/smoke_biobert.log  2>&1 &
PID2=$!
nohup python -u train.py    --config config/scibert.yaml  --data "$SMOKE" > logs/smoke_scibert.log  2>&1 &
PID3=$!

echo "  Waiting for smoke jobs ($PID1 $PID2 $PID3) ..."
wait $PID1 || { echo "baseline smoke FAILED"; exit 1; }
wait $PID2 || { echo "biobert smoke FAILED";  exit 1; }
wait $PID3 || { echo "scibert smoke FAILED";  exit 1; }
echo "--- Smoke test passed ---"

# ── Clean up smoke artefacts ─────────────────────────────────────────────────
echo "=== [2/3] Cleaning up smoke artefacts ==="
rm -f "$SMOKE"
rm -rf output/baseline output/biobert output/scibert

# ── Real run (1/20 sample) ───────────────────────────────────────────────────
echo "=== [3/3] Real run (max_articles=$MAX_ARTICLES) ==="
python -u sample.py --data "$DATA" --out "$SAMPLE" --config config/biobert.yaml --max_articles "$MAX_ARTICLES"

nohup python -u baseline.py --config config/baseline.yaml --data "$SAMPLE" > logs/baseline.log 2>&1 &
echo "  baseline PID $!"
nohup python -u train.py    --config config/biobert.yaml  --data "$SAMPLE" > logs/biobert.log  2>&1 &
echo "  biobert  PID $!"
nohup python -u train.py    --config config/scibert.yaml  --data "$SAMPLE" > logs/scibert.log  2>&1 &
echo "  scibert  PID $!"

echo "=== Jobs launched. Tail with: tail -f logs/baseline.log logs/biobert.log logs/scibert.log ==="

#!/bin/bash
# Run training pipeline in two stages:
#   [1/2] Smoke — BioBERT + SciBERT + PubMedBERT on smoke.json; wait until all finish.
#         Transformer jobs use train.py --no_wandb (no W&B uploads).
#   [2/2] Full — remove smoke outputs, then start the same three jobs on sample.json (nohup).
#
# Baseline (Word2Vec + MLP) already evaluated — commented out below.
#
# Prerequisites: create_sample.sh has produced data/smoke.json and data/sample.json.
#
# Usage:
#   bash run_all.sh
#   bash run_all.sh path/to/smoke.json path/to/sample.json
# Env: SMOKE, SAMPLE (defaults data/smoke.json, data/sample.json)
#
# Detached: nohup ./run_all.sh >> run_all.log 2>&1 & disown
#
# Logs:
#   Smoke: logs/smoke_biobert.log logs/smoke_scibert.log logs/smoke_pubmedbert.log
#   Full:  logs/biobert.log logs/scibert.log logs/pubmedbert.log
#
# Follow:
#   tail -f run_all.log logs/smoke_biobert.log logs/smoke_scibert.log logs/smoke_pubmedbert.log logs/biobert.log logs/scibert.log logs/pubmedbert.log

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

: "${SMOKE:=data/smoke.json}"
: "${SAMPLE:=data/sample.json}"
[[ -n "${1:-}" ]] && SMOKE="$1"
[[ -n "${2:-}" ]] && SAMPLE="$2"
export PYTHONUNBUFFERED=1

mkdir -p logs

for f in "$SMOKE" "$SAMPLE"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing data file: $f (run: bash create_sample.sh)"
    exit 1
  fi
done

# ── [1/2] Smoke — wait for all ───────────────────────────────────────────────
echo "=== [1/2] Smoke (--data $SMOKE) ==="
echo "  Logs -> logs/smoke_biobert.log logs/smoke_scibert.log logs/smoke_pubmedbert.log"
echo "  tail -f run_all.log logs/smoke_biobert.log logs/smoke_scibert.log logs/smoke_pubmedbert.log"

# nohup python -u baseline.py --config config/baseline.yaml --data "$SMOKE" > logs/smoke_baseline.log 2>&1 &
nohup python -u train.py --no_wandb --config config/biobert.yaml    --data "$SMOKE" > logs/smoke_biobert.log    2>&1 &
PID1=$!
nohup python -u train.py --no_wandb --config config/scibert.yaml    --data "$SMOKE" > logs/smoke_scibert.log    2>&1 &
PID2=$!
nohup python -u train.py --no_wandb --config config/pubmedbert.yaml --data "$SMOKE" > logs/smoke_pubmedbert.log 2>&1 &
PID3=$!

echo "  PIDs: biobert=$PID1 scibert=$PID2 pubmedbert=$PID3"
echo "  Waiting for smoke jobs ..."
wait "$PID1" || { echo "biobert smoke FAILED — see logs/smoke_biobert.log"; exit 1; }
wait "$PID2" || { echo "scibert smoke FAILED — see logs/smoke_scibert.log"; exit 1; }
wait "$PID3" || { echo "pubmedbert smoke FAILED — see logs/smoke_pubmedbert.log"; exit 1; }
echo "--- Smoke passed ---"

echo "=== Clearing output/ after smoke (fresh full run) ==="
rm -rf output/biobert output/scibert output/pubmedbert

# ── [2/2] Full sample — fire and forget ─────────────────────────────────────
echo "=== [2/2] Full run (--data $SAMPLE) ==="
echo "  Logs -> logs/biobert.log logs/scibert.log logs/pubmedbert.log"
echo "  tail -f run_all.log logs/biobert.log logs/scibert.log logs/pubmedbert.log"

# nohup python -u baseline.py --config config/baseline.yaml --data "$SAMPLE" > logs/baseline.log 2>&1 &
# echo "  baseline PID $! -> logs/baseline.log"
nohup python -u train.py --config config/biobert.yaml    --data "$SAMPLE" > logs/biobert.log    2>&1 &
echo "  biobert    PID $! -> logs/biobert.log"
nohup python -u train.py --config config/scibert.yaml    --data "$SAMPLE" > logs/scibert.log    2>&1 &
echo "  scibert    PID $! -> logs/scibert.log"
nohup python -u train.py --config config/pubmedbert.yaml --data "$SAMPLE" > logs/pubmedbert.log 2>&1 &
echo "  pubmedbert PID $! -> logs/pubmedbert.log"

echo "=== Full-run jobs started ==="

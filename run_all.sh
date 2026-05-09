#!/bin/bash
# [1/2] Smoke: run baseline + BioBERT + SciBERT on smoke.json, wait for all to finish.
# [2/2] Full: wipe smoke outputs, then start the same three jobs on sample.json (nohup, no wait).
#
# Build JSONs first: bash create_sample.sh
#
# Usage:
#   bash run_all.sh
#   bash run_all.sh path/to/smoke.json path/to/sample.json
# Env (defaults if no matching positional arg): SMOKE, SAMPLE
#
# Detached: nohup ./run_all.sh >> run_all.log 2>&1 & disown
#
# Logs:
#   Smoke: logs/smoke_baseline.log logs/smoke_biobert.log logs/smoke_scibert.log
#   Full:  logs/baseline.log logs/biobert.log logs/scibert.log
#
# Follow smoke + full:
#   tail -f run_all.log logs/smoke_*.log logs/baseline.log logs/biobert.log logs/scibert.log

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

# ── [1/2] Smoke — wait for all three ─────────────────────────────────────────
echo "=== [1/2] Smoke (--data $SMOKE) ==="
echo "  Logs -> logs/smoke_baseline.log logs/smoke_biobert.log logs/smoke_scibert.log"
echo "  tail -f run_all.log logs/smoke_baseline.log logs/smoke_biobert.log logs/smoke_scibert.log"

nohup python -u baseline.py --config config/baseline.yaml --data "$SMOKE" > logs/smoke_baseline.log 2>&1 &
PID1=$!
nohup python -u train.py    --config config/biobert.yaml  --data "$SMOKE" > logs/smoke_biobert.log  2>&1 &
PID2=$!
nohup python -u train.py    --config config/scibert.yaml --data "$SMOKE" > logs/smoke_scibert.log 2>&1 &
PID3=$!

echo "  PIDs: baseline=$PID1 biobert=$PID2 scibert=$PID3"
echo "  Waiting for smoke jobs ..."
wait "$PID1" || { echo "baseline smoke FAILED — see logs/smoke_baseline.log"; exit 1; }
wait "$PID2" || { echo "biobert smoke FAILED — see logs/smoke_biobert.log";  exit 1; }
wait "$PID3" || { echo "scibert smoke FAILED — see logs/smoke_scibert.log"; exit 1; }
echo "--- Smoke passed ---"

echo "=== Clearing output/ after smoke (fresh full run) ==="
rm -rf output/baseline output/biobert output/scibert

# ── [2/2] Full sample — fire and forget ─────────────────────────────────────
echo "=== [2/2] Full run (--data $SAMPLE) ==="
echo "  Logs -> logs/baseline.log logs/biobert.log logs/scibert.log"
echo "  tail -f run_all.log logs/baseline.log logs/biobert.log logs/scibert.log"

nohup python -u baseline.py --config config/baseline.yaml --data "$SAMPLE" > logs/baseline.log 2>&1 &
echo "  baseline PID $! -> logs/baseline.log"
nohup python -u train.py    --config config/biobert.yaml  --data "$SAMPLE" > logs/biobert.log  2>&1 &
echo "  biobert  PID $! -> logs/biobert.log"
nohup python -u train.py    --config config/scibert.yaml --data "$SAMPLE" > logs/scibert.log  2>&1 &
echo "  scibert  PID $! -> logs/scibert.log"

echo "=== Full-run jobs started ==="

#!/bin/bash
# Run training pipeline in two stages:
#   [1/2] Smoke — quick sanity check on smoke.json (1 epoch, short seqs, capped articles).
#   [2/2] Full  — full training on sample.json (nohup, detached).
#
# Usage:
#   bash run_all.sh [biobert|scibert|pubmedbert]   # single model (default: all)
#   bash run_all.sh biobert path/to/smoke.json path/to/sample.json
# Env: SMOKE, SAMPLE (defaults data/smoke.json, data/sample.json)
#   MODEL — alternative to positional arg (MODEL=scibert bash run_all.sh)
#
# Detached: nohup ./run_all.sh biobert >> run_all.log 2>&1 & disown
#
# Logs:
#   Smoke: logs/smoke_<model>.log
#   Full:  logs/<model>.log
#
# Follow:
#   tail -f run_all.log logs/smoke_biobert.log logs/biobert.log

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

# ── Argument parsing ─────────────────────────────────────────────────────────
# First positional arg (if it looks like a model name) selects the model;
# remaining positionals override smoke/sample paths.
ALL_MODELS=(biobert scibert pubmedbert)

arg1="${1:-}"
if [[ "$arg1" == "biobert" || "$arg1" == "scibert" || "$arg1" == "pubmedbert" ]]; then
  MODEL="$arg1"
  shift
else
  : "${MODEL:=all}"
fi

: "${SMOKE:=data/smoke.json}"
: "${SAMPLE:=data/sample.json}"
[[ -n "${1:-}" ]] && SMOKE="$1"
[[ -n "${2:-}" ]] && SAMPLE="$2"
export PYTHONUNBUFFERED=1

if [[ "$MODEL" == "all" ]]; then
  MODELS=("${ALL_MODELS[@]}")
else
  MODELS=("$MODEL")
fi

echo "=== Models: ${MODELS[*]} ==="
mkdir -p logs

for f in "$SMOKE" "$SAMPLE"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing data file: $f (run: bash create_sample.sh)"
    exit 1
  fi
done

# ── [1/2] Smoke — run selected models sequentially ───────────────────────────
echo "=== [1/2] Smoke (--data $SMOKE) ==="

smoke_train_opts=(
  --epochs 1
  --max_length 128
  --batch_size 64
  --early_stopping_patience 0
  --max_articles 128
  --min_label_count 1
)

for m in "${MODELS[@]}"; do
  echo "  Smoke: $m -> logs/smoke_${m}.log"
  python -u train.py --no_wandb --config "config/${m}.yaml" --data "$SMOKE" "${smoke_train_opts[@]}" \
    > "logs/smoke_${m}.log" 2>&1 \
    || { echo "$m smoke FAILED — see logs/smoke_${m}.log"; exit 1; }
  echo "  $m smoke OK"
done

echo "--- Smoke passed ---"

# ── [2/2] Full run — run selected models sequentially ────────────────────────
echo "=== Clearing output/ for selected models ==="
for m in "${MODELS[@]}"; do
  rm -rf "output/${m}"
done

echo "=== [2/2] Full run (--data $SAMPLE) ==="

for m in "${MODELS[@]}"; do
  echo "  Full: $m -> logs/${m}.log"
  nohup python -u train.py --config "config/${m}.yaml" --data "$SAMPLE" \
    > "logs/${m}.log" 2>&1 &
  echo "  $m PID $! -> logs/${m}.log"
done

echo "=== Full-run jobs started ==="

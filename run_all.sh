#!/bin/bash
# Run training pipeline (foreground, no nohup — output also in logs/<model>.log):
#   Full — sequential training on sample.json.
#   (Smoke stage is commented out; re-enable when you want a quick sanity check on smoke.json.)
#
# Usage:
#   bash run_all.sh [biobert|scibert|pubmedbert]   # single model (default: all)
#   SAMPLE=/path/to/sample.json bash run_all.sh biobert
# Env: SAMPLE (default data/sample.json). SMOKE is only used if you re-enable smoke below.
#   MODEL — alternative to positional arg (MODEL=scibert bash run_all.sh)
#   NUM_WORKERS — DataLoader workers (default 4); use 0 if workers hang on NFS.
#
# Optional detach over SSH: nohup bash run_all.sh >> run_all.log 2>&1 &
# Full runs are sequential (one GPU at a time). One model: bash run_all.sh biobert
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

for f in "$SAMPLE"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing data file: $f (run: bash create_sample.sh)"
    exit 1
  fi
done

# ── [1/2] Smoke — disabled (uncomment to run before full training) ───────────
# echo "=== [1/2] Smoke (--data $SMOKE) ==="
# smoke_train_opts=(
#   --epochs 1
#   --max_length 128
#   --batch_size 64
#   --early_stopping_patience 0
#   --max_articles 128
#   --min_label_count 1
# )
# for m in "${MODELS[@]}"; do
#   echo "  Smoke: $m -> logs/smoke_${m}.log"
#   python -u train.py --no_wandb --config "config/${m}.yaml" --data "$SMOKE" "${smoke_train_opts[@]}" \
#     > "logs/smoke_${m}.log" 2>&1 \
#     || { echo "$m smoke FAILED — see logs/smoke_${m}.log"; exit 1; }
#   echo "  $m smoke OK"
# done
# echo "--- Smoke passed ---"

# ── Full run — run selected models sequentially ─────────────────────────────
echo "=== Clearing output/ for selected models ==="
for m in "${MODELS[@]}"; do
  rm -rf "output/${m}"
done

echo "=== Full run (--data $SAMPLE), one model at a time ==="

for m in "${MODELS[@]}"; do
  echo "  Full: $m (terminal + logs/${m}.log)"
  python -u train.py --config "config/${m}.yaml" --data "$SAMPLE" 2>&1 | tee "logs/${m}.log"
  echo "  $m finished -> logs/${m}.log"
done

echo "=== All full runs finished ==="

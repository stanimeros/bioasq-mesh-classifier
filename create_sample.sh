#!/bin/bash
# Build training subsets without scanning the full corpus twice for smoke:
#   1) Stream full BioASQ once -> sample.json (default 400k)
#   2) Shuffle subsample from sample.json -> smoke.json (default 1k)
#
# Usage:
#   bash create_sample.sh [path/to/allMeSH.json]
# Env overrides:
#   SAMPLE=data/sample.json SMOKE=data/smoke.json MAX_ARTICLES=400000 SMOKE_ARTICLES=1000 SEED=42

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

DATA="${1:-data/allMeSH_2022.json}"
SAMPLE="${SAMPLE:-data/sample.json}"
SMOKE="${SMOKE:-data/smoke.json}"
MAX_ARTICLES="${MAX_ARTICLES:-400000}"
SMOKE_ARTICLES="${SMOKE_ARTICLES:-1000}"
SEED="${SEED:-42}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$SAMPLE")" "$(dirname "$SMOKE")"

echo "=== [1/2] Sample ${MAX_ARTICLES} articles -> ${SAMPLE} (one pass over ${DATA}) ==="
python -u sample.py --data "$DATA" --out "$SAMPLE" --config config/biobert.yaml --max_articles "$MAX_ARTICLES"

echo "=== [2/2] Subsample ${SMOKE_ARTICLES} articles -> ${SMOKE} (from ${SAMPLE} only) ==="
python -u sample.py --data "$SAMPLE" --out "$SMOKE" --max_articles "$SMOKE_ARTICLES" --seed "$SEED"

echo "=== Done: ${SAMPLE} (${MAX_ARTICLES}) + ${SMOKE} (${SMOKE_ARTICLES}) ==="

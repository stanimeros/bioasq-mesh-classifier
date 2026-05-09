#!/bin/bash
# Build the two JSON subsets used by run_all.sh from the full BioASQ file:
#   1) sample.json — reservoir sample from allMeSH (default 400k articles)
#   2) smoke.json — random subset from sample.json only (default 1k; no second full scan)
#
# Prerequisites: full corpus on disk (see download_bioasq_mesh.sh). Requires venv (see setup.sh).
#
# Usage:
#   bash create_sample.sh [path/to/allMeSH.json]
# Env: SAMPLE, SMOKE, MAX_ARTICLES, SMOKE_ARTICLES, SEED

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

#!/bin/bash
# Build the two JSON subsets used by run_*.sh from the full BioASQ file:
#   1) sample.json — coverage-first sample (default 50k articles)
#   2) smoke.json  — random subset from sample.json only (default 1k; no second full scan)
#
# Coverage strategy: includes every article that adds a label seen fewer than
# MIN_PER_LABEL times, guaranteeing full label coverage before random fill.
#
# Prerequisites: full corpus on disk (see download_bioasq_mesh.sh). Requires venv (see setup.sh).
#
# Usage:
#   bash create_sample.sh [path/to/allMeSH.json]
# Env: SAMPLE, SMOKE, MAX_ARTICLES, SMOKE_ARTICLES, SEED, MIN_PER_LABEL

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

DATA="${1:-data/allMeSH_2022.json}"
SAMPLE="${SAMPLE:-data/sample.json}"
SMOKE="${SMOKE:-data/smoke.json}"
MAX_ARTICLES="${MAX_ARTICLES:-50000}"
SMOKE_ARTICLES="${SMOKE_ARTICLES:-1000}"
SEED="${SEED:-42}"
MIN_PER_LABEL="${MIN_PER_LABEL:-5}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$SAMPLE")" "$(dirname "$SMOKE")"

echo "=== [1/2] Sample ${MAX_ARTICLES} articles -> ${SAMPLE} (coverage pass over ${DATA}) ==="
python -u sample.py --data "$DATA" --out "$SAMPLE" \
  --max_articles "$MAX_ARTICLES" --seed "$SEED" \
  --strategy coverage --min_per_label "$MIN_PER_LABEL"

echo "=== [2/2] Subsample ${SMOKE_ARTICLES} articles -> ${SMOKE} (from ${SAMPLE} only) ==="
python -u sample.py --data "$SAMPLE" --out "$SMOKE" \
  --max_articles "$SMOKE_ARTICLES" --seed "$SEED"

echo "=== Done: ${SAMPLE} (${MAX_ARTICLES}) + ${SMOKE} (${SMOKE_ARTICLES}) ==="

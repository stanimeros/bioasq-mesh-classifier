#!/bin/bash
# Build the two JSON subsets used by run_*.sh from the full BioASQ file:
#   1) sample.json — reservoir sample from allMeSH (default 100k articles)
#   2) smoke.json — random subset from sample.json only (default 1k; no second full scan)
#
# "Meaningful" / denser supervision (optional env — passed to sample.py):
#   MIN_MESH_LABELS  — e.g. 3  (drop articles with fewer MeSH labels)
#   MIN_TEXT_CHARS   — e.g. 400 (drop very short title+abstract text)
#   OVERSAMPLE_FACTOR — e.g. 2–3 when using filters (larger reservoir before filter/trim; more RAM)
#
# Prerequisites: full corpus on disk (see download_bioasq_mesh.sh). Requires venv (see setup.sh).
#
# Usage:
#   bash create_sample.sh [path/to/allMeSH.json]
# Env: SAMPLE, SMOKE, MAX_ARTICLES, SMOKE_ARTICLES, SEED, MIN_MESH_LABELS, MIN_TEXT_CHARS, OVERSAMPLE_FACTOR

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

DATA="${1:-data/allMeSH_2022.json}"
SAMPLE="${SAMPLE:-data/sample.json}"
SMOKE="${SMOKE:-data/smoke.json}"
MAX_ARTICLES="${MAX_ARTICLES:-100000}"
SMOKE_ARTICLES="${SMOKE_ARTICLES:-1000}"
SEED="${SEED:-42}"
MIN_MESH_LABELS="${MIN_MESH_LABELS:-0}"
MIN_TEXT_CHARS="${MIN_TEXT_CHARS:-0}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-1}"
export PYTHONUNBUFFERED=1

SAMPLE_FLAGS=(
  --min_mesh_labels "$MIN_MESH_LABELS"
  --min_text_chars "$MIN_TEXT_CHARS"
  --oversample_factor "$OVERSAMPLE_FACTOR"
)

mkdir -p "$(dirname "$SAMPLE")" "$(dirname "$SMOKE")"

echo "=== [1/2] Sample ${MAX_ARTICLES} articles -> ${SAMPLE} (one pass over ${DATA}) ==="
python -u sample.py --data "$DATA" --out "$SAMPLE" --config config/biobert.yaml --max_articles "$MAX_ARTICLES" \
  "${SAMPLE_FLAGS[@]}"

echo "=== [2/2] Subsample ${SMOKE_ARTICLES} articles -> ${SMOKE} (from ${SAMPLE} only) ==="
python -u sample.py --data "$SAMPLE" --out "$SMOKE" --max_articles "$SMOKE_ARTICLES" --seed "$SEED" \
  "${SAMPLE_FLAGS[@]}"

echo "=== Done: ${SAMPLE} (${MAX_ARTICLES}) + ${SMOKE} (${SMOKE_ARTICLES}) ==="

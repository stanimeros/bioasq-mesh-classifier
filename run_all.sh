#!/bin/bash
# Run all experiments. Execute this on the AIDA server.
# Usage: bash run_all.sh [data_path]
# Run detached:  chmod +x run_all.sh && rm -f data/sample.json && nohup ./run_all.sh >> run_all.log 2>&1 & disown
# Tail logs:    tail -f run_all.log

set -e

echo "=== Resetting to origin ==="
git fetch origin
git reset --hard origin/$(git rev-parse --abbrev-ref HEAD)

echo "=== Activating venv ==="
source .venv/bin/activate

DATA="${1:-data/allMeSH_2022.json}"
SAMPLE="data/sample.json"
export PYTHONUNBUFFERED=1

echo "=== [0/3] Sampling ==="
python -u sample.py --data $DATA --out $SAMPLE --config config/biobert.yaml

echo "=== [1/3] Word2Vec + MLP baseline ==="
python -u baseline.py --config config/baseline.yaml --data $SAMPLE

echo "=== [2/3] BioBERT ==="
python -u train.py --config config/biobert.yaml --data $SAMPLE

echo "=== [3/3] SciBERT ==="
python -u train.py --config config/scibert.yaml --data $SAMPLE

echo "=== All done. Results: ==="
for dir in output/baseline output/biobert output/scibert; do
    echo -n "$dir: "
    cat $dir/results.txt 2>/dev/null || echo "no results yet"
done

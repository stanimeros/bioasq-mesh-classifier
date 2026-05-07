#!/bin/bash
# Run all experiments. Execute this on the AIDA server.
# Usage: bash run_all.sh [data_path]

DATA="${1:-data/allMeSH_limitjournals.json}"

echo "=== [1/3] Word2Vec + MLP baseline ==="
python baseline.py --config config/baseline.yaml --data $DATA

echo "=== [2/3] BioBERT ==="
python train.py --config config/biobert.yaml --data $DATA

echo "=== [3/3] SciBERT ==="
python train.py --config config/scibert.yaml --data $DATA

echo "=== All done. Results: ==="
for dir in output/baseline output/biobert output/scibert; do
    echo -n "$dir: "
    cat $dir/results.txt 2>/dev/null || echo "no results yet"
done

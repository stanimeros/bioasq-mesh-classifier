#!/bin/bash
# Run all experiments. Execute this on the AIDA server.
# Usage: bash run_all.sh [data_path]
# Run detached:  nohup ./run_all.sh >> run_all.log 2>&1 & disown
# Tail logs:    tail -f run_all.log

DATA="${1:-data/allMeSH_2022.json}"
export PYTHONUNBUFFERED=1

echo "=== [1/3] Word2Vec + MLP baseline ==="
python -u baseline.py --config config/baseline.yaml --data $DATA

echo "=== [2/3] BioBERT ==="
python -u train.py --config config/biobert.yaml --data $DATA

echo "=== [3/3] SciBERT ==="
python -u train.py --config config/scibert.yaml --data $DATA

echo "=== All done. Results: ==="
for dir in output/baseline output/biobert output/scibert; do
    echo -n "$dir: "
    cat $dir/results.txt 2>/dev/null || echo "no results yet"
done

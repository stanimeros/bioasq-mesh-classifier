#!/bin/bash
# Run all experiments. Execute this on the AIDA server.
# Usage: bash run_all.sh <path_to_data.json>

DATA="${1:-data/allMeSH_limitjournals.json}"
EPOCHS=3
BATCH=16

echo "=== [1/3] Word2Vec + MLP baseline ==="
python baseline.py \
    --data $DATA \
    --output_dir output/baseline

echo "=== [2/3] BioBERT ==="
python train.py \
    --data $DATA \
    --model_name dmis-lab/biobert-base-cased-v1.2 \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --output_dir output/biobert

echo "=== [3/3] SciBERT ==="
python train.py \
    --data $DATA \
    --model_name allenai/scibert_scivocab_uncased \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --output_dir output/scibert

echo "=== All done. Results: ==="
for dir in output/baseline output/biobert output/scibert; do
    echo -n "$dir: "
    cat $dir/results.txt 2>/dev/null || echo "no results yet"
done

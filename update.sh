#!/bin/bash
# Pull latest git state and drop untracked clutter — without removing datasets, samples,
# training outputs, or logs.
#
# Keeps (not removed by git clean): data/ (downloads + sample.json/smoke.json etc.);
# output/ (checkpoints, results); logs/; wandb/; any *.json, *.zip, *.gz (any path).
# Other untracked files may still be removed — commit what you need to keep.
#
# Usage: bash update.sh

set -e

cd "$(dirname "$0")"

echo "=== Killing running train/baseline jobs ==="
pkill -f "python.*train\.py" 2>/dev/null || true
pkill -f "python.*baseline\.py" 2>/dev/null || true
pkill -f "python.*sample\.py" 2>/dev/null || true
sleep 1

echo "=== Fetching latest code ==="
git fetch origin
git reset --hard "origin/$(git rev-parse --abbrev-ref HEAD)"

echo "=== git clean (excluding data/, output/, logs/, wandb/, archives) ==="
git clean -fd \
  -e 'data/' \
  -e 'output/' \
  -e 'logs/' \
  -e 'wandb/' \
  -e '*.json' \
  -e '*.zip' \
  -e '*.gz'

chmod +x \
  run_all.sh \
  update.sh \
  setup.sh \
  create_sample.sh \
  download_bioasq_mesh.sh \
  2>/dev/null || true

echo "=== Done ==="

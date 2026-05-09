#!/bin/bash
# Pull latest git state and drop untracked clutter — without removing datasets or archives.
#
# Keeps (not removed by git clean): entire data/ tree; any *.json, *.zip, *.gz (any path).
# Everything else untracked (e.g. logs/, output/, wandb/) can be removed — commit files you need.
#
# Usage: bash update.sh

set -e

cd "$(dirname "$0")"

echo "=== Fetching latest code ==="
git fetch origin
git reset --hard "origin/$(git rev-parse --abbrev-ref HEAD)"

echo "=== git clean (excluding data/ and common dataset extensions) ==="
git clean -fd \
  -e 'data/' \
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

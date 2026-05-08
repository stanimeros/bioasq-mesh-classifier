#!/bin/bash
# Fetch latest code and hard-reset to origin.
# Usage: bash update.sh

set -e

echo "=== Fetching latest code ==="
git fetch origin
git reset --hard origin/$(git rev-parse --abbrev-ref HEAD)
git clean -fd

echo "=== Done ==="

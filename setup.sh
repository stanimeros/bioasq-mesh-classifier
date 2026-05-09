#!/bin/bash
# Install Python dependencies into a local virtual environment (.venv).
#
# Usage: bash setup.sh
# Then: source .venv/bin/activate

set -e

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Activate with: source .venv/bin/activate"

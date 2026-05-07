#!/bin/bash
# Setup virtual environment and install dependencies.
# Usage: bash setup.sh

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Activate with: source .venv/bin/activate"

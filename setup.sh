#!/bin/bash
# Install Python dependencies into a local virtual environment (.venv).
#
# Usage: bash setup.sh
# Then: source .venv/bin/activate
#
# On Linux with an NVIDIA driver, PyTorch is installed from the CUDA 12.1 wheel
# index first so you get a GPU build that still supports older cards (e.g. Pascal
# TITAN Xp). Plain `pip install torch` can pick a wheel that does not match your
# GPU or driver. Set SKIP_CUDA_TORCH=1 to skip that and use PyPI defaults.

set -e

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

if [[ "${SKIP_CUDA_TORCH:-0}" != "1" ]] && [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== NVIDIA GPU detected (nvidia-smi): installing PyTorch with CUDA 12.1 wheels ==="
  pip install "torch>=2.0.0,<3" --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r requirements.txt

echo "=== Verifying torch ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print(
        "capability sm_%d%d"
        % torch.cuda.get_device_capability(0),
        "| total VRAM GiB (approx):",
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
    )
PY

echo "Setup complete. Activate with: source .venv/bin/activate"

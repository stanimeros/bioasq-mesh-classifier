#!/bin/bash
# Install Python dependencies into a local virtual environment (.venv).
#
# Usage: bash setup.sh
# Then: source .venv/bin/activate
#
# Detached (large downloads; survives SSH disconnect):
#   mkdir -p logs && nohup bash setup.sh >> logs/setup.log 2>&1 & disown
#   tail -f logs/setup.log
#
# On Linux with an NVIDIA driver, PyTorch is installed from a PyTorch wheel index
# before the rest of requirements (so you get a proper CUDA build, not a random CPU wheel).
#   - Driver major version >= 550: CUDA 12.4 + torch>=2.6 (matches current Hugging Face
#     transformers for .bin checkpoint loading).
#   - Driver < 550 (e.g. 535): CUDA 12.1 + torch 2.5.x max; requirements.txt caps
#     transformers so BioASQ BERT weights still load.
# Pascal (e.g. TITAN Xp) is supported by these wheels. Set SKIP_CUDA_TORCH=1 to skip
# the pre-install and use PyPI defaults.

set -e

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

if [[ "${SKIP_CUDA_TORCH:-0}" != "1" ]] && [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  drv_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc '0-9')
  drv_major=${drv_major:-0}
  if [[ "${drv_major}" -ge 550 ]]; then
    echo "=== NVIDIA driver ${drv_major}.x (>=550): PyTorch with CUDA 12.4 wheels (torch>=2.6) ==="
    pip install "torch>=2.6.0,<3" --index-url https://download.pytorch.org/whl/cu124
  else
    echo "=== NVIDIA driver ${drv_major}.x (<550): PyTorch with CUDA 12.1 wheels (torch 2.5.x; OK for Pascal) ==="
    echo "=== (transformers is capped in requirements.txt for torch<2.6 + .bin checkpoints) ==="
    pip install "torch>=2.0.0,<3" --index-url https://download.pytorch.org/whl/cu121
  fi
fi

pip install -r requirements.txt

echo "=== Verifying torch / transformers ==="
python - <<'PY'
import torch
import transformers

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("transformers", transformers.__version__)
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

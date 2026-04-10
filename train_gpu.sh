#!/bin/bash
# train_gpu.sh - Lancia train.py su WSL2 con supporto GPU CUDA
# Da PowerShell: wsl -d Ubuntu -- bash /mnt/c/Users/mmoli/Desktop/AIDALab/Rummo_11042026/rummo_cats_dogs/train_gpu.sh

VENV_DIR="$HOME/.venvs/cats_dogs"
PROJECT_DIR="/mnt/c/Users/mmoli/Desktop/AIDALab/Rummo_11042026/rummo_cats_dogs"

if [ ! -d "$VENV_DIR" ]; then
    echo "Errore: venv WSL non trovato in $VENV_DIR"
    echo "Esegui prima: bash $PROJECT_DIR/setup_wsl.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Imposta LD_LIBRARY_PATH per le librerie NVIDIA CUDA nel venv
NVIDIA_LIBS="$VENV_DIR/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

echo "Python: $(python --version) | venv: $VENV_DIR"
echo ""

cd "$PROJECT_DIR"
python train.py

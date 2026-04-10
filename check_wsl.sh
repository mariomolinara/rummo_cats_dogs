#!/bin/bash
# check_wsl.sh - Verifica rapida dello stato del venv WSL
VENV="$HOME/.venvs/cats_dogs"
echo "=== CHECK WSL2 SETUP ==="
if [ -d "$VENV" ]; then
    echo "Venv: OK ($VENV)"
    source "$VENV/bin/activate"

    # Imposta LD_LIBRARY_PATH per le librerie NVIDIA nel venv
    NVIDIA_LIBS="$VENV/lib/python3.12/site-packages/nvidia"
    export LD_LIBRARY_PATH="$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

    echo "Python: $(python --version)"
    python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'CUDA built: {tf.test.is_built_with_cuda()}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU count: {len(gpus)}')
for g in gpus:
    print(f'  {g}')
" 2>&1 | grep -v "^W\|^I0"
else
    echo "Venv: NON TROVATO"
    echo "Esegui: bash /mnt/c/Users/mmoli/Desktop/AIDALab/Rummo_11042026/rummo_cats_dogs/setup_wsl.sh"
fi

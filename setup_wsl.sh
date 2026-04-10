#!/bin/bash
# setup_wsl.sh - Configura l'ambiente WSL2 con TensorFlow + CUDA per il training
# Eseguire da PowerShell:
#   wsl -d Ubuntu -- bash /mnt/c/Users/mmoli/Desktop/AIDALab/Rummo_11042026/rummo_cats_dogs/setup_wsl.sh

set -e

# Venv nel filesystem Linux nativo (molto piu veloce di /mnt/c)
VENV_DIR="$HOME/.venvs/cats_dogs"
PROJECT_DIR="/mnt/c/Users/mmoli/Desktop/AIDALab/Rummo_11042026/rummo_cats_dogs"

echo "=============================================="
echo "  Setup WSL2 - TensorFlow con GPU CUDA"
echo "=============================================="
echo "  Progetto: $PROJECT_DIR"
echo "  Venv:     $VENV_DIR"
echo ""

# 1. Verifica GPU
echo "[1/4] Verifica GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "  ATTENZIONE: nvidia-smi non trovato."
fi
echo ""

# 2. Assicurati che pip sia disponibile
echo "[2/4] Configurazione pip..."
export PATH="$HOME/.local/bin:$PATH"
if ! python3 -m pip --version &>/dev/null 2>&1; then
    echo "  Installazione pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --user --break-system-packages
fi
echo "  pip: $(python3 -m pip --version 2>/dev/null)"
echo ""

# 3. Crea venv nel filesystem Linux nativo
echo "[3/4] Creazione virtual environment..."
mkdir -p "$HOME/.venvs"
if [ -d "$VENV_DIR" ]; then
    echo "  Venv gia esistente, lo ricreo..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv --without-pip "$VENV_DIR"
source "$VENV_DIR/bin/activate"
curl -sS https://bootstrap.pypa.io/get-pip.py | python
echo "  Venv creato: $VENV_DIR"
echo "  Python: $(python --version)"
echo "  pip:    $(pip --version)"
echo ""

# 4. Installa dipendenze
echo "[4/4] Installazione dipendenze..."
pip install --upgrade pip -q
echo "  Installo TensorFlow con CUDA (puo richiedere qualche minuto)..."
pip install "tensorflow[and-cuda]"
echo "  Installo matplotlib, flask, Pillow, scikit-learn..."
pip install matplotlib flask Pillow scikit-learn -q

echo ""
echo "=============================================="
echo "  Verifica TensorFlow + GPU"
echo "=============================================="
python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'Built with CUDA: {tf.test.is_built_with_cuda()}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU rilevate: {len(gpus)}')
for g in gpus:
    print(f'  - {g.name} ({g.device_type})')
if not gpus:
    print('  WARN: nessuna GPU rilevata da TF.')
"

echo ""
echo "=============================================="
echo "  Setup completato!"
echo ""
echo "  Per lanciare il training con GPU:"
echo "    wsl -d Ubuntu -- bash $PROJECT_DIR/train_gpu.sh"
echo "=============================================="

#!/bin/bash
# install_wsl_deps.sh - Installa dipendenze nel venv WSL
# Log scritto su ~/install_tf.log
VENV="$HOME/.venvs/cats_dogs"
LOG="$HOME/install_tf.log"

exec > "$LOG" 2>&1
echo "=== Inizio installazione: $(date) ==="

source "$VENV/bin/activate"
echo "Pip: $(pip --version)"
echo "Python: $(python --version)"
echo ""

echo "Installazione TensorFlow con CUDA..."
pip install --upgrade pip
pip install "tensorflow[and-cuda]"
echo ""

echo "Installazione altre dipendenze..."
pip install matplotlib flask Pillow scikit-learn
echo ""

echo "=== VERIFICA ==="
python -c "
import tensorflow as tf
print(f'TF {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU: {len(gpus)}')
for g in gpus:
    print(f'  {g}')
"
echo ""
echo "=== Fine installazione: $(date) ==="
echo "DONE"

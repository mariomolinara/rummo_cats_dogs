"""
train.py - AlexNet per classificazione Cani/Gatti
===================================================
Addestra un modello AlexNet con TensorFlow/Keras sul dataset PetImages.
- Pulizia immagini corrotte
- Split: 70% training, 20% validation, 10% test
- EarlyStopping su val_loss con patience=30
- Massimo 30 epoche
- Salva grafico training_history.png e modello alexnet_cats_dogs.keras
"""

import os
import subprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib
matplotlib.use('Agg')  # backend non interattivo
import matplotlib.pyplot as plt
from PIL import Image

# ─── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "PetImages")
IMG_SIZE = (227, 227)
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 30
SEED = 42
MODEL_PATH = os.path.join(BASE_DIR, "alexnet_cats_dogs.keras")
HISTORY_IMG = os.path.join(BASE_DIR, "training_history.png")


# ─── Rilevamento e configurazione GPU ──────────────────────────────────────────
def setup_gpu():
    """
    Rileva GPU disponibili (CUDA, DirectML, ecc.), configura memory growth
    e abilita mixed precision se possibile. Ritorna una stringa descrittiva.
    """
    print("-" * 60)
    print("  RILEVAMENTO HARDWARE")
    print("-" * 60)

    # Info sistema
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  Built with CUDA:    {tf.test.is_built_with_cuda()}")

    # Rileva GPU hardware tramite nvidia-smi (se presente)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split("\n")):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    print(f"  GPU hardware #{i}: {parts[0]}, "
                          f"{parts[1]} MB VRAM, Driver {parts[2]}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  nvidia-smi non trovato — nessuna GPU NVIDIA rilevata")

    # Rileva GPU visibili da TensorFlow
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n  ✅ GPU rilevate da TensorFlow: {len(gpus)}")
        for gpu in gpus:
            print(f"     - {gpu.name} ({gpu.device_type})")

        # Configura memory growth per evitare OOM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"     Memory growth abilitato per {gpu.name}")
            except RuntimeError as e:
                print(f"     [WARN] Memory growth fallito per {gpu.name}: {e}")

        # Abilita mixed precision (float16) per velocizzare su GPU
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("  ⚡ Mixed precision (float16) abilitata")
        except Exception as e:
            print(f"  [WARN] Mixed precision non disponibile: {e}")

        device_info = f"GPU ({gpus[0].name})"
    else:
        print(f"\n  ⚠️  Nessuna GPU rilevata da TensorFlow.")
        print("     Il training verrà eseguito su CPU.")
        print("     NOTA: TensorFlow >= 2.11 su Windows nativo non supporta CUDA.")
        print("     Per usare la GPU NVIDIA, usa WSL2 oppure installa")
        print("     il plugin tensorflow-directml (richiede Python <= 3.12).")
        device_info = "CPU"

    print("-" * 60 + "\n")
    return device_info


# ─── Step 1: Pulizia immagini corrotte ─────────────────────────────────────────
def clean_dataset(data_dir):
    """
    Pulisce il dataset: rimuove file non-immagine e immagini corrotte,
    e RI-SALVA TUTTE le immagini come JPEG RGB per garantire compatibilità
    con il decoder TensorFlow (che accetta solo 1, 3 o 4 canali).
    Questo risolve problemi con PNG LA (2 canali), palette, CMYK, ecc.
    """
    removed = 0
    resaved = 0
    total = 0
    for class_name in ("Cat", "Dog"):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Cartella non trovata: {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            total += 1
            # Rimuovi file non-immagine
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
                removed += 1
                continue
            try:
                with Image.open(fpath) as img:
                    img.load()  # forza caricamento completo per rilevare errori
                    # Converti SEMPRE a RGB e risalva come JPEG pulito
                    rgb_img = img.convert("RGB")
                    new_path = os.path.splitext(fpath)[0] + ".jpg"
                    rgb_img.save(new_path, "JPEG", quality=95)
                    # Rimuovi il file originale se aveva estensione diversa
                    if new_path != fpath:
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
                    resaved += 1
            except Exception:
                print(f"  Rimossa immagine corrotta: {fpath}")
                try:
                    os.remove(fpath)
                except OSError:
                    pass
                removed += 1
    print(f"Pulizia completata su {total} file:")
    print(f"  {removed} rimossi, {resaved} ri-salvati come JPEG RGB.\n")


# ─── Step 2: Caricamento e split del dataset ──────────────────────────────────
def load_datasets(data_dir):
    """Carica il dataset e lo divide in train (70%), val (20%), test (10%)."""

    # Carica tutto il dataset con label binarie (Cat=0, Dog=1 in ordine alfabetico)
    full_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=SEED,
        shuffle=True,
    )

    # Stampa mapping classi
    class_names = full_ds.class_names
    print(f"Classi trovate: {class_names}")
    print(f"  0 = {class_names[0]}, 1 = {class_names[1]}\n")

    # Calcola numero totale di batch
    total_batches = tf.data.experimental.cardinality(full_ds).numpy()
    print(f"Batch totali: {total_batches}")

    train_size = int(0.7 * total_batches)
    val_size = int(0.2 * total_batches)
    # test_size = totale - train - val

    train_ds = full_ds.take(train_size)
    remaining = full_ds.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)

    print(f"  Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"  Val   batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"  Test  batches: {tf.data.experimental.cardinality(test_ds).numpy()}\n")

    # Ottimizzazione performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# ─── Step 3: Definizione architettura AlexNet ──────────────────────────────────
def build_alexnet():
    """Costruisce il modello AlexNet adattato per classificazione binaria."""
    model = keras.Sequential([
        # Normalizzazione pixel [0,255] -> [0,1]
        layers.Rescaling(1.0 / 255, input_shape=(227, 227, 3)),

        # Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Conv Block 1
        layers.Conv2D(96, (11, 11), strides=4, activation="relu", padding="valid"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),

        # Conv Block 2
        layers.Conv2D(256, (5, 5), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),

        # Conv Block 3
        layers.Conv2D(384, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),

        # Conv Block 4
        layers.Conv2D(384, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),

        # Conv Block 5
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),

        # Classificatore
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),  # binario
    ], name="AlexNet")

    return model


# ─── Step 4: Training ─────────────────────────────────────────────────────────
def train_model(model, train_ds, val_ds):
    """Compila e addestra il modello."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1,
    )

    return history


# ─── Step 5: Grafico loss ─────────────────────────────────────────────────────
def plot_history(history):
    """Genera e salva il grafico loss training vs validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history["loss"], label="Training Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_title("Loss - Training vs Validation", fontsize=14)
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss (Binary Crossentropy)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax2.set_title("Accuracy - Training vs Validation", fontsize=14)
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_IMG, dpi=150)
    print(f"\nGrafico salvato in: {HISTORY_IMG}")
    plt.close()


# ─── Step 6: Valutazione su test set ──────────────────────────────────────────
def evaluate_model(model, test_ds):
    """Valuta il modello sul test set."""
    print("\n" + "=" * 60)
    print("VALUTAZIONE SUL TEST SET")
    print("=" * 60)
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60)
    return loss, accuracy


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AlexNet - Classificazione Cani vs Gatti")
    print("=" * 60 + "\n")

    # 0. Rilevamento GPU
    device_info = setup_gpu()

    # 1. Pulizia
    print("[1/6] Pulizia immagini corrotte...")
    clean_dataset(DATA_DIR)

    # 2. Caricamento dataset
    print("[2/6] Caricamento e split del dataset...")
    train_ds, val_ds, test_ds, class_names = load_datasets(DATA_DIR)

    # 3. Costruzione modello
    print("[3/6] Costruzione modello AlexNet...")
    model = build_alexnet()

    # 4. Training
    print("[4/6] Avvio training...")
    history = train_model(model, train_ds, val_ds)

    # 5. Grafico
    print("[5/6] Generazione grafico...")
    plot_history(history)

    # 6. Valutazione test
    print("[6/6] Valutazione sul test set...")
    evaluate_model(model, test_ds)

    # Salvataggio modello
    model.save(MODEL_PATH)
    print(f"\nModello salvato in: {MODEL_PATH}")
    print(f"Device utilizzato: {device_info}")
    print("\nTraining completato con successo!")


if __name__ == "__main__":
    main()


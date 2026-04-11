"""
app.py - Web App Flask per classificazione Cani/Gatti
=====================================================
Carica il modello AlexNet addestrato e consente di classificare
immagini caricate via browser. I risultati vengono mostrati in una
tabella cumulativa con persistenza su file JSON.

Avvio: python app.py
URL:   http://localhost:5000
"""

import os
import json
import time
import uuid
import subprocess
from datetime import datetime

import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for

# ─── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "alexnet_cats_dogs.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
HISTORY_FILE = os.path.join(BASE_DIR, "classification_history.json")
IMG_SIZE = (227, 227)
CLASS_NAMES = ["Cat", "Dog"]  # 0=Cat, 1=Dog (ordine alfabetico)


# ─── Rilevamento e configurazione GPU ──────────────────────────────────────────
def setup_gpu():
    """
    Rileva GPU disponibili, configura memory growth e restituisce
    una stringa descrittiva del device in uso e la lista dei device disponibili.
    """
    print("-" * 50)
    print("  RILEVAMENTO HARDWARE")
    print("-" * 50)
    print(f"  TensorFlow version: {tf.__version__}")

    # Rileva GPU hardware tramite nvidia-smi
    gpu_hw_name = "N/A"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 2:
                gpu_hw_name = f"{parts[0]} ({parts[1]} MB VRAM)"
                print(f"  GPU hardware: {gpu_hw_name}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Rileva GPU visibili da TensorFlow
    gpus = tf.config.list_physical_devices("GPU")
    available_devices = ["CPU"]
    if gpus:
        print(f"  ✅ GPU TensorFlow: {len(gpus)} rilevata/e")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        available_devices.append("GPU")
        default_device = "GPU"
    else:
        print("  ⚠️  GPU non disponibile per TF — solo CPU")
        default_device = "CPU"

    print(f"  Device disponibili: {available_devices}")
    print("-" * 50)
    return default_device, available_devices, gpu_hw_name


# ─── Setup GPU all'avvio ───────────────────────────────────────────────────────
default_device, available_devices, gpu_hw_name = setup_gpu()

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # max 16 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}

# ─── Caricamento modello ───────────────────────────────────────────────────────
print("Caricamento modello AlexNet...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modello non trovato: {MODEL_PATH}\n"
        "Esegui prima 'python train.py' per addestrare il modello."
    )
model = tf.keras.models.load_model(MODEL_PATH)
print("Modello caricato con successo!\n")

# Crea cartella uploads se non esiste
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─── Utility ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_history():
    """Carica lo storico delle classificazioni dal file JSON."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_history(history):
    """Salva lo storico delle classificazioni su file JSON."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def classify_image(filepath, use_device="CPU"):
    """
    Classifica un'immagine e restituisce un dizionario con tutti i dettagli.
    use_device: "CPU" o "GPU" — forza l'inferenza sul device scelto.
    """
    # Apri immagine per ottenere dimensioni originali
    with Image.open(filepath) as pil_img:
        original_size = f"{pil_img.width} × {pil_img.height} px"
        # Converti in RGB se necessario
        pil_img_rgb = pil_img.convert("RGB")
        # Ridimensiona
        pil_img_resized = pil_img_rgb.resize(IMG_SIZE, Image.LANCZOS)

    # Prepara per il modello: array (1, 227, 227, 3) con valori [0,255]
    # La normalizzazione è già nel modello (layer Rescaling)
    img_array = np.array(pil_img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Scegli il device per l'inferenza
    if use_device == "GPU" and "GPU" in available_devices:
        device_str = "/GPU:0"
        device_label = f"GPU ({gpu_hw_name})"
    else:
        device_str = "/CPU:0"
        device_label = "CPU"

    # Inferenza con misurazione tempo sul device scelto
    start_time = time.perf_counter()
    with tf.device(device_str):
        prediction = model.predict(img_array, verbose=0)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    raw_prob = float(prediction[0][0])  # probabilità classe Dog
    predicted_class = CLASS_NAMES[1] if raw_prob >= 0.5 else CLASS_NAMES[0]
    confidence = raw_prob if raw_prob >= 0.5 else (1.0 - raw_prob)

    return {
        "predicted_class": predicted_class,
        "confidence_pct": round(confidence * 100, 2),
        "raw_probability": round(raw_prob, 6),
        "original_size": original_size,
        "inference_ms": round(elapsed_ms, 2),
        "device": device_label,
    }


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    history = load_history()
    # Statistiche
    total = len(history)
    cats = sum(1 for h in history if h["predicted_class"] == "Cat")
    dogs = total - cats
    return render_template(
        "index.html",
        history=history,
        total=total,
        cats=cats,
        dogs=dogs,
        available_devices=available_devices,
        default_device=default_device,
        gpu_hw_name=gpu_hw_name,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    # Salva con nome univoco per evitare sovrascritture
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    # Classifica con il device scelto dall'utente
    chosen_device = request.form.get("device", default_device)
    if chosen_device not in available_devices:
        chosen_device = "CPU"
    result = classify_image(save_path, use_device=chosen_device)

    # Crea record
    history = load_history()
    record = {
        "id": len(history) + 1,
        "filename": file.filename,
        "saved_as": unique_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_class": result["predicted_class"],
        "confidence_pct": result["confidence_pct"],
        "raw_probability": result["raw_probability"],
        "original_size": result["original_size"],
        "inference_ms": result["inference_ms"],
        "device": result["device"],
    }
    history.append(record)
    save_history(history)

    return redirect(url_for("index"))


@app.route("/clear", methods=["POST"])
def clear():
    """Cancella lo storico e le immagini caricate."""
    # Svuota file JSON
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

    # Rimuovi immagini caricate
    if os.path.isdir(UPLOAD_FOLDER):
        for fname in os.listdir(UPLOAD_FOLDER):
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

    return redirect(url_for("index"))


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Cat vs Dog Classifier - Web App")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5000)


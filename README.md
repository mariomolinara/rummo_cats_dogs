# 🐱🐶 Cat vs Dog Classifier — AlexNet con TensorFlow

Classificatore di immagini **Cani vs Gatti** basato sull'architettura **AlexNet**, implementato con TensorFlow/Keras. Include una web app Flask per classificare immagini dal browser.

---

## 📋 Requisiti

- **Python 3.10–3.13** (verificato con Python 3.13.7 + TensorFlow 2.21.0)
- Il dataset **PetImages** con le sottocartelle `Cat/` e `Dog/` già presente nella cartella del progetto

---

## 🚀 Installazione passo-passo

### 1. Apri il terminale nella cartella del progetto

```powershell
cd C:\Users\mmoli\Desktop\AIDALab\Rummo_11042026\rummo_cats_dogs
```

### 2. Crea un ambiente virtuale (venv)

Se il venv attuale usa Python 3.13 e TensorFlow non si installa, ricrea il venv con una versione compatibile:

```powershell
# Rimuovi il venv esistente (se necessario)
Remove-Item -Recurse -Force .venv

# Crea un nuovo venv con Python 3.11 (o 3.12)
py -3.11 -m venv .venv
```

### 3. Attiva il venv

```powershell
.\.venv\Scripts\Activate.ps1
```

> Se ricevi un errore sulla policy di esecuzione, esegui prima:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 4. Installa le dipendenze

```powershell
pip install -r requirements.txt
```

---

## 🏋️ Training del modello

### Opzione A — Training su CPU (Windows nativo)

```powershell
python train.py
```

### Opzione B — Training su GPU con WSL2 (consigliato, molto più veloce)

TensorFlow ≥2.11 su Windows nativo non supporta CUDA. Per usare la GPU NVIDIA (es. RTX 4080), esegui il training tramite WSL2:

```powershell
# 1. Setup iniziale WSL2 (solo la prima volta)
wsl -d Ubuntu -- bash /mnt/c/Users/mmoli/Desktop/AIDALab/Rummo_11042026/rummo_cats_dogs/setup_wsl.sh

# 2. Avvia il training con GPU
wsl -d Ubuntu -- bash /mnt/c/Users/mmoli/Desktop/AIDALab/Rummo_11042026/rummo_cats_dogs/train_gpu.sh
```

Cosa fa `train.py`:
1. Rileva automaticamente la GPU (se disponibile)
2. Pulisce il dataset rimuovendo immagini corrotte
3. Divide i dati in training (70%), validation (20%) e test (10%)
4. Addestra AlexNet per massimo 30 epoche (si ferma prima se l'errore non migliora)
5. Genera il grafico `training_history.png`
6. Salva il modello in `alexnet_cats_dogs.keras`

⏱️ **Durata stimata**: ~5-10 min con GPU, ~30-60 min su CPU.

---

## 🌐 Avvio della Web App

Dopo il training, avvia la web app:

```powershell
python app.py
```

Poi apri il browser e vai su:

👉 **http://localhost:5000**

Dalla pagina puoi:
- **Caricare un'immagine** di un gatto o un cane
- **Vedere la classificazione** con dettagli tecnici (confidenza, tempo di inferenza, ecc.)
- **Consultare lo storico** di tutte le classificazioni in una tabella
- **Cancellare lo storico** con un pulsante

---

## 📁 Struttura del progetto

```
rummo_cats_dogs/
├── PetImages/             # Dataset
│   ├── Cat/               # ~12.500 immagini di gatti
│   └── Dog/               # ~12.500 immagini di cani
├── static/
│   └── uploads/           # Immagini caricate via web
├── templates/
│   └── index.html         # Template della web app
├── train.py               # Script di training AlexNet
├── app.py                 # Web app Flask
├── setup_wsl.sh           # Setup ambiente WSL2 con GPU CUDA
├── train_gpu.sh           # Lancia training su WSL2 con GPU
├── requirements.txt       # Dipendenze Python
├── alexnet_cats_dogs.keras # Modello (generato dopo il training)
├── training_history.png   # Grafico loss (generato dopo il training)
├── classification_history.json  # Storico classificazioni web
└── README.md              # Questo file
```

---

## ❓ Problemi comuni

| Problema | Soluzione |
|---|---|
| `pip install tensorflow` fallisce | Usa Python 3.11 o 3.12, non 3.13 |
| `ModuleNotFoundError: No module named 'tensorflow'` | Attiva il venv: `.\.venv\Scripts\Activate.ps1` |
| `Modello non trovato` quando avvii `app.py` | Esegui prima `python train.py` |
| Training molto lento su CPU | Usa WSL2 con GPU: `bash train_gpu.sh` |
| GPU non rilevata in WSL2 | Verifica driver NVIDIA Windows e che WSL2 sia versione 2 (`wsl --list -v`) |


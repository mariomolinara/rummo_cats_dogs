# 🐱🐶 Cat vs Dog Classifier — AlexNet con TensorFlow

Classificatore di immagini **Cani vs Gatti** basato sull'architettura **AlexNet**, implementato con TensorFlow/Keras. Include una web app Flask per classificare immagini dal browser.

---

## 🎓 Contesto didattico

Questo progetto è stato preparato come materiale pratico per un **breve corso introduttivo al Deep Learning e all'Intelligenza Artificiale** tenuto presso il **Liceo Scientifico "G. Rummo" di Benevento**.

Gli incontri si sono svolti in due giornate:
- 📅 **11 aprile 2026**
- 📅 **18 aprile 2026**

Il corso è stato tenuto dal **Prof. Mario Molinara** dell'**Università degli Studi di Cassino e del Lazio Meridionale**.

---

## 📦 Download del dataset

Il dataset va scaricato manualmente da Kaggle:

👉 **https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset/data**

Dopo il download, crea la seguente struttura di cartelle nella directory del progetto:

```
rummo_cats_dogs/
└── PetImages/
    ├── Cat/       ← inserisci qui tutte le immagini di gatti
    └── Dog/       ← inserisci qui tutte le immagini di cani
```

Le immagini contenute in queste cartelle verranno automaticamente suddivise dallo script `train.py` in **training (70%)**, **validation (20%)** e **test (10%)**.

> ⚠️ Non serve creare sottocartelle separate per train/val/test: ci pensa lo script!

---

## 📋 Requisiti

- **Python 3.10–3.13** (verificato con Python 3.13.7 + TensorFlow 2.21.0)
- Il dataset **PetImages** con le sottocartelle `Cat/` e `Dog/` (vedi sezione precedente)

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

## 💻 IDE consigliati

Per lavorare comodamente con il codice sorgente di questo progetto, si consiglia l'uso di un ambiente di sviluppo integrato (IDE). Ecco due ottime opzioni gratuite:

| IDE | Edizione gratuita | Download |
|-----|-------------------|----------|
| **PyCharm** | Community Edition (gratuita, open source) | [jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download/) |
| **Visual Studio Code** | Completamente gratuito | [code.visualstudio.com](https://code.visualstudio.com/) |

### Perché usare un IDE?

- **Syntax highlighting**: il codice Python è colorato e più leggibile
- **Autocompletamento**: suggerimenti intelligenti mentre scrivi
- **Terminale integrato**: puoi lanciare `train.py` e `app.py` direttamente dall'IDE
- **Debugger**: esegui il codice passo-passo per capire cosa succede
- **Gestione del venv**: entrambi gli IDE rilevano automaticamente l'ambiente virtuale

> 💡 **Consiglio per principianti**: PyCharm Community è pensato specificamente per Python ed è molto intuitivo. Visual Studio Code è più leggero e versatile, ma richiede l'installazione dell'estensione Python.

---

## 🌐 Avvio della Web App

Dopo il training, avvia la web app.

### Opzione A — Web App su CPU (Windows nativo)

```powershell
python app.py
```

### Opzione B — Web App su GPU con WSL2 (inferenza più veloce)

```powershell
wsl -d Ubuntu -- bash /mnt/c/Users/mmoli/Desktop/AIDALab/Rummo_11042026/rummo_cats_dogs/app_gpu.sh
```

Poi apri il browser e vai su:

👉 **http://localhost:5000**

Dalla pagina puoi:
- **Scegliere il device** (CPU o GPU) dal menu a tendina prima di classificare
- **Caricare un'immagine** di un gatto o un cane
- **Vedere la classificazione** con dettagli tecnici (confidenza, tempo di inferenza, ecc.)
- **Cliccare sull'anteprima** per aprire una dialog con immagine ingrandita e tutti i dettagli
- **Consultare lo storico** di tutte le classificazioni in una tabella
- **Cancellare lo storico** con un pulsante

---

## 🧠 La rete AlexNet

La rete utilizzata in questo progetto è **AlexNet**, una delle **primissime reti neurali convoluzionali di tipo Deep** nella storia dell'informatica. Proposta da Alex Krizhevsky, Ilya Sutskever e Geoffrey Hinton nel 2012, AlexNet ha vinto la competizione **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** con un margine enorme rispetto ai metodi tradizionali, segnando l'inizio dell'era moderna del Deep Learning.

Le caratteristiche principali di AlexNet sono:
- **5 livelli convoluzionali** con filtri di dimensioni decrescenti (11×11, 5×5, 3×3)
- **3 livelli fully-connected** (densi) per la classificazione finale
- Uso di **ReLU** come funzione di attivazione (al posto della sigmoide, molto più lenta)
- **Dropout** per la regolarizzazione e prevenzione dell'overfitting
- **Batch Normalization** per stabilizzare il training

In questo progetto AlexNet è adattata per la **classificazione binaria** (Gatto vs Cane) con un'uscita sigmoide.

### 🏗️ Architettura della rete AlexNet (questo progetto)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Immagine 227 × 227 × 3 (RGB)                │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RESCALING          1/255 normalizzazione  →  valori da [0,255] a [0,1]    │
│  DATA AUGMENTATION  RandomFlip + RandomRotation + RandomZoom               │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────────────────┐
          │     BLOCCHI CONVOLUZIONALI (feature extraction)    │
          │                                                    │
          │  ┌──────────────────────────────────────────────┐  │
          │  │ CONV1  96 filtri 11×11, stride 4  → ReLU     │  │
          │  │        BatchNorm → MaxPool 3×3, stride 2     │  │
          │  │        Output: 27 × 27 × 96                  │  │
          │  └──────────────────┬───────────────────────────┘  │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐  │
          │  │ CONV2  256 filtri 5×5, pad same  → ReLU      │  │
          │  │        BatchNorm → MaxPool 3×3, stride 2     │  │
          │  │        Output: 13 × 13 × 256                 │  │
          │  └──────────────────┬───────────────────────────┘  │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐  │
          │  │ CONV3  384 filtri 3×3, pad same  → ReLU      │  │
          │  │        BatchNorm                              │  │
          │  │        Output: 13 × 13 × 384                 │  │
          │  └──────────────────┬───────────────────────────┘  │
          │                     ▼                              │
          │  ┌────────────────────────────────────────────────┐  │
          │  │ CONV4  384 filtri 3×3, pad same  → ReLU        │  │
          │  │        BatchNorm                              │  │
          │  │        Output: 13 × 13 × 384                 │  │
          │  └──────────────────┬───────────────────────────┘  │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐  │
          │  │ CONV5  256 filtri 3×3, pad same  → ReLU      │  │
          │  │        BatchNorm → MaxPool 3×3, stride 2     │  │
          │  │        Output: 6 × 6 × 256 = 9.216 valori   │  │
          │  └──────────────────┬───────────────────────────┘  │
          │                     │                              │
          └─────────────────────┼──────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FLATTEN           9.216 valori → vettore monodimensionale                 │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────────────────┐
          │       CLASSIFICATORE (fully-connected)             │
          │                                                    │
          │  ┌──────────────────────────────────────────────┐  │
          │  │ DENSE 1   4096 neuroni → ReLU → Dropout 50%  │  │
          │  └──────────────────┬───────────────────────────┘  │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐  │
          │  │ DENSE 2   4096 neuroni → ReLU → Dropout 50%  │  │
          │  └──────────────────┬───────────────────────────┘  │
          │                     ▼                              │
          │  ┌──────────────────────────────────────────────┐  │
          │  │ DENSE 3   1 neurone → Sigmoid                │  │
          │  └──────────────────┬───────────────────────────┘  │
          │                     │                              │
          └─────────────────────┼──────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT:  0.0 ──────────── 0.5 ──────────── 1.0                            │
│           🐱 Cat                             🐶 Dog                        │
│           (confidenza = 1 - valore)          (confidenza = valore)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Totale parametri**: ~58 milioni (la maggior parte nei livelli Dense)

### 📐 Schema del flusso di Training

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   DATASET    │    │  PULIZIA     │    │    SPLIT     │
│  PetImages/  │───▶│  Rimozione   │───▶│  70% Train   │
│  Cat/ + Dog/ │    │  img corrotte│    │  20% Valid.  │
│  ~25.000 img │    │  Conversione │    │  10% Test    │
└──────────────┘    │  a JPEG RGB  │    └──────┬───────┘
                    └──────────────┘           │
                                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                             │
│                                                                  │
│  Per ogni epoca (max 30):                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  1. Forward pass su TRAINING SET (batch da 32 immagini)    │  │
│  │     Immagine → AlexNet → predizione → loss                 │  │
│  │                                                            │  │
│  │  2. Backward pass (backpropagation)                        │  │
│  │     Calcolo gradienti → aggiornamento pesi (Adam)          │  │
│  │                                                            │  │
│  │  3. Valutazione su VALIDATION SET                          │  │
│  │     Calcolo val_loss e val_accuracy                        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  EarlyStopping: se val_loss non migliora per 30 epoche → STOP   │
│  Restore best weights: torna ai pesi della migliore epoca       │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  VALUTAZIONE FINALE su TEST SET (mai visto durante il training)  │
│  → Accuracy e Loss finali                                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌─────────────────┐   ┌──────────────────┐
          │  Modello salvato │   │  Grafico salvato  │
          │  .keras          │   │  .png             │
          └─────────────────┘   └──────────────────┘
```

---

## 🖥️ Architettura Software del progetto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            UTENTE (Browser)                                 │
│                         http://localhost:5000                                │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  HTTP (upload immagine + scelta device)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        app.py  (Flask Web Server)                           │
│                                                                             │
│  ┌─────────────┐   ┌───────────────┐   ┌────────────────────────────────┐  │
│  │  GET /       │   │ POST /predict │   │  POST /clear                  │  │
│  │  Mostra form │   │ Classifica    │   │  Cancella storico             │  │
│  │  + storico   │   │ immagine      │   │  e immagini caricate          │  │
│  └──────┬──────┘   └───────┬───────┘   └────────────────────────────────┘  │
│         │                  │                                                │
│         │                  ▼                                                │
│         │  ┌──────────────────────────────────────────────────────────┐     │
│         │  │              PIPELINE DI INFERENZA                       │     │
│         │  │                                                          │     │
│         │  │  1. Salva immagine in static/uploads/                    │     │
│         │  │  2. Apri con Pillow → dimensioni originali               │     │
│         │  │  3. Converti a RGB → Resize 227×227                      │     │
│         │  │  4. NumPy array (1, 227, 227, 3) float32                │     │
│         │  │  5. tf.device(CPU o GPU) → model.predict()              │     │
│         │  │  6. Sigmoid output → classe + confidenza                 │     │
│         │  │  7. Misura tempo di inferenza                            │     │
│         │  └───────────────────────┬──────────────────────────────────┘     │
│         │                          │                                        │
│         │                          ▼                                        │
│         │  ┌──────────────────────────────────────────────────────────┐     │
│         │  │         classification_history.json                      │     │
│         │  │         (persistenza risultati su disco)                 │     │
│         │  └──────────────────────────────────────────────────────────┘     │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  templates/index.html  (Jinja2)                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Form upload + selettore device (CPU/GPU)                      │  │   │
│  │  ├────────────────────────────────────────────────────────────────┤  │   │
│  │  │  Statistiche: totale classificazioni, gatti, cani              │  │   │
│  │  ├────────────────────────────────────────────────────────────────┤  │   │
│  │  │  Tabella storico (anteprima cliccabile → dialog modale)        │  │   │
│  │  ├────────────────────────────────────────────────────────────────┤  │   │
│  │  │  Dialog modale: immagine grande + dettagli tecnici             │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HARDWARE / RUNTIME                                   │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐   │
│  │  Opzione A: CPU          │    │  Opzione B: GPU via WSL2             │   │
│  │  Windows nativo          │    │  Ubuntu + CUDA + cuDNN               │   │
│  │  Python 3.13 + TF 2.21  │    │  Python 3.12 + TF 2.21 [and-cuda]   │   │
│  │  .venv\                  │    │  ~/.venvs/cats_dogs/                 │   │
│  │                          │    │  LD_LIBRARY_PATH → librerie NVIDIA   │   │
│  │  ⏱️ Inferenza: ~50-200ms │    │  ⏱️ Inferenza: ~5-20ms               │   │
│  └──────────────────────────┘    └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
├── app_gpu.sh             # Lancia web app su WSL2 con GPU
├── requirements.txt       # Dipendenze Python
├── alexnet_cats_dogs.keras # Modello (generato dopo il training)
├── training_history.png   # Grafico loss (generato dopo il training)
├── classification_history.json  # Storico classificazioni web
└── README.md              # Questo file
```

---

## 🧩 Ruolo dei componenti software

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STACK SOFTWARE                                     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  TensorFlow / Keras                                               │  │
│  │  ─────────────────                                                │  │
│  │  • Definizione dell'architettura AlexNet (Sequential API)         │  │
│  │  • Compilazione: ottimizzatore Adam, loss binary_crossentropy     │  │
│  │  • Training: model.fit() con EarlyStopping                        │  │
│  │  • Inferenza: model.predict() su CPU o GPU                        │  │
│  │  • Salvataggio/caricamento modello (.keras)                       │  │
│  │  • Gestione device: tf.device("/CPU:0" o "/GPU:0")                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Flask                                                            │  │
│  │  ─────                                                            │  │
│  │  • Web server HTTP sulla porta 5000                               │  │
│  │  • Routing: GET /, POST /predict, POST /clear                     │  │
│  │  • Upload file multipart                                          │  │
│  │  • Template rendering con Jinja2                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Pillow (PIL)                                                     │  │
│  │  ───────────                                                      │  │
│  │  • Apertura e validazione immagini                                │  │
│  │  • Conversione a RGB (da CMYK, LA, P, ecc.)                      │  │
│  │  • Resize a 227×227 con filtro LANCZOS                           │  │
│  │  • Pulizia dataset: ri-salvataggio come JPEG pulito               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Matplotlib                                                       │  │
│  │  ──────────                                                       │  │
│  │  • Grafico loss training vs validation                            │  │
│  │  • Grafico accuracy training vs validation                        │  │
│  │  • Salvataggio come PNG ad alta risoluzione (150 dpi)             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  NumPy                                                            │  │
│  │  ─────                                                            │  │
│  │  • Conversione immagine Pillow → array numerico float32           │  │
│  │  • Expand dims per creare il batch di input (1, 227, 227, 3)     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  NVIDIA CUDA + cuDNN  (solo via WSL2)                             │  │
│  │  ──────────────────────────────                                   │  │
│  │  • Accelerazione GPU per operazioni matriciali (Conv2D, Dense)    │  │
│  │  • cuDNN: kernel ottimizzati per reti neurali convoluzionali      │  │
│  │  • Memory growth: allocazione dinamica VRAM                       │  │
│  │  • Mixed precision (float16): raddoppia il throughput su GPU      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Stima computazionale: FLOPS, tempi, potenza ed energia

Stime calcolate per l'hardware di riferimento utilizzato durante il corso:
- **CPU**: Intel Core i9-14900HX (24 core, picco teorico ~1 TFLOPS FP32)
- **GPU**: NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MB VRAM (picco teorico ~12.4 TFLOPS FP32)

### Calcolo dei FLOPS per layer (singola immagine, forward pass)

Per un livello convoluzionale: `FLOPS ≈ 2 × H_out × W_out × K_h × K_w × C_in × C_out`

| Layer | Operazione | Output | FLOPS stimati |
|-------|-----------|--------|---------------|
| Rescaling | Div per 255 | 227×227×3 | ~0.15 MFLOPS |
| **CONV1** | 96 filtri 11×11, stride 4 | 55×55×96 | **~211 MFLOPS** |
| MaxPool1 | 3×3, stride 2 | 27×27×96 | ~0.7 MFLOPS |
| **CONV2** | 256 filtri 5×5, same | 27×27×256 | **~896 MFLOPS** |
| MaxPool2 | 3×3, stride 2 | 13×13×256 | ~0.2 MFLOPS |
| **CONV3** | 384 filtri 3×3, same | 13×13×384 | **~299 MFLOPS** |
| **CONV4** | 384 filtri 3×3, same | 13×13×384 | **~449 MFLOPS** |
| **CONV5** | 256 filtri 3×3, same | 13×13×256 | **~299 MFLOPS** |
| MaxPool5 | 3×3, stride 2 | 6×6×256 | ~0.1 MFLOPS |
| Flatten | → 9.216 valori | 9216 | — |
| **DENSE1** | 9216→4096, ReLU | 4096 | **~75.5 MFLOPS** |
| **DENSE2** | 4096→4096, ReLU | 4096 | **~33.6 MFLOPS** |
| **DENSE3** | 4096→1, Sigmoid | 1 | ~0.008 MFLOPS |
| | | **TOTALE FORWARD** | **≈ 2.26 GFLOPS** |

> **Nota**: il backward pass (backpropagation) richiede circa 2× i FLOPS del forward pass.
> Pertanto un singolo step di training (forward + backward) costa circa **6.8 GFLOPS per immagine**.

### Dataset e batching

| Parametro | Valore |
|-----------|--------|
| Immagini totali | ~24.800 |
| Training set (70%) | ~17.360 immagini → ~542 batch |
| Validation set (20%) | ~4.960 immagini → ~155 batch |
| Test set (10%) | ~2.480 immagini → ~78 batch |
| Batch size | 32 |

### FLOPS totali per epoca

| Fase | Batch | FLOPS/batch | FLOPS totale |
|------|-------|-------------|-------------|
| Training (forward+backward) | 542 | 32 × 6.8 = 217.6 GFLOPS | **~118 TFLOPS** |
| Validation (solo forward) | 155 | 32 × 2.26 = 72.3 GFLOPS | **~11.2 TFLOPS** |
| **Totale per epoca** | | | **≈ 129 TFLOPS** |

### ⏱️ Tempi, FLOPS rate, potenza ed energia

#### Training (150 epoche)

| | **CPU** (i9-14900HX) | **GPU** (RTX 4080 Laptop) |
|---|---|---|
| Tempo per epoca | ~260 s | ~45 s |
| **Tempo 150 epoche** | **39.000 s ≈ 10 h 50 min** | **6.750 s ≈ 1 h 52 min** |
| FLOPS totali (150 ep.) | ~19.350 TFLOPS ≈ **19.4 PFLOPS** | ~19.350 TFLOPS ≈ **19.4 PFLOPS** |
| Throughput effettivo | ~496 GFLOPS/s | ~2.867 GFLOPS/s ≈ **2.87 TFLOPS/s** |
| Utilizzo picco teorico | ~50% del picco CPU | ~23% del picco GPU |
| Potenza assorbita (stima) | ~155 W ¹ | ~185 W ² |
| **Energia totale** | 155 W × 39.000 s = **6.045 kJ ≈ 1.68 kWh** | 185 W × 6.750 s = **1.249 kJ ≈ 0.35 kWh** |
| **Costo energetico** ³ | ~€ 0.50 | ~€ 0.10 |

#### Training (30 epoche — configurazione attuale con EarlyStopping)

| | **CPU** (i9-14900HX) | **GPU** (RTX 4080 Laptop) |
|---|---|---|
| **Tempo 30 epoche** | **7.800 s ≈ 2 h 10 min** | **1.350 s ≈ 22 min 30 s** |
| FLOPS totali | ≈ **3.87 PFLOPS** | ≈ **3.87 PFLOPS** |
| **Energia totale** | ~1.209 kJ ≈ **0.34 kWh** | ~250 kJ ≈ **0.07 kWh** |
| **Costo energetico** ³ | ~€ 0.10 | ~€ 0.02 |

#### Inferenza (singola immagine)

| | **CPU** (i9-14900HX) | **GPU** (RTX 4080 Laptop) |
|---|---|---|
| FLOPS per immagine | **2.26 GFLOPS** | **2.26 GFLOPS** |
| Tempo stimato | ~50–200 ms ⁴ | ~5–20 ms ⁴ |
| Potenza durante inferenza | ~80 W ⁵ | ~120 W ⁵ |
| **Energia per immagine** | ~80 W × 0.1 s ≈ **8 J** | ~120 W × 0.01 s ≈ **1.2 J** |

#### Confronto visivo: energia per 150 epoche di training

```
CPU  ████████████████████████████████████████████████  1.68 kWh ≈ € 0.50
GPU  █████████▌                                        0.35 kWh ≈ € 0.10
     ─────────┼─────────┼─────────┼─────────┼─────────
             0.5       1.0       1.5       2.0 kWh
```

> **La GPU è ~5.8× più veloce e ~4.8× più efficiente energeticamente** per il training.

#### Note

¹ CPU ~120 W (sotto carico sostenuto) + ~5 W (GPU idle) + ~30 W (sistema: RAM, SSD, display)
² CPU ~35 W (carico moderato) + ~120 W (GPU sotto carico) + ~30 W (sistema)
³ Costo energia stimato a €0.30/kWh (media Italia 2025-2026)
⁴ Il tempo effettivo di inferenza include overhead di Python/TF; i tempi puri di calcolo sono inferiori
⁵ Potenza inferiore rispetto al training perché l'inferenza è un singolo forward pass molto breve, senza carico sostenuto

---

## ❓ Problemi comuni

| Problema | Soluzione |
|---|---|
| `pip install tensorflow` fallisce | Usa Python 3.11 o 3.12, non 3.13 |
| `ModuleNotFoundError: No module named 'tensorflow'` | Attiva il venv: `.\.venv\Scripts\Activate.ps1` |
| `Modello non trovato` quando avvii `app.py` | Esegui prima `python train.py` |
| Training molto lento su CPU | Usa WSL2 con GPU: `bash train_gpu.sh` |
| GPU non rilevata in WSL2 | Verifica driver NVIDIA Windows e che WSL2 sia versione 2 (`wsl --list -v`) |


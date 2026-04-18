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

- **Python 3.10, 3.11 o 3.12** (consigliato **3.11**). Python 3.13 funziona ma può dare problemi con alcune librerie.
- **Git** (per scaricare il progetto da GitHub)
- Il dataset **PetImages** con le sottocartelle `Cat/` e `Dog/` (vedi sezione precedente)
- Connessione a Internet (per scaricare le dipendenze)

---

## 🚀 Installazione passo-passo

> 🧑‍🎓 **Per chi non ha mai usato un terminale**: niente panico! Segui i passaggi uno per uno, copia e incolla i comandi, e tutto funzionerà. Se qualcosa va storto, leggi il messaggio di errore e controlla la sezione "Problemi comuni" in fondo.

---

### Passo 0 — Installa Python e Git

Prima di tutto servono due programmi: **Python** (il linguaggio di programmazione) e **Git** (per scaricare il progetto).

<details>
<summary><strong>🪟 Windows</strong></summary>

1. **Installa Python 3.11**
   - Vai su 👉 https://www.python.org/downloads/release/python-3119/
   - Scorri in fondo alla pagina e clicca su **"Windows installer (64-bit)"**
   - Apri il file scaricato
   - ⚠️ **IMPORTANTISSIMO**: nella prima schermata, metti la spunta su **"Add Python 3.11 to PATH"** (in basso)
   - Clicca "Install Now" e attendi la fine

2. **Installa Git**
   - Vai su 👉 https://git-scm.com/download/win
   - Il download parte automaticamente. Apri il file e clicca "Next" su tutto, lasciando le opzioni predefinite.

</details>

<details>
<summary><strong>🍎 macOS</strong></summary>

1. **Installa Python 3.11**
   - Vai su 👉 https://www.python.org/downloads/release/python-3119/
   - Scarica il **"macOS 64-bit universal2 installer"**
   - Apri il file `.pkg` e segui le istruzioni

2. **Installa Git**
   - Apri il Terminale (vedi sotto come fare) e scrivi:
     ```bash
     git --version
     ```
   - Se non è installato, macOS ti chiederà automaticamente di installare gli "Xcode Command Line Tools". Accetta e attendi.

</details>

<details>
<summary><strong>🐧 Linux (Ubuntu/Debian)</strong></summary>

Apri il terminale e digita:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip git -y
```

> Se `python3.11` non è disponibile, aggiungi prima il repository:
> ```bash
> sudo add-apt-repository ppa:deadsnakes/ppa -y
> sudo apt update
> sudo apt install python3.11 python3.11-venv -y
> ```

</details>

---

### Passo 1 — Apri il terminale (prompt dei comandi)

Il **terminale** è una finestra dove si scrivono comandi testuali al computer. Ogni sistema operativo ne ha uno.

<details>
<summary><strong>🪟 Windows — Come aprire PowerShell</strong></summary>

Hai diverse opzioni:

**Opzione A (la più semplice):**
1. Premi i tasti **`Windows` + `R`** insieme (si apre una piccola finestra "Esegui")
2. Scrivi `powershell` e premi **Invio**

**Opzione B:**
1. Clicca sul pulsante **Start** (icona Windows in basso a sinistra)
2. Scrivi `PowerShell`
3. Clicca su **"Windows PowerShell"** (NON scegliere "come amministratore" a meno che non serva)

**Opzione C (da Esplora File):**
1. Apri la cartella del progetto in Esplora File
2. Clicca sulla barra dell'indirizzo in alto (dove c'è il percorso della cartella)
3. Scrivi `powershell` e premi Invio → si apre un terminale già nella cartella giusta!

Vedrai una finestra blu/nera con un testo tipo:
```
PS C:\Users\TuoNome>
```
Questo è il **prompt**: il computer aspetta i tuoi comandi. Il testo prima di `>` indica la cartella in cui ti trovi.

</details>

<details>
<summary><strong>🍎 macOS — Come aprire il Terminale</strong></summary>

1. Premi **`Cmd` + `Spazio`** (si apre Spotlight)
2. Scrivi `Terminale` (o `Terminal`)
3. Premi **Invio**

Vedrai una finestra con un testo tipo:
```
tuonome@MacBook ~ %
```

</details>

<details>
<summary><strong>🐧 Linux — Come aprire il Terminale</strong></summary>

- Premi **`Ctrl` + `Alt` + `T`** (funziona su Ubuntu e molte distribuzioni)
- Oppure cerca "Terminale" nel menu delle applicazioni

</details>

---

### Passo 2 — Scarica il progetto da GitHub

Il progetto è ospitato su GitHub. Per scaricarlo, scrivi nel terminale:

```bash
git clone https://github.com/mariomolinara/rummo_cats_dogs.git
```

**Alternativa senza Git** (download manuale):
1. Vai sulla pagina GitHub del progetto nel browser
2. Clicca il pulsante verde **"Code"** → **"Download ZIP"**
3. Estrai il file ZIP dove preferisci (es. sul Desktop)

---

### Passo 3 — Entra nella cartella del progetto

Ora devi dire al terminale di "entrare" nella cartella del progetto. Il comando è `cd` (change directory).

**Ma come trovo il percorso della cartella?**

<details>
<summary><strong>🪟 Windows — Come trovare il percorso di una cartella</strong></summary>

1. Apri **Esplora File** e naviga fino alla cartella `rummo_cats_dogs`
2. Clicca sulla **barra dell'indirizzo** in alto (dove vedi il percorso tipo `Questo PC > Desktop > ...`)
3. Il percorso diventa selezionabile, ad esempio: `C:\Users\TuoNome\Desktop\rummo_cats_dogs`
4. **Copialo** con `Ctrl+C`

Poi nel terminale scrivi:

```powershell
cd "C:\Users\TuoNome\Desktop\rummo_cats_dogs"
```

> 💡 Puoi incollare nel terminale con il **tasto destro del mouse** oppure `Ctrl+V`.

</details>

<details>
<summary><strong>🍎 macOS — Come trovare il percorso di una cartella</strong></summary>

1. Apri il **Finder** e naviga fino alla cartella `rummo_cats_dogs`
2. Fai **clic destro** sulla cartella → tieni premuto il tasto **`Option` (⌥)** → apparirà la voce **"Copia … come nome file"** (Copy as Pathname)
3. Questo copia il percorso completo, ad esempio: `/Users/TuoNome/Desktop/rummo_cats_dogs`

Poi nel terminale scrivi:

```bash
cd /Users/TuoNome/Desktop/rummo_cats_dogs
```

</details>

<details>
<summary><strong>🐧 Linux — Come trovare il percorso di una cartella</strong></summary>

1. Apri il **File Manager** e naviga fino alla cartella
2. Nella barra dell'indirizzo vedrai il percorso, ad esempio: `/home/tuonome/Desktop/rummo_cats_dogs`

Poi nel terminale scrivi:

```bash
cd /home/tuonome/Desktop/rummo_cats_dogs
```

> 💡 Trucco: scrivi `cd ` (con lo spazio) e poi **trascina la cartella** dal file manager dentro il terminale. Il percorso si incolla automaticamente!

</details>

---

### Passo 4 — Verifica che Python sia installato

Controlliamo che Python funzioni:

**🪟 Windows:**
```powershell
python --version
```

**🍎 macOS / 🐧 Linux:**
```bash
python3 --version
```

Dovresti vedere qualcosa come `Python 3.11.9`. Se vedi un errore, Python non è installato correttamente (torna al Passo 0).

---

### Passo 5 — Crea un ambiente virtuale (venv)

Un **ambiente virtuale** è una "scatola isolata" dove installare le librerie del progetto senza interferire con il resto del computer. È una best practice in Python.

**🪟 Windows:**
```powershell
python -m venv .venv
```

> Se hai più versioni di Python e vuoi usare la 3.11, scrivi:
> ```powershell
> py -3.11 -m venv .venv
> ```

**🍎 macOS / 🐧 Linux:**
```bash
python3.11 -m venv .venv
```

> Se `python3.11` non funziona, prova `python3 -m venv .venv`.

Questo crea una cartella nascosta `.venv` dentro il progetto. Non toccarla manualmente.

---

### Passo 6 — Attiva l'ambiente virtuale

Ogni volta che apri un nuovo terminale per lavorare sul progetto, devi "attivare" il venv.

**🪟 Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

> ⚠️ **Errore comune**: se appare un messaggio rosso tipo *"l'esecuzione di script è disabilitata"*, esegui prima questo comando (una volta sola):
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Poi riprova il comando di attivazione.

**🪟 Windows (Prompt dei comandi classico, cmd):**
```cmd
.venv\Scripts\activate.bat
```

**🍎 macOS / 🐧 Linux:**
```bash
source .venv/bin/activate
```

**Come capisco che il venv è attivo?** Vedrai `(.venv)` all'inizio della riga del terminale:
```
(.venv) PS C:\Users\TuoNome\Desktop\rummo_cats_dogs>
```

---

### Passo 7 — Installa le dipendenze

Ora installiamo tutte le librerie necessarie (TensorFlow, Flask, Pillow, ecc.) con un solo comando:

```bash
pip install -r requirements.txt
```

> ⏳ Questo comando scarica circa 1-2 GB di dati. Ci vogliono alcuni minuti a seconda della velocità della connessione.

Se tutto va bene, vedrai varie righe di progresso e alla fine `Successfully installed ...`.

> ⚠️ Se ricevi errori su TensorFlow e stai usando Python 3.13, ricrea il venv con Python 3.11 (torna al Passo 5).

---

### ✅ Installazione completata!

Se sei arrivato qui senza errori, il progetto è pronto. Ora puoi:
- **Addestrare la rete** → vai alla sezione "Training del modello"
- **Usare la web app** → vai alla sezione "Avvio della Web App" (serve prima il training)

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


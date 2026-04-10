## Plan: AlexNet Training + Web App di Classificazione Cani/Gatti

Creare due script Python: `train.py` per addestrare un modello AlexNet con TensorFlow sul dataset PetImages (Cat/Dog), e `app.py` come web app Flask per classificare immagini caricate dall'utente tramite una form, mostrando i risultati in una tabella cumulativa con persistenza su file JSON.

---

### Step 0 — Compatibilità Python / TensorFlow

TensorFlow potrebbe non supportare Python 3.13. Prima di tutto verificare se `pip install tensorflow` va a buon fine nel venv corrente (Python 3.13.7).

- **Se l'installazione riesce**: procedere normalmente.
- **Se l'installazione fallisce**: eseguire il downgrade creando un nuovo venv con Python 3.10–3.12:
  1. Installare Python 3.11 (o 3.12) dal sito ufficiale python.org.
  2. Ricreare il venv:
     ```powershell
     Remove-Item -Recurse -Force .venv
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
  3. Proseguire con lo Step 1.

---

### Steps

1. **Creare [requirements.txt](requirements.txt)** con le dipendenze: `tensorflow`, `matplotlib`, `flask`, `Pillow`, `scikit-learn`. Installarle nel venv con `pip install -r requirements.txt`.

2. **Creare [train.py](train.py)** con la seguente logica:
   - **Pulizia dataset**: scansionare `PetImages/Cat` e `PetImages/Dog`, rimuovere immagini corrotte (non apribili con Pillow) per evitare errori in training.
   - **Split del dataset**: usare `tf.keras.utils.image_dataset_from_directory` per caricare le immagini (ridimensionate a 227×227), poi dividere in **training (70%)**, **validation (20%)** e **test (10%)** tramite `tf.data.Dataset.take/skip`.
   - **Architettura AlexNet**: definire il modello con `tf.keras.Sequential` — 5 blocchi convoluzionali (Conv2D + BatchNorm + MaxPool) con filtri crescenti (96→256→384→384→256), seguiti da Flatten, 3 Dense (4096, 4096, 1 con sigmoid). Input shape `(227, 227, 3)`.
   - **Compilazione e training**: ottimizzatore Adam, loss `binary_crossentropy`, metrica `accuracy`. Usare `EarlyStopping` monitorando **`val_loss`** con `patience=30` e `restore_best_weights=True`, massimo 30 epoche.
   - **Grafico**: al termine del training, generare con `matplotlib` un grafico loss su training e validation, salvato come `training_history.png`.
   - **Valutazione su test set**: stampare accuracy e loss sul test set.
   - **Salvataggio modello**: salvare il modello finale in formato `.keras` (es. `alexnet_cats_dogs.keras`).

3. **Creare [app.py](app.py)** come web app Flask:
   - Caricare il modello salvato `alexnet_cats_dogs.keras` all'avvio.
   - **Persistenza su file**: lo storico delle classificazioni viene salvato in `classification_history.json`. Al riavvio del server i risultati precedenti vengono ricaricati automaticamente.
   - **Route GET `/`**: renderizzare un template HTML con una form di upload file e sotto una tabella con lo storico delle classificazioni letto dal file JSON.
   - **Route POST `/predict`**: ricevere l'immagine, salvarla in `static/uploads/`, preprocessarla (resize 227×227, normalizzazione /255), eseguire `model.predict`, calcolare classe (`Cat`/`Dog`), confidenza (%), tempo di inferenza. Aggiungere il risultato al file JSON e redirigere a `/`.
   - **Route POST `/clear`**: svuotare lo storico delle classificazioni (cancellare il file JSON e le immagini caricate).
   - **Dettagli tecnici mostrati per ogni immagine nella tabella**: thumbnail, nome file, classe predetta, confidenza %, probabilità raw, dimensione originale dell'immagine, tempo di inferenza in ms.

4. **Creare [templates/index.html](templates/index.html)** con template Jinja2:
   - Form di upload (enctype multipart/form-data) nella parte superiore.
   - Tabella HTML sotto la form con colonne: #, Anteprima, Nome File, Predizione, Confidenza, Probabilità Raw, Dimensioni Originali, Tempo Inferenza (ms). Stile con colori verde/rosso per feedback visivo.
   - Pulsante "Cancella storico" per azzerare la tabella.

5. **Creare la cartella `static/uploads/`** per le immagini caricate dall'utente.

6. **Creare [README.md](README.md)** con istruzioni semplici e chiare per utenti non esperti:
   - Come attivare il venv (e come ricrearlo se Python 3.13 non è compatibile con TensorFlow).
   - Come installare le dipendenze (`pip install -r requirements.txt`).
   - Come eseguire il training (`python train.py`).
   - Come avviare la web app (`python app.py`) e aprire il browser su `http://localhost:5000`.

---

### Decisioni prese

| Domanda | Decisione |
|---|---|
| Python 3.13 e TensorFlow | Lo Step 0 prevede un piano di fallback con downgrade a Python 3.11 |
| EarlyStopping: cosa monitorare | Solo `val_loss` con `patience=30` |
| Persistenza risultati web app | File JSON (`classification_history.json`), ricaricato al riavvio del server |


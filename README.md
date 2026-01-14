# Obiettivo progetto:
Questo progetto rientra nell'ambito dello sviluppo di tecnologie assistive per persone ipovedenti o non vedenti. L'obiettivo è la realizzazione di un modulo di Computer Vision per dispositivi wearable in grado di rilevare ostacoli ambientali e fornire feedback aptico in tempo reale.

La sfida ingegneristica principale risiede nel **trade-off tra accuratezza e latenza** su hardware con risorse limitate (Edge Computing). Questa piattaforma di benchmarking è stata sviluppata per confrontare empiricamente due approcci architetturali distinti:

1.  **ResNet50 (DeepLabV3) - Segmentazione Semantica:** Un approccio "denso" che analizza ogni pixel, offrendo alta precisione spaziale ma elevati costi computazionali.
2.  **EfficientNet-B0 - Classificazione Multi-Label:** Un approccio "leggero" ottimizzato per l'inferenza rapida, che sacrifica la precisione spaziale a favore della velocità.

La piattaforma simula l'ambiente di produzione misurando metriche critiche come **Latenza (ms)**, **Consumo RAM (MB)** e **Throughput (FPS)**.

## Struttura del progetto
Il codice è organizzato in modo tale da separare l'interfaccia utente dalla logica di benchmarking.

La cartella **/modules** presenta due file, **adapters e platform_core**.

Il ruolo di **adapters** è di implementare il Design Pattern Adapter.
Questo modulo funge da **"ponte"** tra il motore di benchmark e i vari modelli di Deep Learning. Poiché ResNet (Segmentazione) e EfficientNet (Classificazione) hanno **input e output** strutturalmente diversi, gli Adapters uniformano le loro interfacce esponendo un unico metodo standard **predict()**.
Le funzioni principali sono di gestione del pre-processing specifico (Resize 512x1024, Normalizzazione ImageNet), gestione del caricamento pesi con gestione errori (strict=False) e calcolo delle metriche custom (es. Mean Region Confidence per la segmentazione).

Per quanto riguarda **platform_core** è il motore di Benchmarking ed ha il compito di monitoraggio RAM a livello di Sistema Operativo tramite libreria psutil, misurazione della latenza di inferenza (ms) sincronizzata (CPU/GPU) e logging dei risultati su file CSV persistenti per analisi successive.

**/checkpoints:** è la cartella che contiene i pesi dei due modelli.

**/dataset_test:** contiene un piccolo dataset di immagini per testare i modelli.

Il file **app** è il frontend della piattaforma di bench, funge da pannello di controllo dei dati in output. Tra le varie funzioni abbiamo quella di visualizzatore dei **KPI(Latenza, Speedup e Throughput)** e grafici comparativi.
Effettua un rendering visivo delle maschere di segmentazione e delle classi predette, inoltre invoca il file **run_thesis** tramite chiamate di sistema **(subprocessing)** per garantire la stabilità della memoria.


Il file **run_thesis** è l'esecutore isolato **(Process Isolation)**, è progettato per eseguire il benchmark di un singolo modello e terminare immediatamente poichè python **non rilascia immediatamente** la memoria RAM al termine di funzioni pesanti. Con questo garantiamo che i valori misurati per il secondo modelo (EfficientNet) non venga influenzato il parametro della RAM dovuto all'esecuzione precedente di ResNet.


## Come avviare il progetto?
Una volta scaricata la cartella in locale, bisogna configurare l'ambiente di esecuzione creando innanzitutto un ambiente virtuale.

### 1. Creazione Environment (Virtual Env)
Apri il terminale all'interno della cartella del progetto e crea un ambiente virtuale per isolare le dipendenze:

* **Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
* **Mac/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 2. Installazione Dipendenze
Con l'ambiente virtuale attivo (vedrai `(venv)` apparire nel terminale), installa automaticamente tutte le librerie necessarie tramite il file incluso:

```bash
pip install -r requirements.txt
```

### 3. Caricamento pesi
Inserire i file .pth all'interno della cartella **/checkpoints**.

### 4. Avviamento applicazione
Nel terminale avviare l'applicazione con il comando:

```bash
streamlit run app.py
```
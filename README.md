Prima di eseguire il progetto, assicurarsi di installare le dipendenze contenute nel file `requirements.txt`. 
È possibile farlo eseguendo il seguente comando nel terminale:

```
pip install -r requirements.txt
```

Dopo aver fatto ciò, si procede con l'installazione del dataset coco2017. 
Come potete notare, il dataset non è incluso nel progetto a causa delle sue dimensioni elevate. Per questo, sarà necessario
scaricarlo grazie ai link contenuti nel file 'coco.yaml' nella cartella 'datasets'.
Per scaricarlo, procedete con l'esecuzione dello script 'download_datasets.py' per il download delle immagini, runnando il seguente comando:

```
python -m scripts.download_datasets
```
Successivamente è necessario scaricare le labels tramite lo script 'download_labels.py', eseguendo il comando:

```
python -m scripts.download_labels
```

Dopo aver fatto ciò, inizia il divertimento! 

Per ottimizzare i modelli, ed eseguire i vari tipi di quantizzazione, eseguite lo script 'optimize.py', 
contenuto nella cartella 'scripts'.

```
python -m scripts.optimize
```

Questo restituirà vari modelli quantizzati, li vedrete nella cartella 'model'

Dopo aver fatto ciò, si passa alla fase di valutazione, tramite l'esecuzione dello script 'benchmark.py',
contenuto nella cartella 'scripts'. Potete selezionare il tipo di modello da valutare modificando il flag 'optimized' nel metodo main()
del file 'benchmark.py'.

```
python -m scripts.benchmark
```

Un paio di parole sulle tecniche di quantizzazione implementate (ricordiamo che vengnono eseguite sul modello convertito in ONNX):
- **Dynamic Quantization**: Questa tecnica quantizza solo i pesi del modello durante l'inferenza, 
    mantenendo le attivazioni in precisione a 32 bit;
- **Static Quantization**: Questa tecnica quantizza sia i pesi che le attivazioni del modello, ma solo di quelle esplicitate dal parametro
  op_types_to_quantize. Richiede una fase di calibrazione in cui il modello viene eseguito su un set di dati di calibrazione per raccogliere statistiche sulle attivazioni.

Queste sono quantizzazioni INT8, che riducono significativamente la dimensione del modello (E DELLO STORAGE!!!) e migliorano le prestazioni di inferenza.
La quantizzazione FP16 viene eseguita esportando il modello ONNX in formato FP16, riducendo la precisione dei pesi e delle attivazioni a 16 bit floating point.

Tutte i metodi che gestiscono le quantizzazioni e le ottimizzazioni (inclusi gli export in ONNX), sono contenuti nel 
file 'optimizer.py', all'interno del pacchetto 'src'. E' UN FILE DIVERSO DA OPTIMIZE.PY, CHE E' UNO SCRIPT CHE RICHIAMA LE FUNZIONI CONTENUTE IN QUESTO FILE.

Per un'esecuzione live del modello tramite la webcam del pc (figata), eseguite il file 'live_demo.py' in 'scripts'.
Selezionate il modello modificando il flag 'model' nel metodo main()

```
python -m scripts.live_demo
```

Nel pacchetto 'src' trovate il file 'patterns.py', che contiene i pattern di progettazione utilizzati nel progetto.
Per ora tralasciate 'pruning_utils.py'

Buon divertimento!

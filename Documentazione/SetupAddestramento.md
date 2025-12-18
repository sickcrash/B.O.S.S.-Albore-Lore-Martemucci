Per poter effettuare l'addestramento del modello, organizzare il dataset (composto da gtFine e leftImg8bit) in 2 cartelle separate con i rispettivi nomi: gtFine e leftImg8bit.
Successivamente avviare runnare il progetto mediante il comando: python main.py --root D:\\Datasets\\Cityscapes --epochs 50 --batch-size 4

N.B.: dovete modificare la root in base alla posizione delle cartelle del dataset (nel mio caso in D:\\Datasets\\Cityscapes sono situate le 2 cartelle citate sopra); il valore assegnato a --epochs è indicativo, io ho effettuato l'addestramento su 10 epoche (dato che richiede abbastanza tempo, soprattutto se non avete una macchina prestante); batch-size potete lasciarlo a 4 (indica quante immagini vengono prese alla volta).


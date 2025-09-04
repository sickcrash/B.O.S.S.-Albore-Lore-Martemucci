#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Addestramento del modello efficientNet con più dataset in contemporanea

Documentazione per l'avvio

Argomenti:
  --data_roots     indica il percorso per il dataset
  --epochs         indice quante il sistema legge il dataset (più volte lo legge, più il sistema legge più apprende e quindi maggiore accuratezza
                   solo che non deve leggere troppe volte il dataset altrimenti il sistema impara a memoria e le prestazioni iniziano a peggiorare)
  --batch_size     indica quante immagini contemporaneamente si possono analizzare prima di aggiornare i pesi della rete neurale, se il numero è 
                   è troppo basso (es 1) significa che ogni immagine che controlla va ad aggiornare i pesi, il che significa che è molto
                   accurato però è troppo lento, invece se fosse troppo alto (es tutto il dataset) sarebbe più veloce però c'è maggiore possibilità di errore
  --arch           indica la versione del modello del che si utilizza, in genere più è alto l'hardware del pc, maggiore è la versione che si può scegliere
                   io avendo una scheda grafica integrata nel processore ho optato per la prima versione del modello (b0) è un po' più lento
                   e meno accurato ovviamente
"""


import os #fornisce funzioni per interagire con il sistema operativo
import argparse #serve per gestire gli argomenti forniti dalla linea di comando
import time # fornisce funzioni relative al tempo
import json #permette di lavorare con dati in formato json
import shutil #permette di eseguire delle operazioni su file e cartelle
from typing import Tuple, Optional, List #importa vari tipi di dati che possono essere utilizzati nel codice (optional perché si può restituire anche None, la tupla per raggruppare dati eterogenei tra di loro)


os.environ["OMP_NUM_THREADS"] = str(os.cpu_count()) #indicando il numero ti thread che le librerie parallele possono utilizzare (infatti os.cpu_count() indica quanto core sono disponibili al momento) questo viene fatto per non sovraccaricare molto la cpu, in particolare in base ai core che stanno vengono utilizzati un tot di thread
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0" #granularity=fine significa che un thread viene associato a un core e non a un gruppo di core (quindi c'è una corrispondenza 1 a 1), compact significa che i thread vicini vengono eseguiti da core vicini
os.environ["KMP_BLOCKTIME"] = "1" #serve per indicare quando un thread diventa inattivo se non ha lavoro da svolgere



import numpy as np #serve per eseguire operazioni vettoriali in modo molto veloce (più l'utilizzo dell'alias)
import pandas as pd #serve per l'analisi e la manipolazione dei dati
from tqdm import tqdm #serve per visualizzare le barre di avanzamento quando è in esecuzione il progetto

import torch
import torch.nn as nn #modulo per le reti neurali
from torch.utils.data import DataLoader
from torchvision import datasets, transforms #servono per gli elementi di computer vision (quindi dataset e transforms per il processing delle immagini)

import timm #contiene dei modelli pre-addestrati di computer vision
from sklearn.metrics import classification_report, confusion_matrix #serve per generare un report sulla valutazione delle metriche

torch.set_num_threads(os.cpu_count()) #è una configurazione di pytorch affinché possa utilizzare tutti i core della cpu


def set_seed(seed: int = 42): #ci sono molti processi stocastici nella fase di addestramento di una rete neurale (come per esempio l'inizializzazione dei pesi ecc)in genre fissiamo il seme a un numero (generalmente 42) in modo che ad ogni esecuzione dell'addestramento otteniamo sempre gli stessi risultati
    import random
    random.seed(seed) #fissiamo il seed che passiamo tra parametri
    np.random.seed(seed) #viene fissato il seed per le operazioni numeriche
    torch.manual_seed(seed) #idem anche per il pytorch
    torch.cuda.manual_seed_all(seed) #fissa il seed per le gpu se sono presenti
    torch.backends.cudnn.deterministic = True #stiamo forzando ad utilizzare operazioni deterministiche (ci sono alcune operazioni di default che non lo sono)
    torch.backends.cudnn.benchmark = False #il benchmark introduce operazioni di ottimizzazione della cpu però ci possono essere delle variabilità nei risultati perciò l'ho disabilitato


#questa funzione si occupa di contare le classi presenti nel dataset
def count_classes_from_train(root: str) -> Tuple[int, List[str]]:
    """Conta le classi con debug integrato"""
    train_dir = os.path.join(root, 'train') #viene unito il percorso principale della cartella con train
    if not os.path.isdir(train_dir):   #se non esiste questa cartella viene stampato un errore
        raise FileNotFoundError(f"Cartella 'train' non trovata in {root}")
    
    
    print(f"\nAnalisi struttura in: {train_dir}")
    #viene creato una lista di tutte le sottocartelle della cartella train quindi di tutte le classi (in ordine alfabetico)
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if len(classes) == 0: #se non ci sono sottocartelle mostriamo i primi 10 elementi (semplicemente per questioni di debug) + stampa di un messaggio di errore
        print(f"Contenuto della cartella: {os.listdir(train_dir)[:10]}...")
        raise RuntimeError("Nessuna sottocartella di classe trovata")
    
    print(f"Trovate {len(classes)} classi. Prime 3: {classes[:3]}") #stampiamo i numeri delle classi trovate e i nomi delle prime 3 (ma serve solo per verifica perché potremmo anche non farlo)
    sample_files = os.listdir(os.path.join(train_dir, classes[0]))[:3]
    print(f"File di esempio in '{classes[0]}': {sample_files}")
    
    return len(classes), classes #viene restituito sia il numero che i nomi delle classi

#crea due set di immagini, uno destinato alla fase di addestramento e uno dedicato alla fase di valutazione
def build_transforms(img_size: int = 128): 
    """Transform ottimizzati per CPU"""
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size), #viene ritagliata casualmente una parte di immagine (128 px) quindi viene aumentata la variabilità dei dati
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), #vieneritagliata una parte più grande dell'immagine per mantenere più informazioni prima dell'effettiva presa in considerazione del 128x128
        transforms.CenterCrop(img_size), #e successivamente si occupa di fare il giusto ritaglio
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms

#crea 3 dataloader(train, val e test)
def build_dataloaders(root: str, img_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """DataLoader ottimizzati per CPU"""
    train_tfms, eval_tfms = build_transforms(img_size)
    
   
    def find_split_dir(base_dir, split_name):
        for dirpath, dirnames, _ in os.walk(base_dir): #cerca ricorsivamente nella cartella
            if split_name.lower() in dirpath.lower(): #controlla se train, test e val sono presenti nel percorso
                return dirpath
        return os.path.join(base_dir, split_name)  #se non trova nulla, usa il percorso predefinito

#viene fatta una ricerca in train test e val e poi viene creato un dataset pytorch
    train_dir = find_split_dir(root, 'train')
    val_dir = find_split_dir(root, 'val')
    test_dir = find_split_dir(root, 'test')

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    
    num_cpu_workers = min(4, os.cpu_count())  #anche se ci sono più core massimo 4 ne vengono utilizzati per non sovraccaricare il sistema
    loader_args = {
        'batch_size': batch_size, #quante immagini contemporaneamente si devono analizzare
        'num_workers': num_cpu_workers,
        'pin_memory': False,  #è un'ottimizzazione disabilitata per la cpu
        'persistent_workers': True if num_cpu_workers > 0 else False
    }
  
    #viene creato il dataLoader (shuffle=true significa che ogni volta che la rete legge il dataset i dati vengono mescolati)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    
    test_loader = None
    if os.path.isdir(test_dir):
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    #restituisce i 3 dataloader + le classi
    return train_loader, val_loader, test_loader, train_ds.classes


#crea un modello di deeplearning preconfigurato
def create_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
   #è un modello ottimizzato nel caso si utilizza la cpu in cui vengono caricati i pesi preimpostati e vengono modificati i nomi delle classi

    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    if not torch.cuda.is_available(): #qui viene controllato il mancato utilizzo della gpu, allora forza il sistema a lavorare in float32 in quanto le cpu lavorano meglio in questo modo
        model.float()  
    
    return model

#verifica quante volte il modello è accurato, cioè da esito positivo calcolandone la percentuale
def accuracy_top1(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item() 
    return correct / targets.size(0) 

#serve per calcolare tutto ciò che viene riportato in output (l'accuratezza e quanto sbaglia)
def evaluate(model, loader, device, criterion=None):
    model.eval() #preparazione del modello pronto per poter essere "valutato"
    #inizializzazione dei contatori
    total_loss = 0.0 #quanto sbaglia in totale
    total_correct = 0 #quante risposte giuste
    total_samples = 0 #quante domande in totale

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader: #consideriamo un gruppo di immagini e le risposte corrette
            inputs = inputs.to(device) #vengono spostati i dati sul dispositivo (in questo caso sulla cpu)
            labels = labels.to(device)

            outputs = model(inputs) #il modello sta dando delle risposte dopo aver visto delle immagini
            if criterion is not None: #serve per capire quanto la risposta sia errata all'interno del modello
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

            total_correct += (outputs.argmax(1) == labels).sum().item() #vengono contate le risposte corrette
            total_samples += inputs.size(0) #le immagini che ha visto in totale

            all_preds.append(outputs.argmax(1).cpu().numpy()) #vengono registrate tutte le "risposte" corrette  e quelle che ha dato in totale per valutare le risposte
            all_labels.append(labels.cpu().numpy())

    avg_loss = (total_loss / total_samples) if criterion is not None else None #quando si sbaglia in media per ogni domanda
    avg_acc = total_correct / total_samples #percentuale di risposte corrette
    all_preds = np.concatenate(all_preds) #rappresentano tutte le risposte date
    all_labels = np.concatenate(all_labels) #rappresentano tutte le risposte corrette
    return avg_loss, avg_acc, all_preds, all_labels


def train_one_dataset(args, data_root: str, device: torch.device, img_size: int):
    # viene creata una cartella per salvare i risultati dell'addestramento
    ds_name = os.path.basename(os.path.normpath(data_root))
    out_dir = os.path.join(args.out_dir, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    # conta quante categorie ci sonoe c'è la divisione nei 3 gruppi cioè training, validation e test
    num_classes, class_names = count_classes_from_train(data_root)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_root, img_size, args.batch_size, args.num_workers
    )

    # creazione del modello: architettura (efficientNet), utilizzo dei pesi pre-addestrati e spostamento del modello su cpu/gpu
    model = create_model(args.arch, num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    # viene bloccato la parte principale del modello, in modo che tutto quello che già sa continua a saperlo...modifica solo la parte "nuova"
    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False #viene disabilitato l'aggiornamento per tutti i parametri del modello (quindi pytorch non calcola i gradienti per questi parametri)
        try: #viene provato a sbloccare solo lo strato finale (e questo è possibile in 3 modi)
            for p in model.get_classifier().parameters(): #questo rappresenta il primo tentativo (con il metodo standard get_classifier())
                p.requires_grad = True
        except Exception:
            if hasattr(model, "classifier"): #secondo metodo si cerca un attributo chiamato classifier (tipico dei modelli cnn)
                for p in model.classifier.parameters():
                    p.requires_grad = True
            elif hasattr(model, "head"): #terzo metodo, ricerca di un attributo chiamato head (questo viene utilizzato per i modelli più moderni)
                for p in model.head.parameters():
                    p.requires_grad = True
    #configurazione dell'addestramento
    criterion = nn.CrossEntropyLoss() #viene creata la funzione che misura quanto sbaglia il modello
    trainable_params = [p for p in model.parameters() if p.requires_grad] #creazione della lista contenente solo i parametri che devono essere aggiornati (vengono presi tutti i parametri e vengono filtrati solo quelli contrassegnati come addestrabili)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # viene creato un file csv per registrare i progressi dell'addestramento
    metrics_path = os.path.join(out_dir, "metrics_epoch.csv")#creazione del percorso del file di log
    with open(metrics_path, "w", encoding="utf-8") as f: #crea un file in modalità scrittura, supporto dei caratteri speciali e gestione automatica del file dopo l'uso
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n") #scrive l'intestazione delle colonne

    best_val_acc = -1.0 #inizializzazione della variabile per tenere traccia della migliore accuratezza di valutazione raggiunta (-1 è un valore impossibile per il modello quindi il primo valore che andiamo ad analizzare è considerato il migliore)
    best_model_path = os.path.join(out_dir, "best_model.pth") #viene creato il percorso dove verrà salvato il miglior modello

    for epoch in range(1, args.epochs + 1): #serve per fare un giro completo di tutto il dataset
        model.train() #preparazione del modello per calcolare i gradienti
        running_loss = 0.0 #accumulazione della loss totale di dopo il controllo di tutto il dataset
        running_correct = 0 #contatore delle predizioni corrette
        total = 0 #numeri degli esempi che sono stati processati

        pbar = tqdm(train_loader, desc=f"[{ds_name}] Epoch {epoch}/{args.epochs}", ncols=100) #barra di avanzamento per ogni epoca
        for inputs, labels in pbar: #caricamento del batch di dati
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() #grandieti vengono azzerati in quanto pytorch accumula gradienti per default
            outputs = model(inputs) #le immagini passano attraverso tutti gli strati della rete
            loss = criterion(outputs, labels) #calcolo della loss
            loss.backward() #calcolare il gradiente della loss rispetto a tutti i parametri
            optimizer.step() #utilizzo dei gradienti calcolati per aggiornare i pesi della rete

            running_loss += loss.item() * inputs.size(0) #viene convertito la loss da tensore a numero e successivamente viene normalizzato
            running_correct += (outputs.argmax(1) == labels).sum().item() #viene trovata la classe predetta
            total += inputs.size(0)

            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{running_correct/total:.4f}") #visualizzazione in tempo reale della loss media e dell'accuratezza

        train_loss = running_loss / total #calcolo della loss media
        train_acc = running_correct / total #calcolo dell'accuratezza media

        # valutazione del modello sui dati non utilizzati per l'addestramento per capire la capacità di generalizzazione che il modello ha
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion=criterion)

        # registrazione dei risultati ottenuti
        current_lr = optimizer.param_groups[0]['lr']
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{current_lr:.8f}\n")

        # salviamo il modello che ha l'accuratezza migliore valida fino a un determinato momento
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        # salvataggio del learning rate (cioè la velocitià con la quale vengono salvati i pesi)
        scheduler.step()

    # viene caricato il file con il modello precedentemente salvato e ricostruisce l'architettura del modello con i pesi ottimali salvati
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # valutazione sul validation set
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device, criterion=criterion)

    #riepilogo di tutto ciò che serve per l'addestramento in formato json
    summary = {
        "dataset": ds_name,
        "arch": args.arch,
        "img_size": img_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_acc": float(val_acc),
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    if test_loader is not None: #verifica se è presente il test per l'addestramento
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device, criterion=criterion) #valutazione sul test set
        summary["test_acc"] = float(test_acc) #aggiuntiamo l'accuratezza del test set al izionario summary
        # Save classification report & confusion matrix
        report = classification_report(test_labels, test_preds, target_names=class_names, digits=4) #calcolo delle metriche dettagliate per ogni classe
        with open(os.path.join(out_dir, "test_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        #creazione della matrice di confusione (questa serve per valutare le prestazioni di un modello, in particolare riesce a capire dove e quando sbaglia il modello)
        cm = confusion_matrix(test_labels, test_preds)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(out_dir, "test_confusion_matrix.csv"), encoding="utf-8")
    else:
        summary["test_acc"] = None #se non esiste il test set allora viene impostato a None
    #crea un summary json nella cartella di output specificata
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    #stampa dei risultati
    print(f"\n[{ds_name}] Finished. Best model: {best_model_path}")
    print(f"[{ds_name}] Val acc: {val_acc:.4f} | Test acc: {summary['test_acc']}")
    print(f"[{ds_name}] Metrics CSV: {metrics_path}")
    return summary

#determina la dimensione predefinita dell'immagine di input per il modello efficient net
def infer_default_img_size(arch: str) -> int:
    # dizionario size_map per contiene le coppie (nome architettura e dimensione input)
    size_map = {
        "efficientnet_b0": 224,
        "efficientnet_b1": 240,
        "efficientnet_b2": 260,
        "efficientnet_b3": 300,
        "efficientnet_b4": 380,
        "efficientnet_b5": 456,
        "efficientnet_b6": 528,
        "efficientnet_b7": 600,
        "tf_efficientnet_b0": 224,
        "tf_efficientnet_b1": 240,
        "tf_efficientnet_b2": 260,
        "tf_efficientnet_b3": 300,
        "tf_efficientnet_b4": 380,
        "tf_efficientnet_b5": 456,
        "tf_efficientnet_b6": 528,
        "tf_efficientnet_b7": 600,
    }
    return size_map.get(arch, 224)


def main():
    #creazione dell'oggetto per festire gli argomenti da linea di comando
    parser = argparse.ArgumentParser(description="Train EfficientNet (CPU-optimized)")
    #aggiunta di tutti gli argomenti
    parser.add_argument("--data_roots", type=str, required=True)
    parser.add_argument("--arch", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)  # Default ridotto
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count()))
    parser.add_argument("--lr", type=float, default=1e-3)  # LR aumentato
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=5)  # Modificato
    parser.add_argument("--gamma", type=float, default=0.5)  # Modificato
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs_cpu")  # Cartella diversa
    parser.add_argument("--img_size", type=int, default=128)  # Dimensione ridotta
    args = parser.parse_args()

    set_seed(42) #inizializzazione dei generatori di numeri casuali
    #controlla se è disponibile una gpu oppure una cpu in base a ciò che è disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Threads CPU: {torch.get_num_threads()}")

    # verifica i percorsi dei dataset
    for root in args.data_roots.split(","):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Percorso non trovato: {root}") #errore visualizzabile nel caso in cui il percorso non esiste

    # determinazione della dimensione dell'immagine
    img_size = args.img_size if args.img_size else infer_default_img_size(args.arch)
    print(f"Image size: {img_size} | Batch size: {args.batch_size}")
    
    #addestramento sui dataset (verifica che il percorso sia una cartella e itera su ogni percorso)
    all_summaries = []
    for root in args.data_roots.split(","):
        if os.path.isdir(root):
            summary = train_one_dataset(args, root.strip(), device, img_size)
            all_summaries.append(summary)
    #salvataggio di un file json nella cartella di output
    with open(os.path.join(args.out_dir, "ALL_SUMMARIES.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

if __name__ == "__main__":
    main()
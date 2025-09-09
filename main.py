import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import logging
import os
import math

from dataset_builder import CocoHomeDataset, get_train_transforms, get_val_transforms
from model_builder import create_ssd_model
from trainer_net import training_net
from tester_net import evaluate_model

# La tua funzione collate_fn
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    batch = list(filter(lambda x: x[1]['boxes'].numel() > 0, batch))
    if not batch: return None, None
    return tuple(zip(*batch))

def main():
    # Svuota la cache della GPU ---
    torch.cuda.empty_cache()

    #Configura la logica dei log
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training_log_ssd_coco.txt', filemode='a')
    with open('config.json', 'r') as f:
        config = json.load(f)

    #Seleziona il dispositivo cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")
    logging.info(f"Dispositivo: {device}, Parametri: batch_size={config['batch_size']}, epochs={config['num_epochs']}, lr={config['learning_rate']}")
    
    #Configura i percorsi dei dataset con le relative annotations
    data_dir = config['data_dir']
    annotations_file_template = config['annotations_file']
    train_ann_file = os.path.join(data_dir, annotations_file_template.format('train'))
    train_img_dir = os.path.join(data_dir, 'train')
    val_ann_file = os.path.join(data_dir, annotations_file_template.format('valid'))
    val_img_dir = os.path.join(data_dir, 'valid')

    print("Caricamento dataset di training...")
    temp_dataset = CocoHomeDataset(images_dir=train_img_dir, annotations_file=train_ann_file)
    num_classes = temp_dataset.num_classes
    train_dataset = CocoHomeDataset(images_dir=train_img_dir, annotations_file=train_ann_file, transforms=get_train_transforms())
    val_dataset = CocoHomeDataset(images_dir=val_img_dir, annotations_file=val_ann_file, transforms=get_val_transforms())
    
    #Caricamento dei Dataloader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], collate_fn=collate_fn)

    # --- Creazione del modello SSD ---
    print("Creazione del modello SSD...")
    model = create_ssd_model(num_classes=num_classes, device=device)

    # --- Strategia di addestramento a due fasi con Early Stopping ---

    # FASE 1: Addestramento della testa
    print("\n--- FASE 1: Congelamento del backbone e addestramento della testa ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    params_head_only = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_head_only, lr=config['learning_rate'])
    total_epochs = config['num_epochs']
    head_train_percent = config.get('head_train_percent', 0.2) 
    head_epochs = math.ceil(total_epochs * head_train_percent)
    print(f"Inizio addestramento della sola testa per {head_epochs} epoche...")
    training_net(model, train_loader, num_epochs=head_epochs, device=device, optimizer=optimizer, lr_scheduler=None)

    # FASE 2: Fine-tuning con Early Stopping
    remaining_epochs = total_epochs - head_epochs
    print(f"\n--- FASE 2: Scongelamento e fine-tuning completo per un massimo di {remaining_epochs} epoche ---")
    
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'] / 100, weight_decay=0.05)
    
    if remaining_epochs > 0:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    else:
        lr_scheduler = None

    # Inizializzazione per l'Early Stopping
    best_map = 0.0
    patience_limit = 10
    patience_counter = 0

    if remaining_epochs > 0:
        for epoch in range(remaining_epochs):
            print(f"\n--- Inizio Fine-Tuning: Epoca {epoch + 1}/{remaining_epochs} ---")
            
            model = training_net(model, train_loader, num_epochs=1, device=device, optimizer=optimizer, lr_scheduler=lr_scheduler)
            results = evaluate_model(model, val_loader, device)
            
            if results is None or 'map' not in results:
                print("Valutazione fallita, continuo con la prossima epoca.")
                continue

            current_map = results['map'].item()
            
            if current_map > best_map:
                best_map = current_map
                patience_counter = 0
                print(f"Nuovo mAP migliore! {best_map:.4f}. Salvo il modello in 'best_model.pth'.")
                torch.save(model.state_dict(), 'best_model.pth')
                logging.info(f"Nuovo mAP migliore: {best_map:.4f} all'epoca {epoch + 1} del fine-tuning.")
            else:
                patience_counter += 1
                print(f"Nessun miglioramento del mAP. Pazienza: {patience_counter}/{patience_limit}")

            if patience_counter >= patience_limit:
                print(f"Early stopping: le performance non migliorano da {patience_limit} epoche. Interrompo l'addestramento.")
                logging.info(f"Early stopping attivato. mAP migliore raggiunto: {best_map:.4f}")
                break
    
    print(f"\nAddestramento completato. Il miglior mAP ottenuto è stato: {best_map:.4f}")
    
    # Carichiamo il modello migliore per sicurezza, anche se l'ultimo salvataggio dovrebbe essere quello
    print("Caricamento del modello con le migliori performance...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Valutazione finale sul modello migliore
    print("\n--- Valutazione finale sul modello migliore ---")
    evaluate_model(model, val_loader, device)

    print("Processo completato.")

if __name__ == '__main__':
    main()
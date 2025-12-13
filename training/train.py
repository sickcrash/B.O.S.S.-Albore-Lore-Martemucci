import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from training.loss import SetCriterion
from training.utils import normalize_box
import os

def run_validation(model,val_loader,criterion,weight_dict,device,best_loss):
    model.eval()
    val_loss = 0
    val_bar = tqdm(val_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for batch in val_bar:
            images = batch["image"].to(device)
            Ks = batch["K"].to(device)
            domain_labels = batch["domain"].to(device)

            targets = []
            batch_size = images.shape[0]

            for i in range(batch_size):
                # numero effettivo di oggetti (ignoriamo il padding)
                n_obj = batch["num_objs"][i].item()

                # estrazione dati validi
                valid_boxes = batch["target_3d"][i, :n_obj].to(device)  # [N, 6]
                valid_labels = batch["target_cls"][i, :n_obj].to(device)  # [N]

                # NORMALIZZAZIONE
                # Portiamo i metri nel range [0, 1] per matchare l'output sigmoid
                norm_boxes_6d = normalize_box(valid_boxes, device)

                # Gestione 6 vs 7 dimensioni
                # Shape finale: [N, 7]
                rotation_placeholder = torch.zeros((n_obj, 1), device=device) + 0.5
                norm_boxes_7d = torch.cat([norm_boxes_6d, rotation_placeholder], dim=1)

                targets.append({
                    'boxes': norm_boxes_7d,
                    'labels': valid_labels
                })

            # --- FORWARD ---
            outputs = model(images, Ks, domain_labels)

            # --- LOSS ---
            # outputs['pred_boxes'] è già sigmoid [0, 1]
            loss_dict = criterion(outputs, targets)

            # calcolo esplicito della loss di dominio
            loss_domain = F.cross_entropy(outputs['pred_domain'], domain_labels)

            # Somma pesata
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
            losses += loss_domain * weight_dict['loss_domain']

            # --- BACKWARD ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            val_loss += losses.item()

        print(f"Validation Loss: {val_loss:.3f} - Avg Val Loss: {val_loss / len(val_loader):.3f}")


    if val_loss < best_loss:
        best_loss = val_loss
        if not os.path.exists("../checkpoints/"):
            os.makedirs("../checkpoints/")
        torch.save(model.state_dict(), '../checkpoints/best_model.pth')
        print(f'Pesi salvati con loss migliorata: {best_loss}')


    return best_loss

def train_model(model, matcher, loader_train, loader_validation, n_epochs=50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    # la loss di regressione è più alta di quella di classificazione perché il task è più difficile
    weight_dict = {
        'loss_ce': 2.0,  # Peso per la classificazione
        'loss_bbox': 5.0,  # Peso per i box 3D
        'loss_domain': 1.0 # Peso per la loss di dominio
    }

    criterion = SetCriterion(num_classes=8, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.1).to(device)

    # divide i parametri in due gruppi per assegnare alla backbone un learning rate più basso
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]

    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)

    best_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch")


        for batch in progress_bar:
            images = batch["image"].to(device)
            Ks = batch["K"].to(device)
            domain_labels = batch["domain"].to(device)

            targets = []
            batch_size = images.shape[0]

            for i in range(batch_size):
                # numero effettivo di oggetti (ignoriamo il padding)
                n_obj = batch["num_objs"][i].item()

                # estrazione dati validi
                valid_boxes = batch["target_3d"][i, :n_obj].to(device)  # [N, 6]
                valid_labels = batch["target_cls"][i, :n_obj].to(device)  # [N]

                # NORMALIZZAZIONE
                # Portiamo i metri nel range [0, 1] per matchare l'output sigmoid
                norm_boxes_6d = normalize_box(valid_boxes, device)

                # Gestione 6 vs 7 dimensioni
                # Shape finale: [N, 7]
                rotation_placeholder = torch.zeros((n_obj, 1), device=device) + 0.5
                norm_boxes_7d = torch.cat([norm_boxes_6d, rotation_placeholder], dim=1)

                targets.append({
                    'boxes': norm_boxes_7d,
                    'labels': valid_labels
                })

            # --- FORWARD ---
            optimizer.zero_grad()
            outputs = model(images, Ks, domain_labels)

            # --- LOSS ---
            # outputs['pred_boxes'] è già sigmoid [0, 1]
            loss_dict = criterion(outputs, targets)

            # calcolo esplicito della loss di dominio
            loss_domain = F.cross_entropy(outputs['pred_domain'], domain_labels)

            # Somma pesata
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
            losses += loss_domain * weight_dict['loss_domain']

            # --- BACKWARD ---
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Vitale per Transformer
            optimizer.step()

            total_loss += losses.item()

        print(f"Train Loss: {total_loss:.3f}")
        print(f"Epoch {epoch+1}: Avg Loss {total_loss / len(loader_train)}")

        if epoch % 10 == 0: # Ogni 10 epoche testiamo sul dataset di validation
            best_loss = run_validation(model, loader_validation, criterion, weight_dict, device, best_loss)

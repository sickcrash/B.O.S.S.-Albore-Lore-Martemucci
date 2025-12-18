import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from training.loss import SetCriterion
from training.utils import normalize_box
import os
import random
import torchvision.transforms as transforms
import cv2
import numpy as np
from training.evaluator import get_3d_box_from_chw, project_points_K
from training.utils import SCENE_LIMITS


EDGES_3D = [
    (0,1),(1,2),(2,3),(3,0),     # fronte
    (4,5),(5,6),(6,7),(7,4),     # retro
    (0,4),(1,5),(2,6),(3,7)      # verticali
]


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
                # numero effettivo di oggetti
                n_obj = batch["num_objs"][i].item()

                # estraiamo solo i dati validi
                valid_boxes = batch["target_3d"][i, :n_obj].to(device)  # [N, 6]
                valid_labels = batch["target_cls"][i, :n_obj].to(device)  # [N]

                # normalizzazione
                domain_str = "kitti" if domain_labels[i].item() == 1 else "sunrgbd"
                norm_boxes_6d = normalize_box(valid_boxes, device, domain_str)

                rotation_placeholder = torch.zeros((n_obj, 1), device=device) + 0.5
                norm_boxes_7d = torch.cat([norm_boxes_6d, rotation_placeholder], dim=1)

                targets.append({
                    'boxes': norm_boxes_7d,
                    'labels': valid_labels
                })

            outputs = model(images, Ks, domain_labels)
            loss_dict = criterion(outputs, targets)

            # calcolo esplicito della loss di dominio
            loss_domain = F.cross_entropy(outputs['pred_domain'], domain_labels)

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
            losses += loss_domain * weight_dict['loss_domain']

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
                # numero effettivo di oggetti
                n_obj = batch["num_objs"][i].item()

                # estraiamo solo i dati validi
                valid_boxes = batch["target_3d"][i, :n_obj].to(device)  # [N, 6]
                valid_labels = batch["target_cls"][i, :n_obj].to(device)  # [N]

                # normalizzazione
                domain_str = "kitti" if domain_labels[i].item() == 1 else "sunrgbd"
                norm_boxes_6d = normalize_box(valid_boxes, device, domain_str)

                rotation_placeholder = torch.zeros((n_obj, 1), device=device) + 0.5
                norm_boxes_7d = torch.cat([norm_boxes_6d, rotation_placeholder], dim=1)

                targets.append({
                    'boxes': norm_boxes_7d,
                    'labels': valid_labels
                })

            optimizer.zero_grad()
            outputs = model(images, Ks, domain_labels)

            if random.random() < 0.01: # probabilità 1% di visualizzare le immagini con i box 3D
                for i in range(batch_size):
                    denorm = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
                    img_t = images[i].detach().cpu()
                    img_t = denorm(img_t)
                    img_np = img_t.numpy().transpose(1, 2, 0)  # (H,W,C)
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    K = Ks[i].detach().cpu().numpy()  # 3x3
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    gt_boxes = batch["target_3d"][i, :n_obj].to(device)

                    domain = 'kitti' if domain_labels[i].item() == 0 else 'sunrgbd'
                    limits = SCENE_LIMITS[domain]
                    min_v = limits['min'].to(device)[:6]
                    max_v = limits['max'].to(device)[:6]
                    scene_range = max_v - min_v

                    p_boxes = outputs['pred_boxes'][i]

                    p_boxes_scaled = p_boxes[..., :6] * scene_range + min_v

                    # disegna GT in verde
                    for gt in gt_boxes[..., :6].detach().cpu().numpy():
                        cx, cy, cz, w, h, d = gt
                        corners3d = get_3d_box_from_chw(cx, cy, cz, w, h, d)
                        pts2d = project_points_K(K, corners3d)
                        for a, b in EDGES_3D:
                            cv2.line(img_bgr, tuple(pts2d[a]), tuple(pts2d[b]), (0, 255, 0), 2)

                    # disegna pred in blu
                    for pb in p_boxes_scaled.detach().cpu().numpy():
                        cx, cy, cz, w, h, d = pb[:6]
                        corners3d = get_3d_box_from_chw(cx, cy, cz, w, h, d)
                        pts2d = project_points_K(K, corners3d)
                        for a, b in EDGES_3D:
                            cv2.line(img_bgr, tuple(pts2d[a]), tuple(pts2d[b]), (0, 0, 255), 2)

                    cv2.imshow("3D boxes debug", img_bgr)
                    cv2.waitKey(1)


            loss_dict = criterion(outputs, targets)

            # calcolo esplicito della loss di dominio
            loss_domain = F.cross_entropy(outputs['pred_domain'], domain_labels)

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
            losses += loss_domain * weight_dict['loss_domain']

            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Vitale per Transformer
            optimizer.step()

            total_loss += losses.item()

        print(f"Train Loss: {total_loss:.3f}")
        print(f"Epoch {epoch+1}: Avg Loss {total_loss / len(loader_train)}")

        if epoch % 10 == 0:
            best_loss = run_validation(model, loader_validation, criterion, weight_dict, device, best_loss)
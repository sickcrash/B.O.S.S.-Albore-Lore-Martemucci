import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from training.utils import SCENE_LIMITS
import torchvision.transforms as transforms
import random
import cv2

def get_3d_box_from_chw(cx, cy, cz, w, h, d):
    dx, dy, dz = w / 2, h / 2, d / 2
    corners = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ], dtype=np.float32)
    return corners  # (8,3)

def project_points_K(K, pts3d):
    pts3d = pts3d.T                      # (3,N)
    pts2d = K @ pts3d                    # (3,N)
    pts2d = pts2d / pts2d[2:3, :]        # normalizza
    return pts2d[:2, :].T.astype(int)    # (N,2)

EDGES_3D = [
    (0,1),(1,2),(2,3),(3,0),     # fronte
    (4,5),(5,6),(6,7),(7,4),     # retro
    (0,4),(1,5),(2,6),(3,7)      # verticali
]

class Evaluator:
    def __init__(self, model, device, num_classes=8, iou_threshold=0.5):
        """
        Classe per valutare il modello unificato Indoor/Outdoor.
        Calcola mAP (Detection), Domain Accuracy e restituisce i dati per la Matrice di Confusione.
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        # Mappa ID -> Nome Classe per i report
        self.class_names = {
            0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist',
            3: 'Chair', 4: 'Table', 5: 'Bed', 6: 'Sofa', 7: 'Bookshelf'
        }

    def compute_iou_aligned(self, box_a, box_b):
        """
        Calcola la 3D Intersection over Union (IoU) per box allineati agli assi.
        Input shape: [N, 6] -> (cx, cy, cz, w, h, d)
        """

        def get_min_max(box):
            mins = box[..., :3] - box[..., 3:6] / 2
            maxs = box[..., :3] + box[..., 3:6] / 2
            return mins, maxs

        a_min, a_max = get_min_max(box_a)
        b_min, b_max = get_min_max(box_b)

        inter_mins = torch.max(a_min.unsqueeze(1), b_min.unsqueeze(0))
        inter_maxs = torch.min(a_max.unsqueeze(1), b_max.unsqueeze(0))
        inter_whd = torch.clamp(inter_maxs - inter_mins, min=0)
        intersection = inter_whd[..., 0] * inter_whd[..., 1] * inter_whd[..., 2]

        vol_a = box_a[..., 3] * box_a[..., 4] * box_a[..., 5]
        vol_b = box_b[..., 3] * box_b[..., 4] * box_b[..., 5]
        union = vol_a.unsqueeze(1) + vol_b.unsqueeze(0) - intersection

        return intersection / (union + 1e-6)

    @torch.no_grad()
    def run_evaluation(self, dataloader, desc="Evaluating"):
        self.model.eval()

        det_results = defaultdict(list)
        total_gt_per_class = defaultdict(int)

        correct_domain = 0
        total_samples = 0

        cls_targets = []
        cls_preds = []

        for batch in tqdm(dataloader, desc=desc):
            images = batch["image"].to(self.device)
            Ks = batch["K"].to(self.device)
            domain_labels = batch["domain"].to(self.device)

            outputs = self.model(images, Ks, domain_labels)

            # 1. Domain Accuracy
            if 'pred_domain' in outputs:
                d_preds = outputs['pred_domain'].argmax(dim=1)
                correct_domain += (d_preds == domain_labels).sum().item()
            total_samples += images.size(0)

            # 2. Detection
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

            probs = pred_logits.softmax(-1)
            scores, labels = probs[..., :-1].max(-1)

            batch_size = images.shape[0]

            for i in range(batch_size):
                n_obj = batch["num_objs"][i].item()
                if n_obj == 0: continue

                gt_boxes = batch["target_3d"][i, :n_obj].to(self.device)
                gt_labels = batch["target_cls"][i, :n_obj].to(self.device)

                for lbl in gt_labels:
                    total_gt_per_class[lbl.item()] += 1

                # Filtro Confidenza
                mask = scores[i] > 0.01

                p_boxes = pred_boxes[i, mask]
                p_labels = labels[i, mask]
                p_scores = scores[i, mask]

                if len(p_boxes) == 0: continue

                # Denormalizzazione
                domain = 'kitti' if domain_labels[i].item() == 0 else 'sunrgbd'
                limits = SCENE_LIMITS[domain]
                min_v = limits['min'].to(self.device)[:6]
                max_v = limits['max'].to(self.device)[:6]
                scene_range = max_v - min_v
                p_boxes_scaled = p_boxes[..., :6] * scene_range + min_v

                if random.random() < 0.4:
                    denorm = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
                    img_t = images[i].detach().cpu()  # (C,H,W), in [0,1] o [0,255]
                    img_t = denorm(img_t)  # Denormalizziamo per far comparire un'immagine visibile
                    img_np = img_t.numpy().transpose(1, 2, 0)  # (H,W,C)
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    K = Ks[i].detach().cpu().numpy()  # 3x3

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

                iou_matrix = self.compute_iou_aligned(p_boxes_scaled, gt_boxes[..., :6])

                # Matching per mAP
                sorted_idx = torch.argsort(p_scores, descending=True)
                used_gt = torch.zeros(n_obj, dtype=torch.bool, device=self.device)

                for idx in sorted_idx:
                    pred_cls = p_labels[idx].item()
                    score = p_scores[idx].item()

                    best_iou = 0
                    best_gt_idx = -1

                    class_gt_indices = [j for j in range(n_obj) if gt_labels[j] == pred_cls]

                    for gt_idx in class_gt_indices:
                        if not used_gt[gt_idx]:
                            iou = iou_matrix[idx, gt_idx].item()
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx

                    if best_iou >= self.iou_threshold:
                        det_results[pred_cls].append({'score': score, 'tp': 1, 'fp': 0})
                        used_gt[best_gt_idx] = True
                    else:
                        det_results[pred_cls].append({'score': score, 'tp': 0, 'fp': 1})

                # Matching per Confusion Matrix
                if iou_matrix.numel() > 0:
                    max_ious, pred_indices = iou_matrix.max(dim=0)
                    for gt_idx, (iou, pred_idx) in enumerate(zip(max_ious, pred_indices)):
                        if iou > 0.1:  # Soglia morbida per localizzazione
                            cls_targets.append(gt_labels[gt_idx].item())
                            cls_preds.append(p_labels[pred_idx].item())

        domain_acc = correct_domain / total_samples if total_samples > 0 else 0

        aps = []
        for cls_id in sorted(total_gt_per_class.keys()):
            if cls_id not in det_results:
                aps.append(0)
                continue

            cls_data = det_results[cls_id]
            cls_data.sort(key=lambda x: x['score'], reverse=True)

            tps = np.cumsum([x['tp'] for x in cls_data])
            fps = np.cumsum([x['fp'] for x in cls_data])

            num_positives = total_gt_per_class[cls_id]
            recalls = tps / num_positives
            precisions = tps / (tps + fps + 1e-6)

            ap = np.trapz(precisions, recalls)
            aps.append(ap)

        mAP = np.mean(aps) if aps else 0

        return {
            'mAP': mAP,
            'domain_acc': domain_acc,
            'targets': cls_targets,
            'preds': cls_preds
        }
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Calcola l'assegnazione ottimale tra le predizioni e i target.
    Costo = -prob_classe + costo_L1_box
    """

    def __init__(self, cost_class=1, cost_bbox=5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict con 'pred_logits' [B, num_queries, num_classes] e 'pred_boxes' [B, num_queries, 7]
        targets: lista di dict (len=B), ogni dict ha 'labels' e 'boxes'
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten per gestire il batch tutto insieme
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, NumClasses]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 7]

        # Concatena i target
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Calcolo Costo Classificazione
        cost_class = -out_prob[:, tgt_ids]

        # Calcolo Costo Box (L1 Loss semplificata sui 7 parametri)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Costo Totale
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # lista di tuple (pred_idx, target_idx) per ogni batch
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """
    Loss Composta:
    1. Classification Loss (Cross Entropy sui match, 'background' sui non-match)
    2. Regression Loss (L1 sui box matchati)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef  # Peso per la classe "Background" (No Object)

        # Buffer per la classe "background" (l'ultima classe, indice num_classes)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('eos_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        # Helper per indicizzare i tensori batchati
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        # Assegna le classi ai match, il resto rimane "background" (self.num_classes)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.eos_weight)
        return loss_ce

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_boxes) == 0:
            return src_boxes.sum() * 0.0

        # Calcoliamo la loss solo sui primi 6 parametri (x,y,z,w,h,d)
        loss_bbox = F.l1_loss(src_boxes[:, :6], target_boxes[:, :6], reduction='mean')

        return loss_bbox

    def forward(self, outputs, targets):
        # matching ottimale
        indices = self.matcher(outputs, targets)

        losses = {}
        losses['loss_ce'] = self.loss_labels(outputs, targets, indices)
        losses['loss_bbox'] = self.loss_boxes(outputs, targets, indices)

        return losses
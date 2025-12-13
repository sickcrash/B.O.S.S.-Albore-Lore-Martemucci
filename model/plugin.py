import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model, group_num=2):
        super().__init__()
        self.d_model = d_model
        self.group_num = group_num
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.group_weight = nn.Embedding(group_num, d_model)
        self.group_bias = nn.Embedding(group_num, d_model)
        self.group_weight.weight.data.fill_(1.0)
        self.group_bias.weight.data.fill_(0.0)

    def forward(self, x, dataset_group_pred=None):
        # x: [B, Seq, C]
        B = x.size(0)
        x = self.layer_norm(x)

        if dataset_group_pred is None:
            dataset_group_pred = x.new_zeros(B, self.group_num)
            dataset_group_pred[:, 0] = 1.0

        # [B, group_num]
        if dataset_group_pred.dim() == 1:
            dataset_group_pred = F.one_hot(dataset_group_pred.long(), num_classes=self.group_num).float()

        # Se è già [B, 2], applica softmax
        if dataset_group_pred.size(-1) == self.group_num:
            dataset_group_pred = F.softmax(dataset_group_pred.float(), dim=-1)
        else:
            raise ValueError(f"dataset_group_pred shape mismatch: {dataset_group_pred.shape}")

        # dataset_group_pred: [B, 2] @ [2, C] = [B, C]
        weight = dataset_group_pred @ self.group_weight.weight  # [B, C]
        bias = dataset_group_pred @ self.group_bias.weight  # [B, C]

        # Reshape: [B, C] -> [B, 1, C]
        weight = weight.unsqueeze(1)
        bias = bias.unsqueeze(1)

        return weight * x + bias


class Plugin(nn.Module):
    def __init__(self, plugin, num_features, num_groups=2, eps=1e-5):
        super().__init__()
        self.plugin = plugin

        if self.plugin == "BatchNorm":
            # Batch Normalization classica 2D per immagini (es. conv output)
            self.norm = nn.BatchNorm2d(num_features, eps=eps)

        elif self.plugin == "NoNorm":
            # Nessuna normalizzazione, layer Identity
            self.norm = nn.Identity()

        elif self.plugin == "ADLN":
            # Adaptive Layer Normalization (qui un LayerNorm standard come esempio)
            # ADLN va definito secondo la formula specifica o libreria esterna
            self.norm = AdaptiveLayerNorm(num_features,num_groups) # Da inserire modulo normalization custom

        else:
            raise ValueError(f"Plugin di normalizzazione '{plugin}' non riconosciuto")

    def forward(self, x,group_pred=None):
        # Per LayerNorm in immagini 4D può essere necessario permutare dimensioni (B,C,H,W -> B,H,W,C)
        if self.plugin == "ADLN":
            x = self.norm(x,group_pred)
        else:
            x = self.norm(x)
        return x


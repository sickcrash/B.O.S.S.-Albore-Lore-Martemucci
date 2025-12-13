import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from model.plugin import Plugin
import math


class SineCosine1D(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: [1, max_len, d_model] per broadcasting su [B, max_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, N):
        # [1, N, C]
        return self.pe[:, :N]


class Transformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.1, num_domains=2):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        # ENCODER
        encoder_layer = TransformerEncoderLayerAdapt(d_model, nhead, dim_feedforward, dropout, num_domains)
        self.encoder = TransformerEncoderAdapt(encoder_layer, num_encoder_layers)

        # DECODER
        decoder_layer = TransformerDecoderLayerAdapt(d_model, nhead, dim_feedforward, dropout, num_domains)
        self.decoder = TransformerDecoderAdapt(decoder_layer, num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, bev_features, query_embed, domain_pred):
        """
        Argomenti:
            bev_features: [B, C, H, W]
            query_embed:  [NumQueries, C]
            domain_pred:  [B, 2]
        """
        B, C, H, W = bev_features.shape
        N_pixels = H * W

        # Flatten Feature Map: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        src_flat = bev_features.flatten(2).transpose(1, 2)

        # Positional Encoding
        pos_encoder = SineCosine1D(d_model=C, max_len=N_pixels)
        pos = pos_encoder(N_pixels).to(src_flat.device)  # Shape [1, H*W, C] -> Broadcastable su B

        # Encoder Pass
        # src_flat: [B, Seq, C]
        # pos: [1, Seq, C]
        memory = self.encoder(src_flat, pos=pos, padding_mask=None, group_pred=domain_pred)

        # Espansione Query Embed per il batch: [N, C] -> [B, N, C]
        query_pos = query_embed.unsqueeze(0).expand(B, -1, -1)

        # Target inizializzato a zero: [B, N, C]
        target = torch.zeros_like(query_pos)

        # Decoder Pass
        # Tutti gli input sono [B, Seq, C]
        hs = self.decoder(
            tgt=target,
            memory=memory,
            query_pos=query_pos,
            pos=pos,
            group_pred=domain_pred
        )

        # Output [B, NumQueries, C]
        return hs


# --- Building Blocks Adattivi (Batch First) ---

class TransformerEncoderLayerAdapt(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, num_domains=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = Plugin("ADLN", d_model, num_domains)
        self.norm2 = Plugin("ADLN", d_model, num_domains)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pos, padding_mask, group_pred):
        # src: [B, Seq, C]
        # pos: [1, Seq, C] o [B, Seq, C]

        q = k = src + pos if pos is not None else src

        # self_attn ritorna (output, weights)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=padding_mask)[0]
        src = src + self.dropout1(src2)

        # Norm 1
        src = self.norm1(src, group_pred)

        # FFN
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)

        # Norm 2
        src = self.norm2(src, group_pred)
        return src


class TransformerDecoderLayerAdapt(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, num_domains=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = Plugin("ADLN", d_model, num_domains)
        self.norm2 = Plugin("ADLN", d_model, num_domains)
        self.norm3 = Plugin("ADLN", d_model, num_domains)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos, pos, group_pred):
        # tgt:    [B, NumQ, C]
        # memory: [B, HW, C]

        # 1. Self Attention (Query-to-Query)
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt, group_pred)

        # 2. Cross Attention (Query-to-Image)
        # query: tgt + query_pos
        # key:   memory + pos
        q = tgt + query_pos if query_pos is not None else tgt
        k = memory + pos if pos is not None else memory

        tgt2 = self.multihead_attn(query=q, key=k, value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt, group_pred)

        # 3. FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt, group_pred)

        return tgt


# --- Container Classes ---

class TransformerEncoderAdapt(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src, pos, padding_mask, group_pred):
        output = src
        for layer in self.layers:
            output = layer(output, pos, padding_mask, group_pred)
        return output


class TransformerDecoderAdapt(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, memory, query_pos, pos, group_pred):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, query_pos, pos, group_pred)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from model.plugin import Plugin
import math


class SineCosine2D(nn.Module):
    """
    Positional Encoding Spaziale per griglie 2D (H, W).
    Ogni pixel riceve una codifica unica basata su (x, y).
    """

    def __init__(self, d_model, max_h=200, max_w=200):
        super().__init__()
        # Assicurati che d_model sia pari e divisibile per 2
        assert d_model % 2 == 0
        d_model_half = d_model // 2

        # Encoding Y (Altezza/Profondità)
        pe_h = torch.zeros(max_h, d_model_half)
        position_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        div_term_h = torch.exp(
            torch.arange(0, d_model_half, 2, dtype=torch.float) * -(math.log(10000.0) / d_model_half))
        pe_h[:, 0::2] = torch.sin(position_h * div_term_h)
        pe_h[:, 1::2] = torch.cos(position_h * div_term_h)

        # Encoding X (Larghezza)
        pe_w = torch.zeros(max_w, d_model_half)
        position_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        div_term_w = torch.exp(
            torch.arange(0, d_model_half, 2, dtype=torch.float) * -(math.log(10000.0) / d_model_half))
        pe_w[:, 0::2] = torch.sin(position_w * div_term_w)
        pe_w[:, 1::2] = torch.cos(position_w * div_term_w)

        self.register_buffer('pe_h', pe_h)  # [MaxH, C/2]
        self.register_buffer('pe_w', pe_w)  # [MaxW, C/2]
        self.d_model = d_model

    def forward(self, H, W):
        # Seleziona le prime H e W posizioni
        # pe_h: [H, C/2] -> [H, 1, C/2] -> [H, W, C/2]
        # pe_w: [W, C/2] -> [1, W, C/2] -> [H, W, C/2]

        h_embed = self.pe_h[:H].unsqueeze(1).expand(-1, W, -1)
        w_embed = self.pe_w[:W].unsqueeze(0).expand(H, -1, -1)

        # Concatena -> [H, W, C]
        pe = torch.cat([w_embed, h_embed], dim=-1)

        # Flatten -> [1, H*W, C] per essere compatibile col Transformer
        return pe.flatten(0, 1).unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.1, num_domains=2):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.pos_encoder = SineCosine2D(d_model)

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
        pos = self.pos_encoder(H, W).to(src_flat.device)  # Shape [1, H*W, C] -> Broadcastable su B

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
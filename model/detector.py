import torch.nn as nn
from model.backbone import Backbone
from model.neck import NeckPAFPN
from model.bridge import Bridge
from model.domain_head import DomainHead
from model.transformer import Transformer

class UnifiedDetector(nn.Module):

    def __init__(self, num_classes=8, num_queries=50):
        super().__init__()

        # 1. Backbone: Estrae feature 2D
        self.backbone = Backbone(pretrained=True)

        # 2. Neck: Fonde le feature mantenendo dettagli spaziali
        # Input: [128, 256, 512] -> Output: 256 canali
        self.neck = NeckPAFPN(in_channels_list=[128, 256, 512], out_channels=256)

        # 3. Bridge: Proietta da 2D a BEV
        # Input: 256 canali -> Output: 128 canali BEV
        self.bridge = Bridge(in_channels=256, out_channels=128)
        # Domain head
        # Input: 256 channel -> Output: 2, predizione di domain(indoor-outdoor)
        self.domain_head = DomainHead(in_channel=256)

        self.transformer = Transformer(
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_domains=2
        )

        # Query Embeddings
        self.query_embed = nn.Embedding(num_queries, 128)

        # Heads finali
        self.class_head = nn.Linear(128, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, image, K, domain_label):
        """
        Flow Completo:
        Img (Pixel) -> Backbone (Feat) -> Neck (Enhanced Feat) -> Bridge (BEV) -> Head (Box)
        """
        # Backbone
        features = self.backbone(image, domain_label)

        # Neck
        neck_out = self.neck(features, domain_label)

        # Bridge
        bev_feat = self.bridge(neck_out, K, domain_label)

        domain_output = self.domain_head(neck_out)

        hs = self.transformer(
            bev_features=bev_feat,
            query_embed=self.query_embed.weight,
            domain_pred=domain_output
        )

        # Predizioni
        out_cls = self.class_head(hs)
        out_bbox = self.bbox_head(hs).sigmoid()

        return {
            'pred_logits': out_cls,
            'pred_boxes': out_bbox,
            'pred_domain': domain_output
        }

import torch.nn as nn
import torch.nn.functional as F

from model.backbone import AdaptiveBatchNorm


class ConvBlock(nn.Module):
    """Conv + AdaptiveNorm + ReLU"""

    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False)
        self.norm = AdaptiveBatchNorm(out_c)
        self.act = nn.ReLU()

    def forward(self, x, domain_label):
        return self.act(self.norm(self.conv(x), domain_label))


class NeckPAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network with Adaptive BatchNorm.
    Migliora la capacità di rappresentazione combinando feature maps a diverse scale."""

    def __init__(self, in_channels_list=[128, 256, 512], out_channels=256):
        super().__init__()

        # top-down layers
        # Qui passiamo dai layer più densi che contengono l'informazione semantica ai layer shallow che contengono l'informazione spaziale
        self.reduce_c4 = ConvBlock(in_channels_list[2], out_channels, 1, 0)
        self.reduce_c3 = ConvBlock(in_channels_list[1], out_channels, 1, 0)
        self.reduce_c2 = ConvBlock(in_channels_list[0], out_channels, 1, 0)

        # bottom-up layers
        # Facciamo esattamente il contrario di top-down
        self.down_p2 = ConvBlock(out_channels, out_channels, 3, 1)
        self.down_sample_p2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self.down_p3 = ConvBlock(out_channels, out_channels, 3, 1)
        self.down_sample_p3 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self.fusion_conv = ConvBlock(out_channels, out_channels, 3, 1)

    def forward(self, features, domain_label):
        # c2 high res, semantic poor
        # c3 medium res, medium semantic
        # c4 low res, semantic rich
        c2, c3, c4 = features # from backbone

        # top-down pathway
        p4 = self.reduce_c4(c4, domain_label)

        # P4 up -> P3
        p4_up = F.interpolate(p4, size=self.reduce_c3(c3, domain_label).shape[-2:], mode='nearest')
        p3_in = self.reduce_c3(c3, domain_label)
        p3_mid = p3_in + p4_up

        # P3 up -> P2
        p3_up = F.interpolate(p3_mid, size=self.reduce_c2(c2, domain_label).shape[-2:], mode='nearest')
        p2_in = self.reduce_c2(c2, domain_label)
        p2_out = p2_in + p3_up

        # bottom-up pathway
        # P2 -> P3
        n2_down = self.down_sample_p2(p2_out)
        n3_out = p3_mid + n2_down

        return self.fusion_conv(n3_out, domain_label)
import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead

def create_ssd_model(num_classes, device):
    weights = SSD300_VGG16_Weights.COCO_V1
    model = ssd300_vgg16(weights=weights)

    num_anchors_list = model.anchor_generator.num_anchors_per_location()
    
    cls_head = model.head.classification_head
    conv_layers = [m for m in cls_head.modules() if isinstance(m, nn.Conv2d)]
    in_channels_list = [c.in_channels for c in conv_layers]

    model.head = SSDHead(
        in_channels=in_channels_list,
        num_anchors=num_anchors_list,
        num_classes=num_classes
    )
    return model.to(device)
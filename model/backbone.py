import torch
import torch.nn as nn
import torchvision.models as models


class AdaptiveBatchNorm(nn.Module):
    """
    Batch norm adattiva in base a indoor / outdoor domain.
    """

    def __init__(self, num_features):
        super().__init__()
        # statistiche separate per i due domini
        self.bn_indoor = nn.BatchNorm2d(num_features)
        self.bn_outdoor = nn.BatchNorm2d(num_features)

    def forward(self, x, domain_label):
        """
        Args:
            x: Feature map [Batch, Channels, H, W]
            domain_label: Tensore [Batch] con 0 (Indoor) o 1 (Outdoor)
        """
        # tutto il batch è di un solo dominio
        if (domain_label == 0).all():
            return self.bn_indoor(x)
        elif (domain_label == 1).all():
            return self.bn_outdoor(x)

        # il batch contiene entrambi i domini
        out = torch.zeros_like(x)

        # Maschere
        mask_indoor = (domain_label == 0)
        mask_outdoor = (domain_label == 1)

        if mask_indoor.any():
            out[mask_indoor] = self.bn_indoor(x[mask_indoor])

        if mask_outdoor.any():
            out[mask_outdoor] = self.bn_outdoor(x[mask_outdoor])

        return out


class Backbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.resnet = models.resnet18(pretrained=pretrained)

        del self.resnet.fc
        del self.resnet.avgpool

        self.resnet.layer4[0].conv1.stride = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)

        # sostituzione con custom BN adattivo
        self._replace_bn_layers(self.resnet)

        self.out_channels = 512
        self.downsample_factor = 16  # 512 -> 32

    def _replace_bn_layers(self, module):
        """
        Funzione ricorsiva che cerca nn.BatchNorm2d e la sostituisce
        con AdaptiveBatchNorm.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_features = child.num_features
                new_bn = AdaptiveBatchNorm(num_features)

                # Copiamo i pesi iniziali dalla BN pre-trainata alla nostra
                # Copiamo su entrambe le strategie per partire da una base comune
                new_bn.bn_indoor.load_state_dict(child.state_dict())
                new_bn.bn_outdoor.load_state_dict(child.state_dict())

                # Sostituiamo il layer nel modulo padre
                setattr(module, name, new_bn)
            else:
                # Continua a cercare nei figli
                self._replace_bn_layers(child)

    def forward(self, x, domain_label):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x, domain_label)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        c1 = self._forward_layer(self.resnet.layer1, x, domain_label)
        c2 = self._forward_layer(self.resnet.layer2, c1, domain_label)
        c3 = self._forward_layer(self.resnet.layer3, c2, domain_label)
        c4 = self._forward_layer(self.resnet.layer4, c3, domain_label)

        return c2, c3, c4 # per compatibilità con Neck

    def _forward_layer(self, layer_block, x, domain_label):
        """
        Helper per passare domain_label attraverso i blocchi Sequential di ResNet
        """
        for module in layer_block:
            x = self._forward_basic_block_adaptive(module, x, domain_label)
        return x

    def _forward_basic_block_adaptive(self, block, x, domain_label):
        """
        Esegue il forward di un BasicBlock di ResNet passando domain_label alle BN.
        """
        identity = x

        out = block.conv1(x)
        out = block.bn1(out, domain_label)
        out = block.relu(out)

        out = block.conv2(out)
        out = block.bn2(out, domain_label)

        if block.downsample is not None:
            for layer in block.downsample:
                if isinstance(layer, AdaptiveBatchNorm):
                    identity = layer(identity, domain_label)
                else:
                    identity = layer(identity)

        out += identity
        out = block.relu(out)

        return out
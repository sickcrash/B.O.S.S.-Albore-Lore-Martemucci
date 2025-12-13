from torch import nn
import torch.nn.functional as F

class DomainHead(nn.Module):
    def __init__(self, in_channel):
        super(DomainHead, self).__init__()

        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # x: [B, in_channel, H, W]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.global_pool(x)  # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]

        x = F.relu(self.fc1(x))  # [B, 32]
        x = self.fc2(x)  # [B, 2]

        return x  # [B, 2]

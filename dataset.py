import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as F
from torchvision import transforms as T

IGNORE_INDEX = 255
OBSTACLE_IDS = [11, 12, 13, 14, 15, 16, 17, 18]  # person, rider, car, truck, bus, train, motorcycle, bicycle

class CityscapesBinary(Dataset):
    def __init__(self, root, split="train", img_size=(512, 1024), augment=True):
        self.ds = Cityscapes(root=root, split=split, mode="fine", target_type="semantic")
        self.img_h, self.img_w = img_size
        self.augment = augment and split == "train"
        self.color_jitter = T.ColorJitter(0.2, 0.2, 0.1, 0.05)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, target = self.ds[idx]  # PIL image + trainIds
        img = F.resize(img, (self.img_h, self.img_w), interpolation=F.InterpolationMode.BILINEAR)
        target = F.resize(target, (self.img_h, self.img_w), interpolation=F.InterpolationMode.NEAREST)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = self.color_jitter(img)
            if torch.rand(1).item() < 0.5:
                img = F.hflip(img)
                target = F.hflip(target)

        img_t = F.to_tensor(img)
        img_t = F.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        target_np = np.array(target, dtype=np.uint8)
        mask = np.zeros_like(target_np, dtype=np.uint8)
        for oid in OBSTACLE_IDS:
            mask[target_np == oid] = 1
        mask[target_np == IGNORE_INDEX] = IGNORE_INDEX

        return img_t, torch.from_numpy(mask.astype(np.int64))
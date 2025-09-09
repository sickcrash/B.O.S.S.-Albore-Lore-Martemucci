import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CocoHomeDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        self.coco = COCO(annotations_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.category_ids = sorted(self.coco.getCatIds())
        self.category_id_to_label = {cat_id: i for i, cat_id in enumerate(self.category_ids)}
        self.num_classes = len(self.category_ids) + 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image = cv2.imread(img_path)
        if image is None:
            print(f"Attenzione: immagine non trovata {img_path}. Campione ignorato.")
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        boxes = []
        labels = []
        for ann in anns:
            if ann['category_id'] not in self.category_id_to_label:
                continue

            x_min, y_min, w, h = ann['bbox']
            x_max = x_min + w
            y_max = y_min + h

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)
            
            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                # LOGICA CORRETTA PER SSD: le etichette degli oggetti partono da 1
                labels.append(self.category_id_to_label[ann['category_id']] + 1)

        target = {}
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            
        return image, target

def get_train_transforms():
    return A.Compose([
        A.RandomSizedBBoxSafeCrop(width=512, height=512, erosion_rate=0.2, p=0.5),

        # --- Trasformazioni Geometriche e di Colore (le tue attuali) ---
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

        # --- Rumore ---
        A.GaussNoise(p=0.2),
        
        # --- Trasformazioni finali ---
        A.Resize(512, 512),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.1))

def get_val_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
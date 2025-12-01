import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
import numpy as np
import cv2
from decord import VideoReader, cpu

##############################################
# 13. DATASET CLASSES (SEMPLIFICATI)
##############################################

class RealBDDImageDataset(Dataset):
    """Dataset BDD100K immagini"""

    def __init__(self, annotations_path, img_root, split: str, is_train: bool = True, max_samples=None):
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
            print(f"✅ BDD {split} - {len(self.annotations)} immagini")
        else:
            self.annotations = []
            print(f"⚠️ Annotations non trovate: {annotations_path}")

        if max_samples and len(self.annotations) > max_samples:
            self.annotations = self.annotations[:max_samples]

        self.img_root = img_root
        self.split = split
        self.is_train = is_train

        if self.is_train:
            self.transforms = A.Compose([
                A.Resize(300, 300),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transforms = A.Compose([
                A.Resize(300, 300),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.annotations)

    def _parse_labels(self, ann):
        boxes = []
        labels = []
        if "labels" in ann:
            for obj in ann["labels"]:
                if "box2d" not in obj:
                    continue
                b = obj["box2d"]
                if obj.get("category") in ["car", "truck", "bus", "person", "rider", "bike", "motor"]:
                    boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                    labels.append(1)
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        return boxes, labels

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        fname = ann["name"]
        img_path = os.path.join(self.img_root, "100k", self.split, fname)
        try:
            if os.path.exists(img_path):
                image = np.array(Image.open(img_path).convert("RGB"))
                boxes, labels = self._parse_labels(ann)
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image_t = transformed["image"]
                boxes_t = transformed["bboxes"]
                labels_t = transformed["labels"]
                target = {
                    "boxes": torch.tensor(boxes_t, dtype=torch.float32),
                    "labels": torch.tensor(labels_t, dtype=torch.int64)
                }
                return image_t, target, img_path
            else:
                raise FileNotFoundError
        except Exception:
            image_t = torch.randn(3, 300, 300)
            target = {
                "boxes": torch.tensor([[0, 0, 1, 1]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64)
            }
            return image_t, target, "error"


class RealBDDAVideoDataset(Dataset):
    """Dataset BDDA video"""

    def __init__(self, root_dir, clip_length=8, max_samples=None):
        self.root_dir = root_dir
        self.camera_dir = os.path.join(root_dir, "camera_videos")
        self.gazemap_dir = os.path.join(root_dir, "gazemap_videos")

        self.video_paths = []
        if os.path.exists(self.camera_dir):
            for fname in sorted(os.listdir(self.camera_dir)):
                if fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    self.video_paths.append(os.path.join(self.camera_dir, fname))

        if max_samples and len(self.video_paths) > max_samples:
            self.video_paths = self.video_paths[:max_samples]

        self.clip_length = clip_length
        self.transforms = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        print(f"✅ BDDA - {len(self.video_paths)} video caricati")

    def __len__(self):
        return len(self.video_paths)

    def _load_clip(self, path, num_frames):
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total = len(vr)
            if total == 0:
                raise RuntimeError("Video vuoto")
            if total < num_frames:
                idx = list(range(total))
                while len(idx) < num_frames:
                    idx.extend(idx)
                idx = idx[:num_frames]
            else:
                idx = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = vr.get_batch(idx).asnumpy()
            processed = []
            for f in frames:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                t = self.transforms(image=rgb)["image"]
                processed.append(t)
            return torch.stack(processed)
        except Exception:
            return torch.randn(num_frames, 3, 300, 300)

    def __getitem__(self, idx):
        cam_path = self.video_paths[idx]
        video_name = os.path.basename(cam_path)
        clip_rgb = self._load_clip(cam_path, self.clip_length)
        gaze_path = os.path.join(self.gazemap_dir, video_name)
        clip_gaze = self._load_clip(gaze_path, self.clip_length) if os.path.exists(gaze_path) else torch.zeros_like(clip_rgb)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64)
        }
        return clip_rgb, clip_gaze, target, cam_path


##############################################
# 12. COLLATE FUNCTIONS
##############################################

def collate_fn(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)

def collate_fn_video(batch):
    clips_rgb, clips_gazemap, targets, paths = zip(*batch)
    return list(clips_rgb), list(clips_gazemap), list(targets), list(paths)
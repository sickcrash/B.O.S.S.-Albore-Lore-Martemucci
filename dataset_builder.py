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
# Dataset - Immagini (BDD100K labels format)
##############################################

class RealBDDImageDataset(Dataset):
    """
    Dataset immagini con classi multi-categoria (macro-classi).

    Classi:
      0: background
      1: person     (person + rider)
      2: vehicle    (car + truck + bus)
      3: bike
      4: motor
    """

    # mapping category -> label
    CLASS_MAP = {
        "person": 1,
        "rider": 1,
        "car": 2,
        "truck": 2,
        "bus": 2,
        "bike": 3,
        "motor": 4,
    }

    def __init__(self, annotations_path, img_root, split: str, is_train: bool = True, max_samples=None):
        if os.path.exists(annotations_path):
            with open(annotations_path, "r") as f:
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

        bbox_params = A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0
        )

        if self.is_train:
            self.transforms = A.Compose(
                [
                    A.Resize(300, 300),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=bbox_params
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Resize(300, 300),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=bbox_params
            )

    def __len__(self):
        return len(self.annotations)

    def _parse_labels(self, ann):
        boxes = []
        labels = []

        if "labels" in ann:
            for obj in ann["labels"]:
                if "box2d" not in obj:
                    continue

                cat = obj.get("category")
                if cat not in self.CLASS_MAP:
                    continue

                b = obj["box2d"]
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

                # scarta box degeneri
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(self.CLASS_MAP[cat])

        # boxes=[], labels=[]
        return boxes, labels

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        fname = ann["name"]
        img_path = os.path.join(self.img_root, self.split, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Immagine non trovata: {img_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        boxes, labels = self._parse_labels(ann)

        # Albumentations con bbox_params accetta anche liste vuote:
        transformed = self.transforms(image=image, bboxes=boxes, labels=labels)

        image_t = transformed["image"]
        boxes_t = transformed["bboxes"]
        labels_t = transformed["labels"]

        # Garantiamo shape corretta anche per empty
        if len(boxes_t) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes_t, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_t, dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return image_t, target, img_path


##############################################
# Dataset - Video (resta invariato: no labels)
##############################################

class RealBDDAVideoDataset(Dataset):
    """Dataset BDDA video (clip RGB + gazemap). Nessuna annotazione box."""
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

            frames = vr.get_batch(idx).asnumpy()  # [T,H,W,C]
            processed = []
            for f in frames:
                # decord spesso già in RGB, ma lasciamo conversione robusta
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                t = self.transforms(image=rgb)["image"]
                processed.append(t)

            return torch.stack(processed)  # [T,C,H,W]
        except Exception as e:
            print(f"❌ Errore caricamento clip: {e}")
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
# Collate functions
##############################################

def collate_fn(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)

def collate_fn_video(batch):
    collated = []
    for clip_rgb, clip_gazemap, target, path in batch:
        collated.append({
            "clip_rgb": clip_rgb,
            "clip_gazemap": clip_gazemap,
            "target": target,
            "path": path
        })
    return collated

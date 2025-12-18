import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
from torchvision import transforms
import os
import albumentations as A


class DatasetAdapter(Dataset):

    def __init__(self, json_file, img_root, target_size=(512, 512), domain_label=0, max_objects=10,
                 enable_transformation=True):
        """
        Args:
            json_file: Path al file .json (KITTI o SUNRGBD)
            img_root: Path alla cartella radice delle immagini
            input_size: Tuple (height, width) per il resize
            domain_label: 0 per Indoor, 1 per Outdoor (servirà per la Strategy)
        """
        self.enable_transformation = enable_transformation
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.img_root = img_root
        self.domain_label = domain_label
        self.images = self.data['images']

        # Mappa image_id -> annotations
        self.img_to_anns = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.images = []
        print(f"Caricamento {os.path.basename(json_file)}...")
        for img_info in self.data['images']:
            img_path = os.path.join(self.img_root, img_info['file_path'])
            if os.path.exists(img_path):
                self.images.append(img_info)

        print(f"Immagini trovate: {len(self.images)} su {len(self.data['images'])} nel JSON.")

        if len(self.images) == 0:
            raise FileNotFoundError(f"Nessuna immagine trovata in {img_root}. Controlla i percorsi nel JSON!")

        # Trasformazione base per l'immagine
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Data Augmentation con albumentations, p indica la probabiltà di applicazione delle trasformazioni
        self.randomized_trans = A.Compose([
            A.OneOf([
                A.GaussianBlur(
                    sigma_limit=(0.2, 0.5),
                    blur_limit=(0, 0),
                    p=0.3
                )
            ]),
            A.RandomBrightnessContrast(p=0.5),
        ])

        self.target_size = target_size
        self.target_h, self.target_w = target_size
        self.max_objects = max_objects

        self.class_map = {
            # Outdoor
            'car': 0, 'van': 0, 'truck': 0,  # Raggruppiamo i veicoli
            'pedestrian': 1, 'person': 1,  # Raggruppiamo le persone
            'cyclist': 2,

            # Indoor
            'chair': 3,
            'table': 4,
            'desk': 4,  # Tavoli e scrivanie insieme
            'bed': 5,
            'sofa': 6,
            'bookshelf': 7
        }

        self.num_classes = 8

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_info = self.images[idx]
        img_path = os.path.join(self.img_root, img_info['file_path'])

        # immagine originale
        raw_img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = raw_img.size

        # letterboxing (resize + padding)
        target_h, target_w = self.target_size  # (512, 512)

        # fattore di scala
        scale = min(target_w / w_orig, target_h / h_orig)

        # dimensioni dell'immagine scalata
        nw = int(w_orig * scale)
        nh = int(h_orig * scale)

        # Resize dell'immagine
        img_resized = raw_img.resize((nw, nh), Image.BILINEAR)

        # canvas finale (grigio 128 per il padding) di dimensione 512x512
        final_img = Image.new("RGB", (target_w, target_h), (128, 128, 128))

        # Incolliamo l'immagine ridimensionata in alto a sinistra (0, 0)
        final_img.paste(img_resized, (0, 0))

        img_tensor = self.transform(final_img)

        # trasformazioni casuali
        if self.enable_transformation:
            trans = self.randomized_trans(image=img_tensor.numpy())
            trans = torch.from_numpy(trans['image'])  # Riconversione a tensore
        else:
            trans = img_tensor

        # La matrice originale è per l'immagine W_orig x H_orig.
        # Noi abbiamo scalato l'immagine di un fattore 'scale'.
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        K_orig = np.array(img_info['K'])
        K_new = K_orig.copy()

        # Scaliamo le lunghezze focali e i centri ottici
        K_new[0, 0] *= scale  # fx
        K_new[1, 1] *= scale  # fy
        K_new[0, 2] *= scale  # cx
        K_new[1, 2] *= scale  # cy

        # target 3D
        anns = self.img_to_anns.get(img_info['id'], [])

        # Filtriamo oggetti validi (con 3D)
        valid_anns = [a for a in anns if 'center_cam' in a and 'dimensions' in a]

        # target matrix: [MAX_OBJ, 6] (cx, cy, cz, w, h, d) per POSIZIONE e DIMENSIONI oggetto
        target_matrix = torch.zeros((self.max_objects, 6), dtype=torch.float32)

        # Vettore Classi: [MAX_OBJ] (Tipo Long/Int per CrossEntropy) per CLASSIFICAZIONE
        target_cls = torch.full((self.max_objects,), -1, dtype=torch.long)

        # Maschera: [MAX_OBJ]
        mask = torch.zeros((self.max_objects), dtype=torch.float32)

        num_objs = 0
        for ann in valid_anns:
            if num_objs >= self.max_objects:
                break

            # nome categoria
            cat_name = ann.get('category_name', 'unknown')

            if cat_name in self.class_map:
                class_id = self.class_map[cat_name]

                target_cls[num_objs] = class_id

                vals = ann['center_cam'] + ann['dimensions']
                target_matrix[num_objs] = torch.tensor(vals, dtype=torch.float32)

                mask[num_objs] = 1.0

                num_objs += 1

        return {
            "image": trans,  # [3, 512, 512]
            "K": torch.tensor(K_new).float(),  # [3, 3]

            "target_3d": target_matrix,  # [MAX_OBJ, 6] (Regressione)
            "target_cls": target_cls,  # [MAX_OBJ]    (Classificazione)
            "mask": mask,  # [MAX_OBJ]    (Validità c'è oggetto?)

            "num_objs": torch.tensor(num_objs),
            "domain": torch.tensor(self.domain_label).long()
        }



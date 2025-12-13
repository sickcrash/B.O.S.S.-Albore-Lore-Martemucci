import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from adapter import DatasetAdapter
from torch.utils.data import DataLoader

class Omni3DViewer:
    def __init__(self, json_path, images_root_dir):

        print(f"Caricamento annotazioni da {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # indice image_id -> annotations
        self.img_id_to_anns = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        self.images_root = images_root_dir
        print(f"Dataset caricato: {len(self.data['images'])} immagini.")

    def show_sequence(self, limit=10):
        """Mostra le prime 'limit' immagini in sequenza"""

        # Prende solo i primi N elementi dalla lista delle immagini
        images_to_show = self.data['images'][:limit]

        print(f"Sto per visualizzare le prime {len(images_to_show)} immagini...")

        for i, img_info in enumerate(images_to_show):
            img_id = img_info['id']
            rel_path = img_info['file_path']
            full_path = os.path.join(self.images_root, rel_path)

            if not os.path.exists(full_path):
                print(f"[SKIP] Immagine {i + 1}/{limit} non trovata: {full_path}")
                continue

            try:
                # Carica immagine
                image = Image.open(full_path)
                anns = self.img_id_to_anns.get(img_id, [])

                # Visualizza
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(image)
                ax.set_title(f"Image #{i + 1} | ID: {img_id} | File: {os.path.basename(rel_path)}")

                # Disegna i box
                for ann in anns:
                    # Omni3D usa [x_min, y_min, x_max, y_max] per 'bbox2D_tight'
                    if 'bbox2D_tight' in ann:
                        box = ann['bbox2D_tight']
                        cat_name = ann.get('category_name', 'Obj')

                        # Conversione per Matplotlib (x, y, w, h)
                        x, y = box[0], box[1]
                        w = box[2] - box[0]
                        h = box[3] - box[1]

                        # Rettangolo
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x, y - 5, cat_name, color='white', fontsize=9, backgroundcolor='red')

                plt.axis('off')
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Errore caricando l'immagine {full_path}: {e}")

PATH_TO_IMAGES = "datasets"

print("\n--- VISUALIZZAZIONE KITTI ---")
kitti_viewer = Omni3DViewer('datasets/KITTI_train.json', PATH_TO_IMAGES)
kitti_viewer.show_sequence(limit=10)

print("\n--- VISUALIZZAZIONE SUNRGBD ---")
sun_viewer = Omni3DViewer('datasets/SUNRGBD_train.json', PATH_TO_IMAGES)
sun_viewer.show_sequence(limit=10)


kitti_ds = DatasetAdapter('datasets/KITTI_train.json', PATH_TO_IMAGES, target_size=(512, 512), domain_label=1,
                          max_objects=10)
loader = DataLoader(kitti_ds, batch_size=4, shuffle=True)

# Test
batch = next(iter(loader))
print("Image shape:", batch['image'].shape) # Deve essere [batch_size, 3, target_size, target_size]
print("K shape:", batch['K'].shape)         # Deve essere [batch_size, 3, 3]
print("Target shape:", batch['target_3d'].shape) # Deve essere [batch_size, max_objects, 6]
print("Domain:", batch['domain'])           # Deve coincidere con domain_label (0 o 1)
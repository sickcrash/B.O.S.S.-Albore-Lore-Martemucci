import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
import timm
import os


class ModelAdapter:

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.name = "Generic_Model"
        self.device = "cpu"

    def load_model(self, device):
        raise NotImplementedError

    def predict(self, image_path):
        raise NotImplementedError


class ResNetAdapter(ModelAdapter):
    def __init__(self, checkpoint_path):
        super().__init__(checkpoint_path)
        self.name = "DeepLabV3_ResNet50"

    def load_model(self, device):
        self.device = device
        print(f"[{self.name}] Configurazione architettura...")
        #Ricostruzione architettura
        self.model = deeplabv3_resnet50(weights=None, aux_loss=True)
        #Modifica per 2 classi (0 per sfondo/1 per ostacolo)
        self.model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

        #Caricamento dei pesi
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=device)

                #Gestione dei vari formati di salvataggio
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    weights = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    weights = checkpoint['state_dict']
                else:
                    weights = checkpoint

                #strict=False ignora l'aux_classifier se non commbacia
                self.model.load_state_dict(weights, strict=False)
                print(f"[{self.name}] Pesi caricati: {self.checkpoint_path}")
            except Exception as e:
                print(f"[{self.name}] ERRORE CRITICO caricamento pesi: {e}")
        else:
            print(f"[{self.name}] ATTENZIONE: File pesi non trovato o non fornito. Uso pesi casuali.")

        self.model.to(device)
        self.model.eval()

    def predict(self, image_path):
        #Caricamento
        img = Image.open(image_path).convert("RGB")

        #Resize e normalizzazione
        img_resized = F.resize(img, (512, 1024))
        img_t = F.to_tensor(img_resized)
        img_t = F.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_t.unsqueeze(0).to(self.device)

        #Inferenza con calcolo probabilità
        with torch.no_grad():
            output = self.model(img_tensor)['out']

            #Applichiamo softmax per avere percentuali da 0.0 a 1.0
            probs = torch.nn.functional.softmax(output, dim=1)

            #Prendiamo la maschera binaria
            prediction = probs.argmax(1).cpu().numpy()[0]  #Matrice di 0 e 1

            #Estraiamo la mappa di probabilità dell'ostacolo
            obstacle_prob_map = probs[0, 1, :, :].cpu().numpy()

            #Calcolo dello score se c'è almeno un pixel che ha rilevato l'ostacolo
            if prediction.max() > 0:
                has_obstacle = 1
                #consideriamo solo i pixel che hanno rilevato ostacoli e facciamo la media delle loro probabilità
                obstacle_pixels = obstacle_prob_map[prediction == 1]
                score = float(obstacle_pixels.mean())
            else:
                has_obstacle = 0
                score = 0.0

        return {"class_id": has_obstacle, "score": score}


class EfficientNetAdapter(ModelAdapter):
    def __init__(self, checkpoint_path, arch='efficientnet_b0', num_classes=2):
        super().__init__(checkpoint_path)
        self.name = f"EfficientNet_{arch}"
        self.arch = arch
        self.num_classes = num_classes

    def load_model(self, device):
        self.device = device
        print(f"[{self.name}] Configurazione architettura...")

        #Creazione Modello con Timm
        #Usiamo pretrained=False perché carichiamo i pesi specifici
        self.model = timm.create_model(self.arch, pretrained=False, num_classes=self.num_classes)

        #Caricamento pesi
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=device)

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

                print(f"[{self.name}] Pesi caricati: {self.checkpoint_path}")
            except Exception as e:
                print(f"[{self.name}] ERRORE caricamento pesi: {e}. Uso modello non allenato.")
        else:
            print(f"[{self.name}] ATTENZIONE: File pesi non trovato. Uso modello non allenato.")

        self.model.to(device)
        self.model.eval()

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")

        img_resized = F.resize(img, (224, 224))
        img_t = F.to_tensor(img_resized)
        img_t = F.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        input_tensor = img_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            #Output sono i logits, usiamo softmax per avere percentuali
            probs = torch.softmax(output, dim=1)

            #Prende la classe con la probabilità più alta
            score, class_idx = torch.max(probs, 1)

        return {"class_id": class_idx.item(), "score": score.item()}
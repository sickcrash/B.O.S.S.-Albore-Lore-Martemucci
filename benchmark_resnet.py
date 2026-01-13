import time
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as F
from PIL import Image
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np

#Configurazione
CHECKPOINT_PATH = "./checkpoints/resnet.pth"
#Immagine di prova
TEST_IMAGE_PATH = "test_image.jpg"
#Usa la GPU se disponibile, altrimenti CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"Sto ricostruendo l'architettura ResNet (DeepLabV3)...")

    #Ricostruiamo l'architettura come in main
    #Nel main viene usato aux_loss=True, quindi dobbiamo metterlo anche qui
    #altrimenti le chiavi del dizionario dei pesi non combaciano.
    model = deeplabv3_resnet50(weights=None, aux_loss=True)

    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

    #Carichiamo i pesi
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Trovato file pesi: {CHECKPOINT_PATH}")
        try:
            #Carichiamo il file .pth
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

            #A seconda di come è stato salvato, i pesi potrebbero essere direttamente nel file o dentro una chiave 'model_state_dict'
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                weights = checkpoint['model_state_dict']
            else:
                weights = checkpoint

            #Carichiamo i pesi nel modello strict=False serve perché nel main non viene modificato l'aux_classifier
            #per le 2 classi, quindi l'aux_classifier nel file .pth avrà 21 classi (quelle di COCO)
            model.load_state_dict(weights, strict=False)
            print(">>> Pesi caricati con successo!")

        except Exception as e:
            print(f"\nERRORE nel caricamento dei pesi: {e}")
            return None
    else:
        print(f"ERRORE: Il file {CHECKPOINT_PATH} non esiste. Controlla il nome o la cartella.")
        return None

    model.to(DEVICE)
    model.eval() #Imposta la modalità valutazione
    return model

def run_benchmark():
    #Misurazione RAM Iniziale
    process = psutil.Process(os.getpid())
    ram_start = process.memory_info().rss / (1024 ** 2) # Convertiamo in MB

    #Caricamento Modello
    model = load_model()
    if model is None: return

    #Calcoliamo quanta RAM ha utilizzato il modello
    ram_model = (process.memory_info().rss / (1024 ** 2)) - ram_start

    #Preparazione Immagine
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"ERRORE: Manca l'immagine {TEST_IMAGE_PATH}")
        return

    print(f"Elaborazione immagine: {TEST_IMAGE_PATH}")
    img_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")

    #Resize e normalizzazione uguali a dataset
    #dataset usa img_size=(512, 1024)
    img_tensor = F.resize(img_pil, (512, 1024), interpolation=F.InterpolationMode.BILINEAR)
    img_t = F.to_tensor(img_tensor)
    #Normalizzazione standard ImageNet usata nel codice
    img_t = F.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #Aggiungiamo la dimensione del batch (da [3, H, W] a [1, 3, H, W])
    input_batch = img_t.unsqueeze(0).to(DEVICE)

    #Warm up
    with torch.no_grad():
        _ = model(input_batch)

    #Benchmark velocità (Inferenza)
    print("Avvio misurazione latenza...")
    start_time = time.time()

    with torch.no_grad():
        #Eseguiamo il modello
        output = model(input_batch)["out"]
        #Argmax trasforma le probabilità in classi (0 o 1)
        #Prende la classe con probabilità più alta per ogni pixel
        prediction = output.argmax(1).cpu().numpy()[0]

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    #Stampa Report Finale
    print("\n" + "="*40)
    print(f" REPORT PRESTAZIONI: ResNet50 (DeepLabV3)")
    print("="*40)
    print(f"Latenza (Tempo di risposta): {latency_ms:.2f} ms")
    print(f"RAM Occupata dal modello:    {ram_model:.2f} MB")
    print(f"Dispositivo usato:           {DEVICE}")
    print("="*40)

    #Visualizzazione Grafica
    plt.figure(figsize=(15, 6))

    #Immagine originale
    plt.subplot(1, 3, 1)
    plt.imshow(img_pil)
    plt.title("Input Originale")
    plt.axis('off')

    #Maschera predetta (Bianco e Nero)
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray', vmin=0, vmax=1)
    plt.title("Maschera Ostacoli")
    plt.axis('off')

    #Sovrapposizione (Overlay)
    plt.subplot(1, 3, 3)
    #Ridimensioniamo la maschera alle dimensioni originali per sovrapporla bene
    mask_resized = Image.fromarray((prediction * 255).astype(np.uint8)).resize(img_pil.size, resample=Image.NEAREST)
    plt.imshow(img_pil)
    plt.imshow(mask_resized, alpha=0.5, cmap='jet') # Alpha = trasparenza
    plt.title("Sovrapposizione")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()
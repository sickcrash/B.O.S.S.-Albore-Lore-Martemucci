import torch
import os
from torch.utils.data import DataLoader, ConcatDataset

from adapter import DatasetAdapter
from model.detector import UnifiedDetector
from training.loss import HungarianMatcher
from training.train import train_model
from training.evaluator import Evaluator
import warnings
warnings.filterwarnings("ignore")


def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Esecuzione su device: {device}")


    PATH_TO_IMAGES = "./datasets/mini_dataset"
    CHECKPOINT_PATH = "best_model.pth"
    FINAL_MODEL_PATH = "final_model.pth"

    TRAIN_MODE = False # true per allenare, false per solo valutazione
    OUT_DOMAIN = None # Se "kitti" o "sunrgbd", valuta un dominio alla volta per valutare degrado fuori dominio
    NUM_EPOCHS = 1
    BATCH_SIZE = 4

    print("--> Loading Datasets...")

    # --- TRAIN & VAL (Dataset Misti) ---
    kitti_train = DatasetAdapter(f'{PATH_TO_IMAGES}/KITTI_train.json', PATH_TO_IMAGES, target_size=(512, 512),
                                 domain_label=1, max_objects=10)
    sun_train = DatasetAdapter(f'{PATH_TO_IMAGES}/SUNRGBD_train.json', PATH_TO_IMAGES, target_size=(512, 512),
                               domain_label=0, max_objects=10)
    combined_train = ConcatDataset([kitti_train, sun_train])

    kitti_val = DatasetAdapter(f'{PATH_TO_IMAGES}/KITTI_val.json', PATH_TO_IMAGES, target_size=(512, 512),
                               domain_label=1, max_objects=10, enable_transformation=False)
    sun_val = DatasetAdapter(f'{PATH_TO_IMAGES}/SUNRGBD_val.json', PATH_TO_IMAGES, target_size=(512, 512),
                             domain_label=0, max_objects=10, enable_transformation=False)
    combined_val = ConcatDataset([kitti_val, sun_val])

    loader_train = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loader_val = DataLoader(combined_val, batch_size=BATCH_SIZE, shuffle=False)

    loader_train_kitti = DataLoader(kitti_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loader_val_kitti = DataLoader(kitti_val, batch_size=BATCH_SIZE, shuffle=False)
    loader_train_sun = DataLoader(sun_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loader_val_sun = DataLoader(sun_val, batch_size=BATCH_SIZE, shuffle=False)

    # --- TEST (Separati per Ablation Study) ---
    kitti_test = DatasetAdapter(f'{PATH_TO_IMAGES}/KITTI_test.json', PATH_TO_IMAGES, target_size=(512, 512),
                                domain_label=1, max_objects=10, enable_transformation=False)
    sun_test = DatasetAdapter(f'{PATH_TO_IMAGES}/SUNRGBD_test.json', PATH_TO_IMAGES, target_size=(512, 512),
                              domain_label=0, max_objects=10, enable_transformation=False)

    loader_test_kitti = DataLoader(kitti_test, batch_size=BATCH_SIZE, shuffle=False)
    loader_test_sun = DataLoader(sun_test, batch_size=BATCH_SIZE, shuffle=False)

    combined_test = ConcatDataset([kitti_test, sun_test])
    loader_test_mixed = DataLoader(combined_test, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # INIZIALIZZAZIONE MODELLO
    # ==========================================
    detector = UnifiedDetector().to(device)
    matcher = HungarianMatcher()

    # ==========================================
    # GESTIONE CHECKPOINT
    # ==========================================
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--> Trovato checkpoint: {CHECKPOINT_PATH}. Caricamento...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

            model_state = model.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_state and model_state[k].shape == v.shape}

            model.load_state_dict(filtered, strict=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                detector.load_state_dict(checkpoint['model_state_dict'])
            else:
                detector.load_state_dict(checkpoint)
            print("--> Pesi caricati con successo.")
        except Exception as e:
            pass
    else:
        print("--> Nessun checkpoint trovato. Modello inizializzato random.")

    # ==========================================
    # FASE DI TRAINING
    # ==========================================
    if TRAIN_MODE:
        print("\n" + "=" * 40)
        print("       AVVIO TRAINING LOOP")
        print("=" * 40)
        if OUT_DOMAIN in ["kitti", "sunrgbd"]:
            print(f"Allenamento solo su dominio: {OUT_DOMAIN}")
            if OUT_DOMAIN == "kitti":
                train_model(detector, matcher, loader_train_kitti, loader_val_kitti, n_epochs=NUM_EPOCHS)
            else:
                train_model(detector, matcher, loader_train_sun, loader_val_sun, n_epochs=NUM_EPOCHS)
        else:
            train_model(detector, matcher, loader_train, loader_val, n_epochs=NUM_EPOCHS)
        torch.save(detector.state_dict(), FINAL_MODEL_PATH)
        print(f"Modello finale salvato in: {FINAL_MODEL_PATH}")

    # ==========================================
    # FASE DI VALUTAZIONE
    # ==========================================
    print("\n" + "=" * 40)
    print("       AVVIO EVALUATION      ")
    print("=" * 40)

    evaluator = Evaluator(model=detector, device=device, num_classes=8, iou_threshold=0.1)

    # [A] Mixed Domain
    metrics_mixed = evaluator.run_evaluation(loader_test_mixed, desc="Mixed Eval")

    # [B] Outdoor (KITTI)
    metrics_kitti = evaluator.run_evaluation(loader_test_kitti, desc="KITTI Eval")

    # [C] Indoor (SUNRGBD)
    metrics_sun = evaluator.run_evaluation(loader_test_sun, desc="SUNRGBD Eval")

    # ==========================================
    # REPORT E GRAFICI
    # ==========================================
    print("\n" + "#" * 50)
    print("           REPORT FINALE D'ESAME           ")
    print("#" * 50)
    print(f"{'TEST SET':<15} | {'mAP (IoU)':<15} | {'DOMAIN ACC':<15}")
    print("-" * 50)
    print(f"{'Misto':<15} | {metrics_mixed['mAP']:.4f}{'':<9} | {metrics_mixed['domain_acc']:.2%}")
    print(f"{'KITTI (Out)':<15} | {metrics_kitti['mAP']:.4f}{'':<9} | {metrics_kitti['domain_acc']:.2%}")
    print(f"{'SUN (In)':<15} | {metrics_sun['mAP']:.4f}{'':<9} | {metrics_sun['domain_acc']:.2%}")
    print("-" * 50)

    # domain gap
    gap = abs(metrics_kitti['mAP'] - metrics_sun['mAP'])
    print(f"\nDomain Gap (Delta mAP): {gap:.4f}")

if __name__ == "__main__":
    run_pipeline()
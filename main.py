import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset

from adapter import DatasetAdapter
from model.detector import UnifiedDetector
from training.loss import HungarianMatcher
from training.train import train_model
from training.evaluator import Evaluator


def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Esecuzione su device: {device}")


    PATH_TO_IMAGES = "datasets/mini_dataset"
    CHECKPOINT_PATH = "best_model.pth"

    # true per allenare, false per solo valutare
    TRAIN_MODE = False

    print("--> Loading Datasets...")

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

    loader_train = DataLoader(combined_train, batch_size=4, shuffle=True, num_workers=2)
    loader_val = DataLoader(combined_val, batch_size=4, shuffle=False)

    kitti_test = DatasetAdapter(f'{PATH_TO_IMAGES}/KITTI_test.json', PATH_TO_IMAGES, target_size=(512, 512),
                                domain_label=1, max_objects=10, enable_transformation=False)
    sun_test = DatasetAdapter(f'{PATH_TO_IMAGES}/SUNRGBD_test.json', PATH_TO_IMAGES, target_size=(512, 512),
                              domain_label=0, max_objects=10, enable_transformation=False)

    loader_test_kitti = DataLoader(kitti_test, batch_size=4, shuffle=False)
    loader_test_sun = DataLoader(sun_test, batch_size=4, shuffle=False)

    combined_test = ConcatDataset([kitti_test, sun_test])
    loader_test_mixed = DataLoader(combined_test, batch_size=4, shuffle=False)


    detector = UnifiedDetector().to(device)
    matcher = HungarianMatcher()

    # checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--> Trovato checkpoint: {CHECKPOINT_PATH}. Caricamento...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                detector.load_state_dict(checkpoint['model_state_dict'])
            else:
                detector.load_state_dict(checkpoint)
            print("--> Pesi caricati con successo.")
        except Exception as e:
            print(f"ATTENZIONE: Errore nel caricamento dei pesi: {e}")
    else:
        print("--> Nessun checkpoint trovato. Modello inizializzato random.")

    # training
    if TRAIN_MODE:
        print("\n" + "=" * 40)
        print("       AVVIO TRAINING")
        print("=" * 40)
        train_model(detector, matcher, loader_train, loader_val, n_epochs=10)
        torch.save(detector.state_dict(), "best_model_retrained.pth")
        print("Training completato e modello salvato.")


    print("\n" + "=" * 40)
    print("       AVVIO EVALUATION")
    print("=" * 40)
    # evaluation
    evaluator = Evaluator(model=detector, device=device, num_classes=8, iou_threshold=0.1)

    # [A] Mixed Domain
    metrics_mixed = evaluator.run_evaluation(loader_test_mixed, desc="Mixed Eval")

    # [B] Outdoor (KITTI)
    metrics_kitti = evaluator.run_evaluation(loader_test_kitti, desc="KITTI Eval")

    # [C] Indoor (SUNRGBD)
    metrics_sun = evaluator.run_evaluation(loader_test_sun, desc="SUNRGBD Eval")

    # report
    print("\n" + "#" * 50)
    print("           REPORT FINALE           ")
    print("#" * 50)
    print(f"{'TEST SET':<15} | {'mAP (IoU=.25)':<15} | {'DOMAIN ACC':<15}")
    print("-" * 50)
    print(f"{'Misto':<15} | {metrics_mixed['mAP']:.4f}{'':<9} | {metrics_mixed['domain_acc']:.2%}")
    print(f"{'KITTI (Out)':<15} | {metrics_kitti['mAP']:.4f}{'':<9} | {metrics_kitti['domain_acc']:.2%}")
    print(f"{'SUN (In)':<15} | {metrics_sun['mAP']:.4f}{'':<9} | {metrics_sun['domain_acc']:.2%}")
    print("-" * 50)

    # Analisi automatica del "Gap"
    gap = abs(metrics_kitti['mAP'] - metrics_sun['mAP'])
    print(f"\nDomain Gap (Delta mAP): {gap:.4f}")

    # confusion matrix
    targets = metrics_mixed['targets']
    preds = metrics_mixed['preds']

    if len(targets) > 0:
        unique_labels = sorted(list(set(targets + preds)))
        target_names = [evaluator.class_names.get(i, str(i)) for i in unique_labels]

        cm = confusion_matrix(targets, preds, labels=unique_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix (Mixed Domain)')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')

        plt.savefig('confusion_matrix_result.png')
        print("Grafico salvato come 'confusion_matrix_result.png'")

        plt.show()
    else:
        print("Non abbastanza dati rilevati per generare la matrice di confusione.")


if __name__ == "__main__":
    run_pipeline()
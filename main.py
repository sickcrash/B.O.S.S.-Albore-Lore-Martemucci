
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model_builder import EnhancedTemporalSSD, ModelExporter
from dataset_builder import RealBDDImageDataset, RealBDDAVideoDataset, collate_fn, collate_fn_video
from trainer_net import CompleteTrainingPipeline, AblationStudy
from tester_net import CompleteTestingPipeline


##############################################
# 14. MAIN COMPLETE - VERSIONE FINALE SEMPLIFICATA
##############################################

def main_complete_real():
    print("=" * 80)
    print("🚀 PIPELINE COMPLETA REALE - TRAIN, VALIDATION, TEST, ABLATION, EXPORT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Dispositivo: {device}")

    # PATH CONFIGURATION
    GDRIVE_ROOT = "/content/drive/MyDrive/Boss/dataset"
    BDD_IMG_ROOT = f"{GDRIVE_ROOT}/bdd100k"
    BDD_LABEL_TRAIN = f"{GDRIVE_ROOT}/bdd100k_labels_release/labels/bdd100k_labels_images_train.json"
    BDD_LABEL_VAL = f"{GDRIVE_ROOT}/bdd100k_labels_release/labels/bdd100k_labels_images_val.json"
    BDD_LABEL_TEST = BDD_LABEL_VAL
    BDDA_TRAIN_ROOT = f"{GDRIVE_ROOT}/BDDA/training"
    BDDA_VAL_ROOT = f"{GDRIVE_ROOT}/BDDA/validation"
    BDDA_TEST_ROOT = f"{GDRIVE_ROOT}/BDDA/test"

    MAX_TRAIN_IMG, MAX_TRAIN_VID = 500, 50
    MAX_VAL_IMG, MAX_VAL_VID = 100, 20
    MAX_TEST_IMG, MAX_TEST_VID = 50, 10

    print("\n" + "=" * 50)
    print("📂 CARICAMENTO DATASET")
    print("=" * 50)

    train_img_dataset = RealBDDImageDataset(BDD_LABEL_TRAIN, BDD_IMG_ROOT, "train", is_train=True, max_samples=MAX_TRAIN_IMG)
    train_vid_dataset = RealBDDAVideoDataset(BDDA_TRAIN_ROOT, clip_length=8, max_samples=MAX_TRAIN_VID)
    val_img_dataset = RealBDDImageDataset(BDD_LABEL_VAL, BDD_IMG_ROOT, "val", is_train=False, max_samples=MAX_VAL_IMG)
    val_vid_dataset = RealBDDAVideoDataset(BDDA_VAL_ROOT, clip_length=8, max_samples=MAX_VAL_VID)
    test_img_dataset = RealBDDImageDataset(BDD_LABEL_TEST, BDD_IMG_ROOT, "val", is_train=False, max_samples=MAX_TEST_IMG)
    test_vid_dataset = RealBDDAVideoDataset(BDDA_TEST_ROOT, clip_length=8, max_samples=MAX_TEST_VID)

    train_img_loader = DataLoader(train_img_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)
    train_vid_loader = DataLoader(train_vid_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_video, num_workers=2)
    val_img_loader = DataLoader(val_img_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)
    val_vid_loader = DataLoader(val_vid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_video, num_workers=2)
    test_img_loader = DataLoader(test_img_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_vid_loader = DataLoader(test_vid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_video, num_workers=2)

    print(f"\n📊 STATISTICHE:")
    print(f"TRAIN: {len(train_img_dataset)} img, {len(train_vid_dataset)} vid")
    print(f"VAL: {len(val_img_dataset)} img, {len(val_vid_dataset)} vid")
    print(f"TEST: {len(test_img_dataset)} img, {len(test_vid_dataset)} vid")

    print("\n" + "=" * 50)
    print("🧪 ABLATION STUDY")
    print("=" * 50)

    ablation = AblationStudy(save_dir='logs/ablation')

    experiments = [
        ("Baseline", {"use_adapters": False, "use_temporal": False, "use_weather_aug": False}),
        ("Adapters Only", {"use_adapters": True, "use_temporal": False, "use_weather_aug": False}),
        ("Temporal Only", {"use_adapters": False, "use_temporal": True, "use_weather_aug": False}),
        ("Weather Aug Only", {"use_adapters": False, "use_temporal": False, "use_weather_aug": True}),
        ("Adapters + Temporal", {"use_adapters": True, "use_temporal": True, "use_weather_aug": False}),
        ("Adapters + Weather", {"use_adapters": True, "use_temporal": False, "use_weather_aug": True}),
        ("All Components", {"use_adapters": True, "use_temporal": True, "use_weather_aug": True}),
    ]

    for exp_name, config in experiments:
        ablation.run_experiment(exp_name, config, None, val_img_loader, device)

    ablation.save_results()

    print("\n" + "=" * 50)
    print("🧠 TRAINING MODEL (FULL CONFIGURATION)")
    print("=" * 50)

    model = EnhancedTemporalSSD(
        num_classes=2,
        device=device,
        use_adapters=True,
        use_temporal=True,
        use_weather_aug=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4
    )

    trainer = CompleteTrainingPipeline(
        model, train_img_loader, train_vid_loader,
        val_img_loader, device, use_focal_loss=True,
        val_vid_loader=val_vid_loader
    )

    EPOCHS = 2
    best_val_map = 0.0
    save_path = "/content/drive/MyDrive/Boss/complete_full_model.pth"

    for epoch in range(1, EPOCHS + 1):
        print(f"\n⭐ EPOCH {epoch}/{EPOCHS}")
        train_loss = trainer.train_epoch(optimizer, epoch)
        val_metrics = trainer.validate(epoch)

        val_temporal = trainer.validate_video_clips(epoch)

        val_map = val_metrics['mAP_avg']
        val_map_05 = val_metrics['IoU_0.5']['mAP']
        val_map_075 = val_metrics['IoU_0.75']['mAP']

        print(f"📊 Train Loss: {train_loss:.4f}")
        print(f"   Val mAP@0.5:  {val_map_05:.4f}")
        print(f"   Val mAP@0.75: {val_map_075:.4f}")
        print(f"   Val mAP (avg): {val_map:.4f}")

        if val_temporal:
            print(f" Temporal IoU: {val_temporal['temporal_iou_mean']:.4f}")
            print(f"ID Switch Rate: {val_temporal['id_switch_rate']:.4f}")

        if val_map > best_val_map:
            best_val_map = val_map
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_map": best_val_map,
            }, save_path, _use_new_zipfile_serialization=False)
            print(f"💾 Best model salvato")

    print("\n" + "=" * 50)
    print("🧪 TESTING")
    print("=" * 50)

    tester = CompleteTestingPipeline(save_path, device)
    img_results, img_metrics = tester.test_images(test_img_loader)
    vid_results, vid_temporal_metrics = tester.test_videos(test_vid_loader)

    print("\n" + "=" * 50)
    print("💾 EXPORT MODELLO")
    print("=" * 50)

    exporter = ModelExporter(model, save_dir='/content/drive/MyDrive/Boss')

    # Export SOLO ONNX
    onnx_path = exporter.export_models()

    metadata = {
        'model_type': 'SSD300_Enhanced',
        'num_classes': 2,
        'input_size': [300, 300],
        'components': {
            'domain_adapters': True,
            'temporal_consistency': True,
            'weather_augmentation': True
        },
        'export_date': str(datetime.now()),
        'best_val_map': best_val_map,
        'test_metrics': img_metrics,
        'export_status': {
            'onnx': 'success' if onnx_path else 'failed',
            'torchscript': 'disabled'
        }
    }
    exporter.save_metadata(metadata)

    print("\n" + "=" * 50)
    print("🎯 PIPELINE COMPLETATA!")
    print("=" * 50)
    print(f"✅ Modello salvato: {save_path}")
    print(f"✅ ONNX: {onnx_path}")

    print(f"\n📊 METRICHE FINALI TEST:")
    print(f"   mAP medio: {img_metrics['mAP_avg']:.4f}")
    print(f"   mAP@0.5:   {img_metrics['IoU_0.5']['mAP']:.4f}")
    print(f"   mAP@0.75:  {img_metrics['IoU_0.75']['mAP']:.4f}")


if __name__ == "__main__":
    main_complete_real()

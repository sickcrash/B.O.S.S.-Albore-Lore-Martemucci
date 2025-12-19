import os
import shutil
import yaml
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model_builder import EnhancedTemporalSSD, ModelExporter
from dataset_builder import RealBDDImageDataset, RealBDDAVideoDataset, collate_fn, collate_fn_video
from trainer_net import CompleteTrainingPipeline, AblationStudy
from tester_net import CompleteTestingPipeline


def resolve_yaml_variables(config):
    paths = config["paths"]
    dataset_root = paths["dataset_root"]
    for key, value in paths.items():
        if isinstance(value, str) and "${paths.dataset_root}" in value:
            paths[key] = value.replace("${paths.dataset_root}", dataset_root)
    return config


def backup_if_exists(path: str):
    if os.path.exists(path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(path)
        backup_path = f"{base}_backup_{ts}{ext}"
        shutil.copy2(path, backup_path)
        print(f"🧷 Backup modello precedente -> {backup_path}")


def main_complete_real():
    print("=" * 80)
    print("🚀 PIPELINE COMPLETA REALE - TRAIN, VALIDATION, TEST, ABLATION, EXPORT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Dispositivo: {device}")

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    def resolve_path(path):
        return os.path.abspath(os.path.join(PROJECT_ROOT, path))

    with open(resolve_path("config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    config = resolve_yaml_variables(config)

    BDD_IMG_ROOT = resolve_path(config["paths"]["bdd_img_root"])
    BDD_LABEL_TRAIN = resolve_path(config["paths"]["bdd_label_train"])
    BDD_LABEL_VAL = resolve_path(config["paths"]["bdd_label_val"])
    BDD_LABEL_TEST = resolve_path(config["paths"]["bdd_label_test"])
    BDDA_TRAIN_ROOT = resolve_path(config["paths"]["bdda_train_root"])
    BDDA_VAL_ROOT = resolve_path(config["paths"]["bdda_val_root"])
    BDDA_TEST_ROOT = resolve_path(config["paths"]["bdda_test_root"])
    LOGS_DIR = resolve_path(config["paths"]["logs_dir"])
    EXPORTS_DIR = resolve_path(config["paths"]["exports_dir"])
    MODEL_SAVE_PATH = resolve_path(config["paths"]["model_save_path"])

    CLASS_NAMES = {
        0: "background",
        1: "person",
        2: "vehicle",
        3: "bike",
        4: "motor",
    }
    NUM_CLASSES = len(CLASS_NAMES)


    backup_if_exists(MODEL_SAVE_PATH)

    MAX_TRAIN_IMG, MAX_TRAIN_VID = 5000, 50
    MAX_VAL_IMG, MAX_VAL_VID = 1000, 20
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
    print(f"VAL:   {len(val_img_dataset)} img, {len(val_vid_dataset)} vid")
    print(f"TEST:  {len(test_img_dataset)} img, {len(test_vid_dataset)} vid")

    print("\n" + "=" * 50)
    print("🧪 ABLATION STUDY")
    print("=" * 50)

    ablation = AblationStudy(save_dir=LOGS_DIR, num_classes=NUM_CLASSES, device=device)

    experiments = [
        ("Baseline", {"use_adapters": False, "use_temporal": False, "use_weather_aug": False}),
        ("Adapters Only", {"use_adapters": True, "use_temporal": False, "use_weather_aug": False}),
        ("Temporal Only", {"use_adapters": False, "use_temporal": True, "use_weather_aug": False}),
        ("Weather Aug Only", {"use_adapters": False, "use_temporal": False, "use_weather_aug": True}),
        ("Adapters + Temporal", {"use_adapters": True, "use_temporal": True, "use_weather_aug": False}),
        ("Adapters + Weather", {"use_adapters": True, "use_temporal": False, "use_weather_aug": True}),
        ("All Components", {"use_adapters": True, "use_temporal": True, "use_weather_aug": True}),
    ]

    for exp_name, cfg in experiments:
        ablation.run_experiment(exp_name, cfg, val_img_loader, device, num_classes=NUM_CLASSES)

    ablation.save_results()

    print("\n" + "=" * 50)
    print("🧠 TRAINING MODEL (FULL CONFIGURATION)")
    print("=" * 50)

    model = EnhancedTemporalSSD(
        num_classes=NUM_CLASSES,
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
        model=model,
        train_img_loader=train_img_loader,
        train_vid_loader=train_vid_loader,
        val_img_loader=val_img_loader,
        val_vid_loader=val_vid_loader,
        device=device,
        num_classes=NUM_CLASSES,
        use_focal_loss=True
    )

    EPOCHS = 10
    best_val_map = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n⭐ EPOCH {epoch}/{EPOCHS}")
        train_loss = trainer.train_epoch(optimizer, epoch)
        val_metrics = trainer.validate(epoch)
        val_temporal = trainer.validate_video_clips(epoch)

        val_map = float(val_metrics["mAP_avg"])
        print(f"📊 Train Loss: {train_loss:.4f}")
        print(f"   Val mAP (avg): {val_map:.4f}")

        if val_temporal:
            print(f" Temporal IoU: {val_temporal['temporal_iou_mean']:.4f}")
            print(f"ID Switch Rate: {val_temporal['id_switch_rate']:.4f}")

        # salva best
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_map": best_val_map,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                },
                MODEL_SAVE_PATH,
                _use_new_zipfile_serialization=False
            )
            print(f"💾 Best model salvato -> {MODEL_SAVE_PATH}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": EPOCHS,
            "best_val_map": best_val_map,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
        },
        MODEL_SAVE_PATH,
        _use_new_zipfile_serialization=False
    )
    print(f"💾 Last model salvato -> {MODEL_SAVE_PATH}")

    print("\n" + "=" * 50)
    print("🧪 TESTING")
    print("=" * 50)

    tester = CompleteTestingPipeline(MODEL_SAVE_PATH, device)
    img_results, img_metrics = tester.test_images(test_img_loader)
    vid_results, vid_temporal_metrics = tester.test_videos(test_vid_loader)

    print("\n" + "=" * 50)
    print("💾 EXPORT MODELLO")
    print("=" * 50)

    exporter = ModelExporter(model, save_dir=EXPORTS_DIR)
    onnx_path = exporter.export_models()

    metadata = {
        "model_type": "SSD300_Enhanced",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "input_size": [300, 300],
        "components": {
            "domain_adapters": True,
            "temporal_consistency": True,
            "weather_augmentation": True
        },
        "export_date": str(datetime.now()),
        "best_val_map": best_val_map,
        "test_metrics": img_metrics,
        "export_status": {
            "onnx": "success" if onnx_path else "failed",
            "torchscript": "disabled"
        }
    }
    exporter.save_metadata(metadata)

    print("\n" + "=" * 50)
    print("🎯 PIPELINE COMPLETATA!")
    print("=" * 50)
    print(f"✅ Modello salvato: {MODEL_SAVE_PATH}")
    print(f"✅ ONNX: {onnx_path}")


if __name__ == "__main__":
    main_complete_real()

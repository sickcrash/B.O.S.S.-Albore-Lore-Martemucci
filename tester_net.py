import torch
from tqdm import tqdm
import numpy as np
import os
import yaml

from model_builder import (
    EnhancedTemporalSSD,
    DetectionMetricsCalculator,
    TemporalMetrics
)

class CompleteTestingPipeline:
    """Pipeline testing con metriche complete - COERENTE CON num_classes"""

    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if model_path is None:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            model_path = config["paths"]["model_save_path"]

        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model_path = model_path

        # --- 1) load checkpoint (se esiste) per dedurre num_classes ---
        checkpoint = None
        if model_path and os.path.exists(model_path):
            print(f"📦 Checkpoint trovato: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"❌ Errore torch.load: {e}")
                checkpoint = None
        else:
            print("⚠️ Modello non trovato, userò pesi non addestrati")

        self.num_classes = 2
        self.class_names = None
        if isinstance(checkpoint, dict):
            if "num_classes" in checkpoint:
                self.num_classes = int(checkpoint["num_classes"])
            if "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]

        # --- 2) crea modello col num_classes corretto ---
        self.model = EnhancedTemporalSSD(
            num_classes=self.num_classes,
            device=self.device,
            use_adapters=True,
            use_temporal=True,
            use_weather_aug=False
        ).to(self.device)

        # --- 3) carica pesi (se disponibili) ---
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                print(f"✅ Modello caricato (num_classes={self.num_classes})")
            except Exception as e:
                print(f"❌ Errore caricamento modello: {e}")
                print("⚠️ Uso pesi non addestrati")
        else:
            if checkpoint is not None:
                print("⚠️ Struttura checkpoint non riconosciuta, uso pesi non addestrati")

        self.model.eval()

        self.metrics_calc = DetectionMetricsCalculator(num_classes=self.num_classes)

    def test_images(self, test_img_loader, confidence_threshold=0.25):
        print("🧪 TESTING SU IMMAGINI...")
        self.model.eval()

        all_detections = []
        all_ground_truth = []

        with torch.no_grad():
            for images, targets, paths in tqdm(test_img_loader, desc="Testing immagini"):
                try:
                    images = [img.to(self.device) for img in images]
                    outputs = self.model(images)

                    for output, target, path in zip(outputs, targets, paths):
                        # predictions
                        for i in range(len(output["boxes"])):
                            if float(output["scores"][i]) >= confidence_threshold:
                                all_detections.append({
                                    "image_path": path,
                                    "bbox": output["boxes"][i].cpu().numpy().tolist(),
                                    "score": float(output["scores"][i].cpu().item()),
                                    "label": int(output["labels"][i].cpu().item())
                                })

                        # GT
                        gt_boxes = target["boxes"].cpu().numpy()
                        gt_labels = target["labels"].cpu().numpy()
                        for i in range(len(gt_boxes)):
                            all_ground_truth.append({
                                "bbox": gt_boxes[i].tolist(),
                                "label": int(gt_labels[i])
                            })

                except Exception as e:
                    print(f"❌ Errore: {e}")
                    continue

        metrics = self.metrics_calc.compute_metrics(all_detections, all_ground_truth)

        print(f"\n📊 METRICHE IMMAGINI:")
        print(f"   mAP medio (tutti IoU): {metrics['mAP_avg']:.4f}")
        for iou_thresh in self.metrics_calc.iou_thresholds:
            iou_key = f"IoU_{iou_thresh}"
            print(f"\n   IoU @ {iou_thresh}:")
            print(f"      mAP: {metrics[iou_key]['mAP']:.4f}")

        return all_detections, metrics

    def test_videos(self, test_vid_loader, confidence_threshold=0.25):
        print("🎥 TESTING SU VIDEO CON METRICHE TEMPORALI...")
        self.model.eval()

        all_video_detections = {}
        all_temporal_metrics = []
        temporal_calc = TemporalMetrics()

        with torch.no_grad():
            for batch in tqdm(test_vid_loader, desc="Testing video"):
                for sample in batch:
                    clip_rgb = sample["clip_rgb"]
                    video_path = sample["path"]

                    try:
                        images = [frame.to(self.device) for frame in clip_rgb]
                        detections_clip = self.model.forward_video_clip(images)

                        if not isinstance(detections_clip, list):
                            continue

                        temporal_metrics = temporal_calc.compute_all_metrics(detections_clip)
                        all_temporal_metrics.append(temporal_metrics)

                        video_detections = []
                        for frame_idx, frame_dets in enumerate(detections_clip):
                            for det in frame_dets:
                                score = float(det.get("score", 0.0))
                                if score >= confidence_threshold:
                                    video_detections.append({
                                        "frame_id": frame_idx,
                                        "bbox": det.get("bbox"),
                                        "score": score,
                                        "label": det.get("label"),
                                        "track_id": det.get("track_id", -1)
                                    })

                        video_name = os.path.basename(video_path)
                        all_video_detections[video_name] = video_detections

                    except Exception as e:
                        print(f"❌ Errore durante test video {video_path}: {e}")
                        continue

        aggregated = {}
        if all_temporal_metrics:
            aggregated = {
                "id_switch_rate": float(np.mean([m["id_switches"]["id_switch_rate"] for m in all_temporal_metrics])),
                "temporal_iou_mean": float(np.mean([m["temporal_iou"]["temporal_iou_mean"] for m in all_temporal_metrics])),
                "mean_track_length": float(np.mean([m["track_consistency"]["mean_track_length"] for m in all_temporal_metrics])),
                "track_completion_rate": float(np.mean([m["track_consistency"]["track_completion_rate"] for m in all_temporal_metrics]))
            }

            print(f"\n📊 METRICHE TEMPORALI VIDEO TEST:")
            print(f"   ID Switch Rate: {aggregated['id_switch_rate']:.4f}")
            print(f"   Temporal IoU:   {aggregated['temporal_iou_mean']:.4f}")

        return all_video_detections, aggregated

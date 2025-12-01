import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import csv

from model_builder import (
    FocalLoss,
    DetectionMetricsCalculator,
    TemporalMetrics, EnhancedTemporalSSD
)

##############################################
# 8. ABLATION STUDY FRAMEWORK
##############################################

class AblationStudy:
    """Framework per ablation study sistematico"""

    def __init__(self, save_dir='logs/ablation'):
        self.save_dir = save_dir if save_dir.startswith("/content") else "/content/drive/MyDrive/Boss"
        os.makedirs(save_dir, exist_ok=True)
        self.results = []

    def run_experiment(self, experiment_name, model_config, trainer, val_loader, device):
        print(f"\n{'='*60}")
        print(f"🧪 Esperimento: {experiment_name}")
        print(f"Config: {model_config}")
        print(f"{'='*60}")
        model = EnhancedTemporalSSD(num_classes=2, device=device, **model_config).to(device)
        result = {
            'experiment': experiment_name,
            'config': model_config,
            'timestamp': str(datetime.now())
        }
        self.results.append(result)
        return result

    def save_results(self, filename='ablation_results.json'):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Risultati ablation salvati: {filepath}")
        csv_path = os.path.join(self.save_dir, filename.replace('.json', '.csv'))
        if self.results:
            keys = self.results[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"✅ Risultati CSV salvati: {csv_path}")
##############################################
# 10. COMPLETE TRAINING PIPELINE
##############################################

class CompleteTrainingPipeline:
    """Pipeline completa con metriche, ablation, export"""

    def __init__(self, model, train_img_loader, train_vid_loader,
                 val_img_loader, device, use_focal_loss=True,
                 writer_logdir='logs/complete_training',
                 val_vid_loader=None):
        self.model = model
        self.train_img_loader = train_img_loader
        self.train_vid_loader = train_vid_loader
        self.val_img_loader = val_img_loader
        self.val_vid_loader = val_vid_loader
        self.device = device
        self.writer = SummaryWriter(writer_logdir)
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss()
        self.metrics_calc = DetectionMetricsCalculator()
        self.temporal_metrics = TemporalMetrics()

    def train_epoch(self, optimizer, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Training su immagini
        if self.train_img_loader:
            for batch_idx, (images, targets, paths) in enumerate(
                    tqdm(self.train_img_loader, desc=f"Epoch {epoch} Img")):
                try:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"❌ Errore training immagini: {e}")
                    continue

        # Training su video con mini-clip temporali
        if self.train_vid_loader:
            for clips_rgb, clips_gazemap, targets, paths in tqdm(self.train_vid_loader, desc=f"Epoch {epoch} Vid"):
                try:
                    for clip in clips_rgb:
                        if clip.size(0) < 3:
                            continue
                        max_start_idx = clip.size(0) - 3
                        start_idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
                        mini_clip = clip[start_idx:start_idx+3].to(self.device)
                        for frame_idx in range(mini_clip.size(0)):
                            frame = mini_clip[frame_idx]
                            images = [frame]
                            empty_target = [{
                                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                                "labels": torch.zeros((0,), dtype=torch.int64, device=self.device)
                            }]
                            loss_dict = self.model(images, empty_target)
                            loss = sum(loss_dict.values())
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                            num_batches += 1
                except Exception as e:
                    print(f"❌ Errore training video: {e}")
                    continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        """✅ METODO VALIDATE CORRETTO - DENTRO LA CLASSE"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_ground_truth = []

        with torch.no_grad():
            if self.val_img_loader:
                for images, targets, paths in tqdm(self.val_img_loader, desc=f"Val {epoch}"):
                    try:
                        images = [img.to(self.device) for img in images]
                        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                        outputs = self.model(images)
                        for output, target in zip(outputs, targets):
                            pred_boxes = output['boxes'].cpu().numpy()
                            pred_scores = output['scores'].cpu().numpy()
                            pred_labels = output['labels'].cpu().numpy()
                            for i in range(len(pred_boxes)):
                                all_predictions.append({
                                    'bbox': pred_boxes[i].tolist(),
                                    'score': float(pred_scores[i]),
                                    'label': int(pred_labels[i])
                                })
                            gt_boxes = target['boxes'].cpu().numpy()
                            gt_labels = target['labels'].cpu().numpy()
                            for i in range(len(gt_boxes)):
                                all_ground_truth.append({
                                    'bbox': gt_boxes[i].tolist(),
                                    'label': int(gt_labels[i])
                                })
                    except Exception as e:
                        print(f"❌ Errore validation: {e}")
                        continue

        # Calcola metriche
        metrics = self.metrics_calc.compute_metrics(all_predictions, all_ground_truth)

        # Log mAP_avg (media su tutti gli IoU threshold)
        self.writer.add_scalar("Metrics/mAP_avg", metrics['mAP_avg'], epoch)

        # Log mAP per ogni IoU threshold
        for iou_thresh in self.metrics_calc.iou_thresholds:
            iou_key = f'IoU_{iou_thresh}'
            self.writer.add_scalar(f"Metrics/mAP@{iou_thresh}", metrics[iou_key]['mAP'], epoch)

            # Log Precision/Recall per classe
            for class_id, class_metrics in metrics[iou_key]['per_class'].items():
                self.writer.add_scalar(
                    f"Metrics/Precision_class{class_id}@{iou_thresh}",
                    class_metrics['Precision'],
                    epoch
                )
                self.writer.add_scalar(
                    f"Metrics/Recall_class{class_id}@{iou_thresh}",
                    class_metrics['Recall'],
                    epoch
                )

        return metrics

    def validate_video_clips(self, epoch):
        """
        ✅ NUOVO METODO: Validazione su clip video con metriche temporali
        """
        if self.val_vid_loader is None:
            print("⚠️ Nessun video loader per validazione")
            return {}

        print(f"\n🎥 Validazione clip video (epoch {epoch})...")
        self.model.eval()

        all_temporal_metrics = []

        with torch.no_grad():
            for clips_rgb, clips_gazemap, targets, paths in tqdm(
                self.val_vid_loader, desc=f"Val Video {epoch}"
            ):
                for clip_rgb, video_path in zip(clips_rgb, paths):
                    try:
                        # Forward pass sul clip
                        clip_detections = self.model(clip_rgb)

                        # Calcola metriche temporali
                        metrics = self.temporal_metrics.compute_all_metrics(clip_detections)
                        all_temporal_metrics.append(metrics)

                    except Exception as e:
                        print(f"❌ Errore validazione video: {e}")
                        continue

        if not all_temporal_metrics:
            return {}

        # Aggrega metriche
        aggregated = {
            'id_switch_rate': np.mean([
                m['id_switches']['id_switch_rate']
                for m in all_temporal_metrics
            ]),
            'temporal_iou_mean': np.mean([
                m['temporal_iou']['temporal_iou_mean']
                for m in all_temporal_metrics
            ]),
            'mean_track_length': np.mean([
                m['track_consistency']['mean_track_length']
                for m in all_temporal_metrics
            ]),
            'track_completion_rate': np.mean([
                m['track_consistency']['track_completion_rate']
                for m in all_temporal_metrics
            ])
        }

        # Log su TensorBoard
        self.writer.add_scalar("TemporalMetrics/id_switch_rate",
                              aggregated['id_switch_rate'], epoch)
        self.writer.add_scalar("TemporalMetrics/temporal_iou",
                              aggregated['temporal_iou_mean'], epoch)
        self.writer.add_scalar("TemporalMetrics/mean_track_length",
                              aggregated['mean_track_length'], epoch)
        self.writer.add_scalar("TemporalMetrics/track_completion_rate",
                              aggregated['track_completion_rate'], epoch)

        print(f"\n📊 METRICHE TEMPORALI:")
        print(f"   ID Switch Rate: {aggregated['id_switch_rate']:.4f}")
        print(f"   Temporal IoU:   {aggregated['temporal_iou_mean']:.4f}")
        print(f"   Mean Track Len: {aggregated['mean_track_length']:.2f}")
        print(f"   Track Complet.: {aggregated['track_completion_rate']:.4f}")

        return aggregated

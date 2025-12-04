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

##############################################
# Pipeline di testing completa con metriche
##############################################

class CompleteTestingPipeline:
    """Pipeline testing con metriche complete - FIXED VERSION"""

    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if model_path is None:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            model_path = config["paths"]["model_save_path"]

        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.model = EnhancedTemporalSSD(
            num_classes=2,
            device=self.device,
            use_adapters=True,
            use_temporal=True,
            use_weather_aug=False
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"✅ Modello caricato: {model_path}")
                else:
                    print("⚠️ Struttura checkpoint non riconosciuta")
            except Exception as e:
                print(f"❌ Errore caricamento modello: {e}")
                print("⚠️ Uso pesi non addestrati")
        else:
            print("⚠️ Modello non trovato, uso pesi non addestrati")

        self.model.eval()
        self.metrics_calc = DetectionMetricsCalculator()

    def test_images(self, test_img_loader, confidence_threshold=0.5):
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
                        for i in range(len(output['boxes'])):
                            if output['scores'][i] >= confidence_threshold:
                                all_detections.append({
                                    'image_path': path,
                                    'bbox': output['boxes'][i].cpu().numpy().tolist(),
                                    'score': float(output['scores'][i].cpu().item()),
                                    'label': int(output['labels'][i].cpu().item())
                                })
                        gt_boxes = target['boxes'].cpu().numpy()
                        gt_labels = target['labels'].cpu().numpy()
                        for i in range(len(gt_boxes)):
                            all_ground_truth.append({
                                'bbox': gt_boxes[i].tolist(),
                                'label': int(gt_labels[i])
                            })
                except Exception as e:
                    print(f"❌ Errore: {e}")
                    continue

        # Calcola metriche
        metrics = self.metrics_calc.compute_metrics(all_detections, all_ground_truth)

        # Print metriche corrette
        print(f"\n📊 METRICHE IMMAGINI:")
        print(f"   mAP medio (tutti IoU): {metrics['mAP_avg']:.4f}")

        # Stampa dettagli per ogni IoU threshold
        for iou_thresh in self.metrics_calc.iou_thresholds:
            iou_key = f'IoU_{iou_thresh}'
            print(f"\n   IoU @ {iou_thresh}:")
            print(f"      mAP: {metrics[iou_key]['mAP']:.4f}")

            for class_id, m in metrics[iou_key]['per_class'].items():
                print(f"      Classe {class_id}: AP={m['AP']:.4f}, Prec={m['Precision']:.4f}, Rec={m['Recall']:.4f}")

        return all_detections, metrics

    def test_videos(self, test_vid_loader, confidence_threshold=0.5):
        # Testing completo su video clips con validazione struttura dati
        print("🎥 TESTING SU VIDEO CON METRICHE TEMPORALI...")
        self.model.eval()
        all_video_detections = {}
        all_temporal_metrics = []

        temporal_calc = TemporalMetrics()

        with torch.no_grad():
            for batch in tqdm(test_vid_loader, desc="Testing video"):
                for sample in batch:
                    clip_rgb = sample["clip_rgb"]  # [T, C, H, W]
                    video_path = sample["path"]

                    try:
                        # Converti tensore in lista di frame
                        images = [frame.to(self.device) for frame in clip_rgb]

                        detections_clip = self.model.forward_video_clip(images)

                        if not isinstance(detections_clip, list):
                            print(f"⚠️ Errore: detections_clip non è una lista. Tipo: {type(detections_clip)}")
                            continue

                        valid_clip = True
                        for frame_idx, frame_dets in enumerate(detections_clip):
                            if not isinstance(frame_dets, list):
                                print(f"⚠️ Frame {frame_idx} non è una lista: {type(frame_dets)}")
                                valid_clip = False
                                break
                            
                            for det_idx, det in enumerate(frame_dets):
                                if not isinstance(det, dict):
                                    print(f"⚠️ Detection {det_idx} nel frame {frame_idx} non è dict: {type(det)}")
                                    valid_clip = False
                                    break
                                
                                required_keys = ['bbox', 'score', 'label']
                                missing_keys = [k for k in required_keys if k not in det]
                                if missing_keys:
                                    print(f"⚠️ Detection manca chiavi {missing_keys}: {det.keys()}")
                                    valid_clip = False
                                    break
                            
                            if not valid_clip:
                                break
                        
                        if not valid_clip:
                            print(f"⚠️ Clip {video_path} ha struttura dati non valida, skip")
                            continue

                        # Calcolo metriche temporali
                        temporal_metrics = temporal_calc.compute_all_metrics(detections_clip)
                        all_temporal_metrics.append(temporal_metrics)

                        video_detections = []
                        for frame_idx, frame_dets in enumerate(detections_clip):
                            for det in frame_dets:
                                score = det.get("score", 0.0)
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
                        import traceback
                        traceback.print_exc()
                        continue

        if all_temporal_metrics:
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

            print(f"\n📊 METRICHE TEMPORALI VIDEO TEST:")
            print(f"   ID Switch Rate: {aggregated['id_switch_rate']:.4f}")
            print(f"   Temporal IoU:   {aggregated['temporal_iou_mean']:.4f}")
            print(f"   Mean Track Len: {aggregated['mean_track_length']:.2f}")
            print(f"   Track Complet.: {aggregated['track_completion_rate']:.4f}")

            return all_video_detections, aggregated
        else:
            print("⚠️ Nessuna metrica temporale calcolata")
            return all_video_detections, {}
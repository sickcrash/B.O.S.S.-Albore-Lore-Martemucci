import os
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import deque
from datetime import datetime
import random
import contextlib
import torch.nn.functional as F
import yaml


##############################################
# 1. MOCK DETECTOR (per testing senza modello)
##############################################

class MockDetector:
    """Generatore di detections fake per testing pipeline"""

    def __init__(self, num_detections_per_frame=5):
        self.num_detections = num_detections_per_frame

    def generate_detections(self, frame_shape=(300, 300), num_frames=1):
        H, W = frame_shape
        all_detections = []
        for frame_idx in range(num_frames):
            frame_dets = []
            num_dets = random.randint(3, self.num_detections)
            for _ in range(num_dets):
                x1 = random.randint(0, W - 50)
                y1 = random.randint(0, H - 50)
                x2 = random.randint(x1 + 30, W)
                y2 = random.randint(y1 + 30, H)
                frame_dets.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': random.uniform(0.5, 0.99),
                    'label': random.randint(0, 1),
                    'track_id': random.randint(0, num_dets)
                })
            all_detections.append(frame_dets)
        return all_detections

    def to_dict(self):
        return {
            'mock': True,
            'num_detections_per_frame': self.num_detections,
            'timestamp': str(datetime.now())
        }


##############################################
# 2. ADVANCED WEATHER AUGMENTATION
##############################################

class AdvancedWeatherAugmentation:
    """Augmentation avanzata per condizioni outdoor"""

    def __init__(self, apply_prob=0.4):
        self.apply_prob = apply_prob
        self.rain_cache = None

    def _generate_rain_texture(self, size):
        h, w = size
        rain = np.zeros((h, w), dtype=np.float32)
        for _ in range(150):
            length = random.randint(20, 60)
            width = random.randint(1, 3)
            x = random.randint(0, w - 1)
            y = random.randint(-length, h - 1)
            intensity = random.uniform(0.2, 0.6)
            for i in range(length):
                y_pos = y + i
                x_pos = x + int(i * 0.3)
                if 0 <= y_pos < h and 0 <= x_pos < w:
                    rain[y_pos, max(0, x_pos - 1):min(w, x_pos + width)] += intensity
        return np.clip(rain, 0, 1)

    def add_realistic_rain(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > self.apply_prob:
            return image
        C, H, W = image.shape
        if self.rain_cache is None:
            self.rain_cache = self._generate_rain_texture((H, W))
        rain_layer = torch.from_numpy(self.rain_cache).to(image.device)
        rain_layer = rain_layer.unsqueeze(0).repeat(3, 1, 1)
        wet_effect = torch.clamp(image * (1 + 0.1 * rain_layer), 0, 1)
        result = wet_effect * (1 - 0.3 * rain_layer) + 0.7 * rain_layer
        return torch.clamp(result, 0, 1)

    def add_dense_fog(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > self.apply_prob:
            return image
        C, H, W = image.shape
        fog_intensity = random.uniform(0.2, 0.6)
        depth_gradient = torch.linspace(0.3, 1.0, H).view(H, 1).repeat(1, W).to(image.device)
        fog_layer = depth_gradient * fog_intensity
        fog_color = random.uniform(0.7, 0.9)
        fog_tensor = torch.ones_like(image) * fog_color
        result = image * (1 - fog_layer) + fog_tensor * fog_layer
        return torch.clamp(result, 0, 1)

    def simulate_backlight(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > self.apply_prob:
            return image
        C, H, W = image.shape
        contrast_reduction = random.uniform(0.3, 0.6)
        image_low_contrast = (image - 0.5) * contrast_reduction + 0.5
        center_y, center_x = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, x = y.float().to(image.device), x.float().to(image.device)
        distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = torch.sqrt(torch.tensor(center_x ** 2 + center_y ** 2, dtype=torch.float32))
        exposure_map = 1 - (distance / max_dist) * 0.8
        overexposed = image_low_contrast + exposure_map.unsqueeze(0) * 0.4
        return torch.clamp(overexposed, 0, 1)

    def day_to_night_transition(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > self.apply_prob:
            return image
        brightness_reduction = random.uniform(0.5, 0.8)
        dark_image = image * (1 - brightness_reduction)
        night_balance = torch.tensor([0.7, 0.9, 1.2], device=image.device).view(3, 1, 1)
        night_image = dark_image * night_balance
        noise_intensity = random.uniform(0.01, 0.03)
        noise = torch.randn_like(night_image) * noise_intensity
        night_image += noise
        return torch.clamp(night_image, 0, 1)

    def apply_weather_effects(self, image: torch.Tensor) -> torch.Tensor:
        weather_types = [
            self.add_realistic_rain,
            self.add_dense_fog,
            self.simulate_backlight,
            self.day_to_night_transition
        ]
        num_effects = random.randint(1, 2)
        selected_effects = random.sample(weather_types, num_effects)
        result = image
        for effect in selected_effects:
            result = effect(result)
        return result


##############################################
# 3. TEMPORAL CONSISTENCY MODULE
##############################################

class AdvancedTemporalConsistency:
    """Modulo per consistenza temporale su clip video"""

    def __init__(self, memory_size=10, iou_threshold=0.4):
        self.memory_size = memory_size
        self.iou_threshold = iou_threshold
        self.detection_memory = deque(maxlen=memory_size)

    def _calculate_iou(self, bbox1, bbox2):
        x11, y11, x21, y21 = bbox1
        x12, y12, x22, y22 = bbox2
        xi1 = max(x11, x12)
        yi1 = max(y11, y12)
        xi2 = min(x21, x22)
        yi2 = min(y21, y22)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = (x21 - x11) * (y21 - y11)
        bbox2_area = (x22 - x12) * (y22 - y12)
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _smooth_bbox(self, current_bbox, previous_bbox, alpha=0.7):
        x1_c, y1_c, x2_c, y2_c = current_bbox
        x1_p, y1_p, x2_p, y2_p = previous_bbox
        x1_s = alpha * x1_p + (1 - alpha) * x1_c
        y1_s = alpha * y1_p + (1 - alpha) * y1_c
        x2_s = alpha * x2_p + (1 - alpha) * x2_c
        y2_s = alpha * y2_p + (1 - alpha) * y2_c
        return [x1_s, y1_s, x2_s, y2_s]

    def _compute_trajectories(self, current_detections, previous_detections):
        if not previous_detections:
            return current_detections
        trajectories = []
        for curr_idx, curr_det in enumerate(current_detections):
            best_match_idx = -1
            best_iou = 0.0
            for prev_idx, prev_det in enumerate(previous_detections):
                iou = self._calculate_iou(curr_det['bbox'], prev_det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = prev_idx
            if best_match_idx != -1:
                smoothed_bbox = self._smooth_bbox(
                    curr_det['bbox'],
                    previous_detections[best_match_idx]['bbox']
                )
                trajectories.append({
                    'bbox': smoothed_bbox,
                    'score': (curr_det['score'] + previous_detections[best_match_idx]['score']) / 2,
                    'label': curr_det['label'],
                    'track_id': previous_detections[best_match_idx].get('track_id', curr_idx)
                })
            else:
                trajectories.append({
                    **curr_det,
                    'track_id': len(trajectories)
                })
        return trajectories

    def process_clip(self, clip_detections):
        consistent_detections = []
        for frame_idx, frame_dets in enumerate(clip_detections):
            if frame_idx == 0:
                consistent_detections.append(frame_dets)
                self.detection_memory.append(frame_dets)
                continue
            previous_dets = self.detection_memory[-1] if self.detection_memory else []
            consistent_frame = self._compute_trajectories(frame_dets, previous_dets)
            consistent_detections.append(consistent_frame)
            self.detection_memory.append(consistent_frame)
        return consistent_detections


##############################################
# Moduli di supporto per il modello Enhanced SSD
##############################################

class DomainAdapter(nn.Module):
    """Lightweight adapter per domain adaptation"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Passaggio in avanti per il DomainAdapter"""
        return x * self.adapter(x)

##############################################
# Custom Loss Function
##############################################

class FocalLoss(nn.Module):
    # Focal Loss per gestione class imbalance

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        p = torch.softmax(predictions, dim=1)
        ce = F.cross_entropy(predictions, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce
        return focal_loss.mean()


##############################################
# Enanced Temporal SSD con supporto ablativo
##############################################

class EnhancedTemporalSSD(nn.Module):
    # SSD con Domain Adapter, Weather Augmentation, Temporal Consistency

    def __init__(self, num_classes: int, device: torch.device,
                 use_adapters: bool = True, use_temporal: bool = True,
                 adapter_reduction: int = 16, use_weather_aug: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.device = device
        self.use_adapters = use_adapters
        self.use_temporal = use_temporal
        self.use_weather_aug = use_weather_aug

        weights = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1
        self.ssd = torchvision.models.detection.ssd300_vgg16(weights=weights)

        num_anchors = self.ssd.anchor_generator.num_anchors_per_location()
        in_channels = [512, 1024, 512, 256, 256, 256]
        self.ssd.head = torchvision.models.detection.ssd.SSDHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )

        if use_adapters:
            self.adapters = nn.ModuleDict({
                'conv7': DomainAdapter(1024, adapter_reduction),
                'conv8_2': DomainAdapter(512, adapter_reduction),
                'conv9_2': DomainAdapter(256, adapter_reduction),
            })

        else:
            self.adapters = None

        if self.adapters is not None:
            self._register_adapter_hooks()

        self._freeze_backbone()

        if use_weather_aug:
            self.weather_aug = AdvancedWeatherAugmentation()
        else:
            self.weather_aug = None

        if use_temporal:
            self.temporal_processor = AdvancedTemporalConsistency()
        else:
            self.temporal_processor = None

        self._analyze_parameters()

    def _register_adapter_hooks(self):
        """
        Ricerca ricorsiva per conv7 e ricerca mirata per extra.
        Registra hook per Domain Adapters su layer specifici del backbone SSD.
        """

        hooks_registered = {
            'conv7': False,
            'conv8_2': False,
            'conv9_2': False
        }

        def make_hook(name):
            def hook(module, input, output):
                return self.adapters[name](output)
            return hook

        print("\n🔧 Registering Domain Adapter Hooks (Deep Search)...")

        candidates = []
        for name, module in self.ssd.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == 1024:
                candidates.append((name, module))

        if len(candidates) > 0:
            layer_name, target_layer = candidates[-1]
            target_layer.register_forward_hook(make_hook('conv7'))
            hooks_registered['conv7'] = True
            print(f"✓ Hook 'conv7' collegato a '{layer_name}' (trovati {len(candidates)} candidati)")
        else:
            print("⚠️ ERRORE: Nessun layer Conv2d con 1024 canali trovato nel backbone!")

        targets_extra = {
            512: 'conv8_2',
            256: 'conv9_2'
        }

        if hasattr(self.ssd.backbone, 'extra'):
            for i, layer in enumerate(self.ssd.backbone.extra):
                target_layer = layer
                if isinstance(layer, nn.Sequential):
                    for sub_layer in layer:
                        if isinstance(sub_layer, nn.Conv2d):
                            target_layer = sub_layer
                            break
                if isinstance(target_layer, nn.Conv2d):
                    ch = target_layer.out_channels
                    if ch in targets_extra:
                        hook_name = targets_extra[ch]
                        target_layer.register_forward_hook(make_hook(hook_name))
                        hooks_registered[hook_name] = True
                        print(f"✓ Hook '{hook_name}' collegato a backbone.extra[{i}] (Canali: {ch})")
                        del targets_extra[ch]
        else:
            print("⚠️ WARNING: backbone.extra non trovato!")

        print("\n🔍 Validazione Hook Registration:")
        all_registered = True
        for hook_name, is_registered in hooks_registered.items():
            status = "✅" if is_registered else "❌"
            print(f"  {status} {hook_name}: {'Registrato' if is_registered else 'MANCANTE'}")
            if not is_registered:
                all_registered = False

        if not all_registered:
            print("⚠️ Alcuni hook Domain Adapter non sono stati registrati correttamente.")
        else:
            print("\n✅ Tutti gli hook Domain Adapter registrati con successo!\n")

        return hooks_registered

    def _freeze_backbone(self):
        for param in self.ssd.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")

    def _analyze_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

        print(f"\n📊 PARAMETER ANALYSIS:")
        print(f"  - Total parameters:     {total_params:>12,}")
        print(f"  - Trainable parameters: {trainable_params:>12,}")
        print(f"  - Trainable ratio:      {trainable_ratio:>12.4f} ({trainable_ratio * 100:.2f}%)")

        if trainable_ratio > 0.05:
            print("⚠️  WARNING: Trainable parameters >5%")
        else:
            print("✓ Trainable parameters <5%")

    def forward(self, images, targets=None):
        if self.training:
            if self.weather_aug is not None and isinstance(images, list):
                images = [self.weather_aug.apply_weather_effects(img) for img in images]
            return self.ssd(images, targets)
        else:
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                return self.forward_video_clip(images)
            else:
                return self.ssd(images)

    def forward_single_frame(self, image: torch.Tensor):
        detection = self.ssd([image])[0]
        formatted_output = {
            'boxes': detection['boxes'].cpu().detach().numpy(),
            'scores': detection['scores'].cpu().detach().numpy(),
            'labels': detection['labels'].cpu().detach().numpy()
        }
        return formatted_output

    def _format_detections(self, detection):
        # Formatta le detections dal modello SSD in formato standard

        boxes = detection['boxes'].cpu().detach().numpy() if isinstance(detection['boxes'], torch.Tensor) else detection['boxes']
        scores = detection['scores'].cpu().detach().numpy() if isinstance(detection['scores'], torch.Tensor) else detection['scores']
        labels = detection['labels'].cpu().detach().numpy() if isinstance(detection['labels'], torch.Tensor) else detection['labels']
        
        # Crea lista di detections nel formato atteso
        formatted = []
        for i in range(len(boxes)):
            formatted.append({
                'bbox': boxes[i].tolist() if isinstance(boxes[i], np.ndarray) else boxes[i],
                'score': float(scores[i]),
                'label': int(labels[i]),
                'track_id': i
            })
        
        return formatted

    def forward_video_clip(self, images):
        """
        Forward pass su un batch di frame video.
        Restituisce lista di frame, ogni frame è lista di detections dict.
        """
        batch_detections = []
        
        for image in images:
            image = image.to(self.device)
            det = self.ssd([image])[0]  # Output del modello SSD
            
            # Validazione output
            if not isinstance(det, dict):
                print(f"⚠️ Errore: Output non è dizionario. Tipo: {type(det)}")
                continue
            
            boxes = det['boxes'].cpu().detach().numpy()
            scores = det['scores'].cpu().detach().numpy()
            labels = det['labels'].cpu().detach().numpy()
            
            frame_detections = []
            for i in range(len(boxes)):
                frame_detections.append({
                    'bbox': boxes[i].tolist(),
                    'score': float(scores[i]),
                    'label': int(labels[i]),
                    'track_id': i  # ID temporaneo
                })
            
            batch_detections.append(frame_detections)
        
        # Applica consistenza temporale se abilitata
        if self.temporal_processor is not None:
            batch_detections = self.temporal_processor.process_clip(batch_detections)
        
        return batch_detections

##############################################
# Calcolo metriche di detection
##############################################

class DetectionMetricsCalculator:
    # Calcola mAP, Precision, Recall a diversi IoU threshold

    def __init__(self, num_classes=2, iou_thresholds=[0.5, 0.75]):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds

    def _calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _compute_metrics_at_iou(self, predictions, ground_truth, iou_threshold, conf_threshold):
        metrics_by_class = {}
        for class_id in range(self.num_classes):
            tp, fp, fn = 0, 0, 0
            pred_class = [p for p in predictions if p['label'] == class_id and p['score'] >= conf_threshold]
            gt_class = [g for g in ground_truth if g['label'] == class_id]
            matched_gt = set()
            for pred in sorted(pred_class, key=lambda x: x['score'], reverse=True):
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gt_class):
                    if gt_idx in matched_gt:
                        continue
                    iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            fn = len(gt_class) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            ap = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics_by_class[class_id] = {
                'AP': ap,
                'Precision': precision,
                'Recall': recall,
                'TP': tp,
                'FP': fp,
                'FN': fn
            }
        mean_ap = np.mean([m['AP'] for m in metrics_by_class.values()]) if metrics_by_class else 0.0
        return {'per_class': metrics_by_class, 'mAP': mean_ap}

    def compute_metrics(self, predictions, ground_truth, conf_threshold=0.5):
        results = {}
        for iou_thresh in self.iou_thresholds:
            metrics = self._compute_metrics_at_iou(predictions, ground_truth, iou_thresh, conf_threshold)
            results[f'IoU_{iou_thresh}'] = metrics
        all_maps = [results[f'IoU_{t}']['mAP'] for t in self.iou_thresholds]
        results['mAP_avg'] = np.mean(all_maps) if all_maps else 0.0
        return results


##############################################
# Calcolo metriche di consistenza temporale
##############################################

class TemporalMetrics:
    # Calcola metriche di consistenza temporale su clip video

    def __init__(self):
        self.id_switches_count = 0
        self.total_tracks = 0

    def compute_id_switches(self, clip_detections):
        """
        Conta quante volte un track_id cambia oggetto tra frame consecutivi.

        Args:
            clip_detections: Lista di frame, ogni frame è lista di detections
                            con 'bbox', 'track_id', 'label'
        Returns:
            dict con id_switches e total_tracks
        """
        id_switches = 0
        track_history = {}  # {track_id: [bboxes]}

        for frame_idx, frame_dets in enumerate(clip_detections):
            current_tracks = {}

            for det in frame_dets:
                track_id = det.get('track_id', -1)
                if track_id == -1:
                    continue

                bbox = det['bbox']

                if track_id in track_history:
                    # Verifica se il bbox è coerente con lo storico
                    prev_bbox = track_history[track_id][-1]
                    iou = self._calculate_iou(bbox, prev_bbox)

                    # Se IoU è troppo basso, potrebbe essere un ID switch
                    if iou < 0.3:
                        id_switches += 1

                    track_history[track_id].append(bbox)
                else:
                    track_history[track_id] = [bbox]

        return {
            'id_switches': id_switches,
            'total_tracks': len(track_history),
            'id_switch_rate': id_switches / max(len(track_history), 1)
        }

    def compute_temporal_iou(self, clip_detections):
        """
        Calcola IoU medio tra frame consecutivi per lo stesso oggetto.
        Misura quanto stabilmente vengono tracciati gli oggetti.

        Args:
            clip_detections: Lista di frame con detections
        Returns:
            dict con temporal_iou medio e per-track
        """
        track_ious = {}  # {track_id: [ious]}

        for frame_idx in range(len(clip_detections) - 1):
            current_frame = clip_detections[frame_idx]
            next_frame = clip_detections[frame_idx + 1]

            # Crea dizionario track_id -> bbox per frame corrente
            current_tracks = {
                det.get('track_id', -1): det['bbox']
                for det in current_frame
                if det.get('track_id', -1) != -1
            }

            # Confronta con frame successivo
            for det in next_frame:
                track_id = det.get('track_id', -1)
                if track_id == -1 or track_id not in current_tracks:
                    continue

                iou = self._calculate_iou(
                    current_tracks[track_id],
                    det['bbox']
                )

                if track_id not in track_ious:
                    track_ious[track_id] = []
                track_ious[track_id].append(iou)

        # Calcola media
        if not track_ious:
            return {'temporal_iou_mean': 0.0, 'per_track_iou': {}}

        all_ious = [iou for ious in track_ious.values() for iou in ious]
        mean_iou = np.mean(all_ious) if all_ious else 0.0

        per_track_mean = {
            tid: np.mean(ious) for tid, ious in track_ious.items()
        }

        return {
            'temporal_iou_mean': mean_iou,
            'per_track_iou': per_track_mean,
            'num_tracked_objects': len(track_ious)
        }

    def compute_track_consistency(self, clip_detections):
        """
        Misura consistenza dei track: per quanto tempo un oggetto
        viene tracciato consecutivamente.

        Args:
            clip_detections: Lista di frame con detections
        Returns:
            dict con lunghezza media track e distribuzione
        """
        track_lengths = {}  # {track_id: num_frames}

        for frame_dets in clip_detections:
            active_tracks = set()
            for det in frame_dets:
                track_id = det.get('track_id', -1)
                if track_id != -1:
                    active_tracks.add(track_id)
                    track_lengths[track_id] = track_lengths.get(track_id, 0) + 1

        if not track_lengths:
            return {
                'mean_track_length': 0.0,
                'max_track_length': 0,
                'num_complete_tracks': 0
            }

        lengths = list(track_lengths.values())
        total_frames = len(clip_detections)

        return {
            'mean_track_length': np.mean(lengths),
            'max_track_length': max(lengths),
            'min_track_length': min(lengths),
            'num_complete_tracks': sum(1 for l in lengths if l == total_frames),
            'track_completion_rate': sum(1 for l in lengths if l == total_frames) / len(lengths)
        }

    def compute_all_metrics(self, clip_detections):
        # Calcola tutte le metriche di consistenza temporale
        return {
            'id_switches': self.compute_id_switches(clip_detections),
            'temporal_iou': self.compute_temporal_iou(clip_detections),
            'track_consistency': self.compute_track_consistency(clip_detections)
        }

    def _calculate_iou(self, bbox1, bbox2):
        # Calcola Intersection over Union (IoU) tra due bounding box
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0


##############################################
# Esportazione modello in ONNX
##############################################

@contextlib.contextmanager
def suppress_stdout_stderr():
    try:
        # Salva i file descriptors originali
        with open(os.devnull, "w") as devnull:
            old_stdout = os.dup(1)
            old_stderr = os.dup(2)
            try:
                # Reindirizza stdout e stderr su /dev/null
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                # Ripristina i file descriptors originali
                os.dup2(old_stdout, 1)
                os.dup2(old_stderr, 2)
    except Exception:
        # Se qualcosa va storto nel setup del silenzio, esegue comunque il codice
        yield


class ModelExporter:
    # Esporta il modello Enhanced SSD in ONNX

    def __init__(self, model, save_dir=None):
        if save_dir is None:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            save_dir = config["paths"]["exports_dir"]

        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def export_models(self, dummy_input_shape=(1, 3, 300, 300)):
        # Export semplificato - SOLO ONNX senza TorchScript
        filepath = os.path.join(self.save_dir, 'ssd_backbone.onnx')
        print(f"⚙️ Export ONNX in corso... ", end="", flush=True)

        try:
            self.model.eval()

            # Disattiva gradienti
            for param in self.model.parameters():
                param.requires_grad = False

            dummy_input = torch.randn(*dummy_input_shape).to(self.model.device)

            # Export ONNX
            with suppress_stdout_stderr():
                with torch.no_grad():
                    torch.onnx.export(
                        self.model.ssd.backbone,  # Solo backbone
                        dummy_input,
                        filepath,
                        input_names=['input'],
                        output_names=['features'],
                        dynamic_axes={'input': {0: 'batch_size'}},
                        opset_version=18,
                        verbose=False
                    )

            if os.path.exists(filepath):
                print("✅ FATTO!")
                return filepath
            else:
                print("❌ Errore: file ONNX non creato.")
                return None

        except Exception as e:
            print(f"\n❌ Export ONNX fallito: {str(e)[:150]}...")
            return None

    def save_metadata(self, metadata_dict, filename='model_metadata.json'):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        print(f"✅ Metadati salvati: {filepath}")

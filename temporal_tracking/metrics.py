import numpy as np
from iou_utils import extract_boxes_from_labels, compute_iou, find_best_matches
from scipy.optimize import linear_sum_assignment

import csv
import os


class TemporalTrack:
    
    #Rappresenta una sequenza temporale di bounding boxes dello stesso oggetto.
    
    def __init__(self, track_id, initial_box, initial_category, frame_idx):
        self.track_id = track_id
        self.boxes = [initial_box]  # Lista di (x1, y1, x2, y2)
        self.category = initial_category
        self.frame_indices = [frame_idx]  
        self.iou_sequence = []  
        self.active = True  
    
    def add_detection(self, box, frame_idx):
        # calcolo IoU con l'ultima box
        if len(self.boxes) > 0:
            iou = compute_iou(self.boxes[-1], box)
            self.iou_sequence.append(iou)
        
        self.boxes.append(box)
        self.frame_indices.append(frame_idx)
        self.active = True
    
    def mark_lost(self):
        #Marca il track come perso (nessuna detection nel frame corrente).
        self.active = False
    
    def get_length(self):
       #lunghezza num di frame
        return len(self.boxes)
    
    def get_iou_stats(self):
        #stast iou
        if len(self.iou_sequence) == 0:
            return None
        
        return {
            'mean': np.mean(self.iou_sequence),
            'variance': np.var(self.iou_sequence),
            'std': np.std(self.iou_sequence),
            'min': np.min(self.iou_sequence),
            'max': np.max(self.iou_sequence),
            'sequence': self.iou_sequence.copy()
        }


class TemporalConsistencyAnalyzer:
    """
    Analizza la consistenza temporale delle detection su brevi clip.
    """
    def __init__(self, iou_threshold=0.3, max_lost_frames=5):
        """
        Args:
            iou_threshold: soglia minima di IoU per associare box tra frame
            max_lost_frames: numero massimo di frame senza detection prima di terminare un track
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 0
        self.active_tracks = {}  # {category: [TemporalTrack, ...]}
        self.completed_tracks = {}  # {category: [TemporalTrack, ...]}
    
    def _get_next_track_id(self):
       #generazione id univoco per ogno track
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
    
    def _initialize_tracks_for_category(self, category):
        if category not in self.active_tracks:
            self.active_tracks[category] = []
        if category not in self.completed_tracks:
            self.completed_tracks[category] = []
    
    def process_frame(self, frame_detections, frame_idx):
        """
        Processa un singolo frame associando le detection ai track esistenti.
        
        Args:
            frame_detections: dict con 'labels' (formato JSON del tuo dataset)
            frame_idx: indice del frame corrente
        """
        labels = frame_detections.get('labels', [])
        
        # Raggruppa detection per categoria
        categories_detections = {}
        for label in labels:
            category = label['category']
            if category not in categories_detections:
                categories_detections[category] = []
            
            box2d = label['box2d']
            box = (box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2'])
            categories_detections[category].append(box)
        
        # Processa ogni categoria separatamente
        for category, current_boxes in categories_detections.items():
            self._initialize_tracks_for_category(category)
            self._associate_detections_to_tracks(category, current_boxes, frame_idx)
            
        # per tutte le categorie che hanno track ma non appaiono nel frame → mark lost
        for category in list(self.active_tracks.keys()):
            if category not in categories_detections:
                self._associate_detections_to_tracks(category, [], frame_idx)
        
        self._handle_lost_tracks(frame_idx)
    
    def _associate_detections_to_tracks(self, category, current_boxes, frame_idx):
        """
        Associa le detection correnti ai track attivi usando IoU (hungarian algorithm).
        """
        active_tracks = [t for t in self.active_tracks[category] if t.active]
        
        if len(active_tracks) == 0:
            # Nessun track attivo: crea nuovi track per tutte le detection
            for box in current_boxes:
                new_track = TemporalTrack(
                    self._get_next_track_id(), 
                    box, 
                    category, 
                    frame_idx
                )
                self.active_tracks[category].append(new_track)
            return
        
        if len(current_boxes) == 0:
            # Nessuna detection: marca tutti i track come persi
            for track in active_tracks:
                track.mark_lost()
            return
        
        # Estrai le ultime box dai track attivi
        track_boxes = [t.boxes[-1] for t in active_tracks]
        
        # Calcola matrice IoU
        from iou_utils import compute_iou_matrix
        iou_matrix = compute_iou_matrix(current_boxes, track_boxes)
        
        # Hungarian Algorithm: bipartite matching ottimale
        det_indices, track_indices = linear_sum_assignment(-iou_matrix)  # negativo per massimizzare
        
        matched_detections = set()
        matched_tracks = set()
        
        # Applica solo match con IoU >= threshold
        for det_idx, track_idx in zip(det_indices, track_indices):
            if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                active_tracks[track_idx].add_detection(current_boxes[det_idx], frame_idx)
                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)
        
        # Marca come persi i track non associati
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.mark_lost()
        
        # Crea nuovi track per detection non associate
        for det_idx, box in enumerate(current_boxes):
            if det_idx not in matched_detections:
                new_track = TemporalTrack(
                    self._get_next_track_id(), 
                    box, 
                    category, 
                    frame_idx
                )
                self.active_tracks[category].append(new_track)
    
    def _handle_lost_tracks(self, frame_idx):
        """
        Gestisce i track che non hanno ricevuto detection.
        Termina i track persi da troppo tempo.
        """
        for category in self.active_tracks:
            tracks_to_keep = []
            
            for track in self.active_tracks[category]:
                if not track.active:
                    frames_lost = frame_idx - track.frame_indices[-1] - 1
                    
                    if frames_lost > self.max_lost_frames:
                        # Track perso da troppo tempo: terminalo
                        self.completed_tracks[category].append(track)
                    else:
                        # Mantienilo ancora attivo
                        tracks_to_keep.append(track)
                else:
                    tracks_to_keep.append(track)
            
            self.active_tracks[category] = tracks_to_keep
    
    def finalize(self):
        for category in self.active_tracks:
            self.completed_tracks[category].extend(self.active_tracks[category])
            self.active_tracks[category] = []
    
    def get_results(self):
        """
       Ritorna i risultati dell'analisi.
        
        Returns:
            dict: {
                'category_name': {
                    'tracks': [TemporalTrack, ...],
                    'total_tracks': int,
                    'avg_track_length': float,
                    'avg_iou': float,  # IoU media tra tutti i track
                    'std_iou': float   # Deviazione standard IoU
                },
                ...
            }
        """
        results = {}
        
        for category in self.completed_tracks:
            tracks = self.completed_tracks[category]
            
            if len(tracks) == 0:
                continue
            
            track_lengths = [t.get_length() for t in tracks]
            
            # Raccogli tutte le IoU da tutti i track
            all_ious = []
            for track in tracks:
                iou_stats = track.get_iou_stats()
                if iou_stats:
                    all_ious.extend(iou_stats['sequence'])
            
            results[category] = {
                'tracks': tracks,
                'total_tracks': len(tracks),
                'avg_track_length': np.mean(track_lengths),
                'min_track_length': np.min(track_lengths),
                'max_track_length': np.max(track_lengths),
                'avg_iou': np.mean(all_ious) if all_ious else None,
                'std_iou': np.std(all_ious) if all_ious else None
            }
        
        return results
    
    def _reactivate_completed_track(self, category, new_box, frame_idx):
        """Cerca di reattivare un track completed con IoU alto."""
        if category not in self.completed_tracks:
            return None
        
        best_iou = 0
        best_track = None
        
        for track in self.completed_tracks[category]:
            iou = compute_iou(track.boxes[-1], new_box)
            if iou > 0.6 and iou > best_iou:  # Soglia alta per reattivazione
                best_iou = iou
                best_track = track
        
        if best_track:
            best_track.add_detection(new_box, frame_idx)
            best_track.active = True
            return best_track
        
        return None


def analyze_clip(frames_detections, iou_threshold=0.3, max_lost_frames=5, category_filter=None):
    """
    Funzione helper per analizzare una clip completa.
    """
    analyzer = TemporalConsistencyAnalyzer(iou_threshold, max_lost_frames)
    
    for frame_idx, frame_data in enumerate(frames_detections):
        analyzer.process_frame(frame_data, frame_idx)
    
    analyzer.finalize()
    
    results = analyzer.get_results()
    
    if category_filter:
        results = {k: v for k, v in results.items() if k in category_filter}
    
    return results

def compute_all_metrics(results, total_frames=None):
    """
    Calcola metriche aggregate per ogni categoria a partire dai risultati di analyze_clip().
    Restituisce dict {category: {metrics...}}.
    """
    out = {}
    for cat, info in results.items():
        tracks = info.get('tracks', [])
        total_tracks = info.get('total_tracks', 0)
        avg_len = info.get('avg_track_length', None)

        # completeness: percentuale di frame coperti da almeno una detection 
        frames_set = set()
        for t in tracks:
            frames_set.update(t.frame_indices)
        if total_frames:
            completeness = len(frames_set) / float(total_frames)
        else:
            completeness = None

        # iou raccolte da tutti i track
        all_ious = []
        for t in tracks:
            iou_stats = t.get_iou_stats()
            if iou_stats:
                all_ious.extend(iou_stats['sequence'])

        # media e deviazione standard delle iou
        avg_iou = float(np.mean(all_ious)) if all_ious else None
        std_iou = float(np.std(all_ious)) if all_ious else None

        out[cat] = {
            'total_tracks': total_tracks,
            'avg_track_length': float(avg_len) if avg_len is not None else None,
            'avg_iou': avg_iou,
            'std_iou': std_iou,
            'completeness': completeness,
            'tracks': tracks
        }
    return out

def generate_report(metrics_dict, total_frames, clip_name="clip", show_track_details=False):

    lines = []
    lines.append("=" * 80)
    lines.append(f"TEMPORAL CONSISTENCY REPORT - {clip_name}")
    lines.append("=" * 80)
    lines.append(f"Total frames: {total_frames}")
    lines.append("")

    for cat, m in metrics_dict.items():
        lines.append(f"Category: {cat}")
        lines.append(f"  Total tracks: {m['total_tracks']}")
        lines.append(f"  Avg track length: {m['avg_track_length']}")
        lines.append(f"  Avg IoU: {m['avg_iou']}")
        lines.append(f"  IoU std: {m['std_iou']}")
        lines.append(f"  Completeness (frames covered / total): {m['completeness']}")
        if show_track_details and m.get('tracks'):
            for t in m['tracks']:
                iou_stats = t.get_iou_stats()
                lines.append(f"    Track {t.track_id} | len={t.get_length()} | frames={t.frame_indices} | iou_stats={iou_stats}")
        lines.append("-" * 80)

    return "\n".join(lines)


def save_report(report_str, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    return output_path

def save_metrics_to_csv(metrics_dict, filename="complete_output.csv"):
 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, filename)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        writer.writerow([
            "category",
            "total_tracks",
            "avg_track_length",
            "avg_iou",
            "std_iou",
            "completeness",
        ])
        
        for category, m in metrics_dict.items():
            writer.writerow([
                category,
                m.get("total_tracks"),
                m.get("avg_track_length"),
                m.get("avg_iou"),
                m.get("std_iou"),
                m.get("completeness"),
            ])

    print(f"CSV salvato in: {output_path}")
    return output_path

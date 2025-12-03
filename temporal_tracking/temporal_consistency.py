import numpy as np
import csv
import os
from iou_utils import extract_boxes_from_labels, compute_iou, find_best_matches

__all__ = ["analyze_clip", "compute_all_metrics", "generate_report", "save_report"]


class TemporalTrack:

    def __init__(self, track_id, initial_box, initial_category, frame_idx):
        self.track_id = track_id
        self.boxes = [initial_box]  # Lista di (x1, y1, x2, y2)
        self.category = initial_category
        self.frame_indices = [frame_idx]  # Indici dei frame
        self.iou_sequence = [] 
        self.active = True  
        
    def add_detection(self, box, frame_idx):
        """Aggiunge una nuova detection al track."""
        if len(self.boxes) > 0:
            iou = compute_iou(self.boxes[-1], box)
            self.iou_sequence.append(iou)
        
        self.boxes.append(box)
        self.frame_indices.append(frame_idx)
        self.active = True
    
    def mark_lost(self):
        self.active = False
    
    def get_length(self):
        return len(self.boxes)
    
    def get_iou_stats(self):
        """Ritorna statistiche sulle IoU."""
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

    def __init__(self, iou_threshold=0.3, max_lost_frames=5):

        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 0
        self.active_tracks = {}  
        self.completed_tracks = {} 
    
    def _get_next_track_id(self):
        #Genera un nuovo ID univoco per un track.
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
    
    def _initialize_tracks_for_category(self, category):
        if category not in self.active_tracks:
            self.active_tracks[category] = []
        if category not in self.completed_tracks:
            self.completed_tracks[category] = []
    
    def process_frame(self, frame_detections, frame_idx):
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
        
        # Gestisci track che non hanno ricevuto detection in questo frame
        self._handle_lost_tracks(frame_idx)
    
    def _associate_detections_to_tracks(self, category, current_boxes, frame_idx):

        active_tracks = [t for t in self.active_tracks[category] if t.active]
        
        if len(active_tracks) == 0:
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
            for track in active_tracks:
                track.mark_lost()
            return
        
        # Estrai le ultime box dai track attivi
        track_boxes = [t.boxes[-1] for t in active_tracks]
        
        # Trova le migliori associazioni usando IoU
        matches = find_best_matches(current_boxes, track_boxes, self.iou_threshold)
        
        matched_detections = set()
        matched_tracks = set()
        
        # Aggiornamento track
        for det_idx, track_idx in matches.items():
            active_tracks[track_idx].add_detection(current_boxes[det_idx], frame_idx)
            matched_detections.add(det_idx)
            matched_tracks.add(track_idx)
        
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.mark_lost()
        
        # Creazione  nuovi track per detection non associate
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
                    frames_lost = frame_idx - track.frame_indices[-1]
                    
                    if frames_lost > self.max_lost_frames:
                        # Track perso da troppo tempo
                        self.completed_tracks[category].append(track)
                    else:
                        tracks_to_keep.append(track)
                else:
                    tracks_to_keep.append(track)
            
            self.active_tracks[category] = tracks_to_keep
    
    def finalize(self):
        for category in self.active_tracks:
            self.completed_tracks[category].extend(self.active_tracks[category])
            self.active_tracks[category] = []
    
    def get_results(self):
        
        #Ritorna i risultati dell'analisi.
      
        results = {}
        
        for category in self.completed_tracks:
            tracks = self.completed_tracks[category]
            
            if len(tracks) == 0:
                continue
            
            track_lengths = [t.get_length() for t in tracks]
            
            results[category] = {
                'tracks': tracks,
                'total_tracks': len(tracks),
                'avg_track_length': np.mean(track_lengths),
                'min_track_length': np.min(track_lengths),
                'max_track_length': np.max(track_lengths)
            }
        
        return results


def analyze_clip(frames_detections, iou_threshold=0.3, max_lost_frames=5, category_filter=None, save_csv=False):

    analyzer = TemporalConsistencyAnalyzer(iou_threshold, max_lost_frames)
    
    # Processa ogni frame
    for frame_idx, frame_data in enumerate(frames_detections):
        analyzer.process_frame(frame_data, frame_idx)
    
    analyzer.finalize()
    
    results = analyzer.get_results()
    
    if category_filter:
        results = {k: v for k, v in results.items() if k in category_filter}
    
    if save_csv:
        try:
            total_frames = len(frames_detections)
            metrics = compute_all_metrics(results, total_frames=total_frames)
            csv_path = os.path.join(os.getcwd(), "complete_output.csv")
            with open(csv_path, mode="w", newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "category", "total_tracks", "avg_track_length",
                    "min_track_length", "max_track_length",
                    "avg_iou", "std_iou", "completeness"
                ])
                for category, m in metrics.items():
                    writer.writerow([
                        category,
                        m.get('total_tracks'),
                        m.get('avg_track_length'),
                        m.get('min_track_length'),
                        m.get('max_track_length'),
                        m.get('avg_iou'),
                        m.get('std_iou'),
                        m.get('completeness')
                    ])
        except Exception as e:
            print(f"Warning: unable to save CSV: {e}")
    
    return results
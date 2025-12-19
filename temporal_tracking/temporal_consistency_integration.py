from temporal_tracking.temporal_consistency import analyze_clip
from temporal_tracking.metrics import compute_all_metrics


class TemporalConsistencyIntegration:
    """
    Integra il temporal_tracking con i risultati di detection dal modello.
    """
    
    def __init__(self, iou_threshold=0.3, max_lost_frames=5):
        """
        Args:
            iou_threshold: soglia minima IoU per associare detection tra frame
            max_lost_frames: max frame senza detection prima di terminare un track
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
    
    def convert_model_output_to_json(self, video_results_dict, confidence_threshold=0.5):
        """
        Converte l'output del modello SSD300 nel formato JSON atteso dal temporal_tracking.
        """
        all_videos_json = {}
        
        for video_name, detections_list in video_results_dict.items():
            frames_json = []
            frames_by_id = {}
            
            for detection in detections_list:
                frame_id = detection.get('frame_id', 0)
                
                if frame_id not in frames_by_id:
                    frames_by_id[frame_id] = []
                
                frames_by_id[frame_id].append(detection)
            
            for frame_id in sorted(frames_by_id.keys()):
                labels = []
                
                for detection in frames_by_id[frame_id]:
                    score = detection.get('score', 0)
                    
                    if score < confidence_threshold:
                        continue
                    
                    bbox = detection.get('bbox', None)
                    if bbox is None or len(bbox) < 4:
                        continue
                    
                    label = detection.get('label', 'unknown')
                    
                    label_dict = {
                        'category': str(label),
                        'box2d': {
                            'x1': float(bbox[0]),
                            'y1': float(bbox[1]),
                            'x2': float(bbox[2]),
                            'y2': float(bbox[3])
                        },
                        'confidence': float(score)
                    }
                    
                    labels.append(label_dict)
                
                frames_json.append({'labels': labels})
            
            all_videos_json[video_name] = frames_json
        
        return all_videos_json
    
    def analyze_temporal_consistency(self, video_results_dict, confidence_threshold=0.5, 
                                    category_filter=None, save_csv=False):
        """
        Esegue l'analisi completa della consistenza temporale per tutti i video.
        """
        all_videos_json = self.convert_model_output_to_json(video_results_dict, confidence_threshold)
        
        all_results = {}
        
        for video_name, frames_json in all_videos_json.items():
            results = analyze_clip(
                frames_json,
                iou_threshold=self.iou_threshold,
                max_lost_frames=self.max_lost_frames,
                category_filter=category_filter
            )
            
            metrics = compute_all_metrics(results, total_frames=len(frames_json))
            
            all_results[video_name] = {
                'raw_results': results,
                'metrics': metrics,
                'total_frames': len(frames_json)
            }
        
        return all_results
    
    def print_temporal_report(self, all_analysis_results):
        """Stampa un report dettagliato dell'analisi per tutti i video."""
        
        print("\n" + "="*80)
        print("TEMPORAL CONSISTENCY ANALYSIS - TUTTI I VIDEO")
        print("="*80)
        
        for video_name, analysis_result in all_analysis_results.items():
            metrics = analysis_result['metrics']
            total_frames = analysis_result['total_frames']
            
            print(f"\nVideo: {video_name}")
            print(f"   Total frames: {total_frames}")
            print("-"*80)
            """ Metriche (stampate nel json e csv, usate nella shell per fare il debug)
            
            for category, m in metrics.items():
                print(f"   Category: {category}")
                print(f"      Total tracks: {m.get('total_tracks', 'N/A')}")
                print("-"*80)
                
                avg_track_length = m.get('avg_track_length', 'N/A')
                avg_track_length = round(avg_track_length, 2) if isinstance(avg_track_length, (int, float)) else avg_track_length
                print(f"      Avg track length: {avg_track_length}")
                print("-"*80)
                
                avg_iou = m.get('avg_iou', 'N/A')
                avg_iou = round(avg_iou, 4) if isinstance(avg_iou, (int, float)) else avg_iou
                print(f"      Avg IoU: {avg_iou}")
                print("-"*80)
                
                std_iou = m.get('std_iou', 'N/A')
                std_iou = round(std_iou, 4) if isinstance(std_iou, (int, float)) else std_iou
                print(f"      IoU std: {std_iou}")
                print("-"*80)
                
                completeness = m.get('completeness', 'N/A')
                completeness = round(completeness, 4) if isinstance(completeness, (int, float)) else completeness
                print(f"      Completeness: {completeness}")
                print("-"*80)
        """
        print("\n" + "="*80)
        print("Analysis completed!")
        print("="*80 + "\n")
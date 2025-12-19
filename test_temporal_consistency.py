"""
Script per testare il modello con l'analisi della consistenza temporale.
"""

import torch
import csv
import json
import os
from datetime import datetime
from tester_net import CompleteTestingPipeline
from dataset_builder import RealBDDAVideoDataset, collate_fn_video
from torch.utils.data import DataLoader
from temporal_tracking.temporal_consistency_integration import TemporalConsistencyIntegration

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Carica modello
print("Caricamento modello...")
tester = CompleteTestingPipeline(
    model_path="./complete_full_model.pth",
    device=device
)
print("Modello caricato!\n")

# Carica dataset
print("Caricamento dataset video...")
test_dataset = RealBDDAVideoDataset(
    root_dir="./dataset/BDDA/test/",
    clip_length=8,
    max_samples=200
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=collate_fn_video
)
print(f"Dataset caricato ({len(test_dataset)} video)\n")

# Se non carica video, prova l'altra cartella
if len(test_dataset) == 0:
    print("Nessun video in camera_videos, provo gazemap_videos...")
    test_dataset = RealBDDAVideoDataset(
        root_dir="./dataset/BDDA/test/gazemap_videos",
        clip_length=8,
        max_samples=200
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn_video
    )
    print(f"Dataset caricato ({len(test_dataset)} video)\n")

# Esegui test
print("Testing con detection...\n")
video_results, _ = tester.test_videos(test_loader, confidence_threshold=0.5)

# Filtra video problematici
print("Filtraggio video problematici...")
if "1011.mp4" in video_results:
    del video_results["1011.mp4"]
    print("Video 1011.mp4 escluso (video nero/corrotto)\n")

# Inizializza integration wrapper
print("Inizializzazione temporal consistency integration...\n")
temporal_integration = TemporalConsistencyIntegration(
    iou_threshold=0.5,
    max_lost_frames=5
)

# Analizza consistenza temporale
print("Esecuzione analisi temporale...\n")
analysis_results = temporal_integration.analyze_temporal_consistency(
    video_results_dict=video_results,
    confidence_threshold=0.5,
    category_filter=None,
    save_csv=False
)

# Stampa del report
temporal_integration.print_temporal_report(analysis_results)

# Salva risultati in CSV e JSON
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Crea cartella output se non esiste
output_dir = "./temporal_tracking/metrics_output"
os.makedirs(output_dir, exist_ok=True)

csv_filename = os.path.join(output_dir, f"temporal_metrics_{timestamp}.csv")
json_filename = os.path.join(output_dir, f"temporal_metrics_{timestamp}.json")

print(f"Salvataggio metriche in temporal_tracking/...")

# SALVA CSV
csv_data = []
for video_name, analysis_result in analysis_results.items():
    metrics = analysis_result['metrics']
    total_frames = analysis_result['total_frames']
    
    for category, m in metrics.items():
        csv_data.append({
            'Video': video_name,
            'Total_Frames': total_frames,
            'Category': category,
            'Total_Tracks': m.get('total_tracks', 'N/A'),
            'Avg_Track_Length': round(m.get('avg_track_length', 0), 2) if m.get('avg_track_length') is not None else 'N/A',
            'Avg_IoU': round(m.get('avg_iou', 0), 4) if m.get('avg_iou') is not None else 'N/A',
            'IoU_Std': round(m.get('std_iou', 0), 4) if m.get('std_iou') is not None else 'N/A',
            'Completeness': round(m.get('completeness', 0), 4) if m.get('completeness') is not None else 'N/A'
        })

if csv_data:
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"CSV salvato: {csv_filename}")
else:
    print("Nessun dato da salvare in CSV")

# SALVA JSON
json_data = {
    'timestamp': timestamp,
    'total_videos': len(analysis_results),
    'results': {}
}

for video_name, analysis_result in analysis_results.items():
    metrics = analysis_result['metrics']
    
    serializable_metrics = {}
    for category, m in metrics.items():
        serializable_metrics[category] = {
            'total_tracks': m.get('total_tracks', 'N/A'),
            'avg_track_length': round(m.get('avg_track_length', 0), 2) if m.get('avg_track_length') is not None else 'N/A',
            'avg_iou': round(m.get('avg_iou', 0), 4) if m.get('avg_iou') is not None else 'N/A',
            'std_iou': round(m.get('std_iou', 0), 4) if m.get('std_iou') is not None else 'N/A',
            'completeness': round(m.get('completeness', 0), 4) if m.get('completeness') is not None else 'N/A'
        }
    
    json_data['results'][video_name] = {
        'total_frames': analysis_result['total_frames'],
        'metrics': serializable_metrics
    }

with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"JSON salvato: {json_filename}")

print("\n" + "="*80)
print("TEST COMPLETATO!")
print("="*80)
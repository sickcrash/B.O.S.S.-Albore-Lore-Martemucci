import torch
import logging
from torchmetrics.detection import MeanAveragePrecision

def evaluate_model(model, dataloader, device):
    print("\nInizio fase di valutazione...")
    logging.info("--- INIZIO VALUTAZIONE ---")
    model.eval()
    
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device)

    with torch.no_grad():
        for images, targets in dataloader:
            if images is None or not images:
                continue

            images = list(img.to(device) for img in images)
            predictions = model(images)
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
            metric.update(predictions, targets_on_device)

    try:
        results = metric.compute()
        
        full_summary_message = f"Valutazione completata. Risultati mAP Dettagliati: {results}"
        print(full_summary_message)
        logging.info(full_summary_message)
        
        map_val = results['map'].item()
        mar_100_val = results['mar_100'].item()
        compact_summary = f"RIEPILOGO VALUTAZIONE -> mAP: {map_val:.4f}, mAR@100: {mar_100_val:.4f}"
        
        print(compact_summary)
        logging.info(compact_summary)
        
    except Exception as e:
        print(f"Errore durante il calcolo delle metriche: {e}")
        logging.error(f"Errore durante il calcolo delle metriche: {e}")
        results = None
        
    logging.info("--- VALUTAZIONE TERMINATA ---\n")
    return results
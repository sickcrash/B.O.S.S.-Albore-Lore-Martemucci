#!/usr/bin/env python3
"""
YOLOv8 Live Demo - Edge Detection in tempo reale
Implementazione con Pattern Corretti: Facade, Strategy, Dependency Injection
"""
import argparse
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

from src.platform import EdgePlatform
from src.patterns import FastPreprocessor, FastNMS, ONNXBackend, TorchBackend

# Nomi classi COCO standard
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Colori per le classi
COLORS = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3))


def draw_detections(frame: np.ndarray, detections: List[Dict],
                    scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """
    Disegna le detection sul frame

    Args:
        frame: Frame su cui disegnare
        detections: Lista di detection
        scale_x: Fattore di scala X (per adattare bbox al frame originale)
        scale_y: Fattore di scala Y

    Returns:
        Frame con detection disegnate
    """
    for det in detections:
        bbox = det['bbox']  # [x1, y1, x2, y2]
        conf = det['confidence']
        cls_id = det['class_id']

        # Scala coordinate al frame originale
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)

        # Label e colore
        label = f"{COCO_CLASSES.get(cls_id, 'Unk')} {conf:.2f}"
        color = tuple(map(int, COLORS[cls_id % len(COLORS)]))

        # Rettangolo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Testo con sfondo
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5,
                    [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return frame


def draw_info_overlay(frame: np.ndarray, fps: float, inf_time: float,
                      backend_name: str, num_detections: int) -> np.ndarray:
    """Disegna overlay informativo"""
    h, w = frame.shape[:2]

    # FPS e timing
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {inf_time * 1000:.1f}ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {num_detections}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Info backend
    cv2.putText(frame, f"Backend: {backend_name}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Live Demo')

    parser.add_argument('--model', default='./model/yolov8nbase_int8_static.onnx',
                        help='Path al modello')
    parser.add_argument('--backend', choices=['onnx', 'torch'], default='onnx',
                        help='Backend di inferenza')
    parser.add_argument('--imgsz', type=int, default=480,
                        help='Dimensione input modello')
    parser.add_argument('--cam', type=int, default=0,
                        help='Indice camera')
    parser.add_argument('--conf', type=float, default=0.40,
                        help='Soglia confidenza')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='Soglia IOU per NMS')
    parser.add_argument('--max-det', type=int, default=50,
                        help='Numero massimo detection')
    parser.add_argument('--display-size', type=int, default=640,
                        help='Dimensione finestra display')

    args = parser.parse_args()

    # Validazione modello
    model_path = Path(args.model)
    print(model_path)
    if not model_path.exists():
        print(f"Errore: Modello non trovato in {model_path}")
        print("   Esegui optimize.py prima di avviare il demo!")
        return

    print(f"YOLOv8 Live Demo")
    print(f"   Modello: {model_path.name}")
    print(f"   Backend: {args.backend.upper()}")
    print(f"   Input: {args.imgsz}x{args.imgsz}")
    print(f"   Display: {args.display_size}x{args.display_size}")
    print("   Premi 'q' per uscire, 's' per salvare frame\n")

    # Creazione backend con Strategy Pattern
    if args.backend == 'onnx':
        backend = ONNXBackend(str(model_path))
    else:
        backend = TorchBackend(str(model_path))

    # Inizializzazione piattaforma Edge (Dependency Injection)
    platform = EdgePlatform(
        preprocessor=FastPreprocessor(),
        postprocessor=FastNMS(max_det=args.max_det),
        backend=backend,
        input_size=(args.imgsz, args.imgsz)
    )

    # Usa HardwareIO (Facade Pattern) attraverso EdgePlatform
    # Inizializza camera
    test_frame = platform.hardware_io.read_from_camera(args.cam)
    if test_frame is None:
        print(" Errore: Impossibile aprire la webcam.")
        platform.cleanup()
        return

    # Ottieni proprietà camera
    cam_props = platform.hardware_io.get_camera_properties()
    if cam_props:
        print(f" Camera: {cam_props['width']}x{cam_props['height']} @ {cam_props['fps']:.1f}fps")
        print(f"   Backend: {cam_props['backend']}\n")

    # Variabili per FPS
    prev_time = time.time()
    fps_avg = 0
    frame_count = 0

    try:
        while True:
            #  Lettura frame tramite Facade
            frame = platform.hardware_io.read_from_camera(args.cam)

            if frame is None:
                print(" Frame perso, continuo...")
                continue

            # Dimensioni originali per scaling delle bbox
            orig_h, orig_w = frame.shape[:2]

            #  Detection (il resize è fatto internamente dal preprocessor)
            detections, inf_time = platform.detect(frame, conf=args.conf, iou=args.iou)

            # Resize per display (opzionale, solo estetico)
            if args.display_size != orig_w:
                frame_display = cv2.resize(frame, (args.display_size, args.display_size))
                scale_x = args.display_size / args.imgsz
                scale_y = args.display_size / args.imgsz
            else:
                frame_display = frame.copy()
                scale_x = orig_w / args.imgsz
                scale_y = orig_h / args.imgsz

            # Disegna detection (con scaling corretto)
            frame_display = draw_detections(frame_display, detections, scale_x, scale_y)

            # Calcolo FPS
            curr_time = time.time()
            loop_time = curr_time - prev_time
            prev_time = curr_time
            fps = 1 / loop_time if loop_time > 0 else 0
            fps_avg = 0.9 * fps_avg + 0.1 * fps  # Media mobile

            # Overlay informazioni
            frame_display = draw_info_overlay(
                frame_display, fps_avg, inf_time,
                backend.get_name(), len(detections)
            )

            # Mostra frame
            cv2.imshow('YOLOv8 Live Demo - Edge Detection', frame_display)
            frame_count += 1

            # Gestione tasti
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salva frame
                filename = f"capture_{int(time.time())}.jpg"
                platform.hardware_io.write_to_file(frame_display, filename, quality=95)
                print(f"📸 Frame salvato: {filename}")

    except KeyboardInterrupt:
        print("\n Interruzione utente")

    finally:
        #  Cleanup con pattern Facade
        print(f"\n Statistiche:")
        print(f"   Frame processati: {frame_count}")
        print(f"   FPS medio: {fps_avg:.2f}")
        platform.cleanup()
        print("✓ Risorse rilasciate")


if __name__ == "__main__":
    main()
"""Benchmark modelli ottimizzati"""

import argparse
import json
import numpy as np
from pathlib import Path

from src.platform import EdgePlatform
from src.patterns import FastPreprocessor, FastNMS, TorchBackend, ONNXBackend


def load_test_images(image_dir: str, num: int = 10):
    """Carica immagini di test"""
    import cv2
    images = []
    image_paths = list(Path(image_dir).glob("*.jpg"))[:num]

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)

    if not images:
        print("⚠️ Nessuna immagine trovata, uso immagini random")
        images = [np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8) for _ in range(10)]

    return images


def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLO')
    parser.add_argument('--test-images', default="./datasets/coco/images/val2017")
    parser.add_argument('--baseline', default='./model/yolov8n.pt')
    parser.add_argument('--optimized', default='./model/yolov8nbase_int8_dynamic.onnx')
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--output', default='results/benchmark.json')

    args = parser.parse_args()

    print("=" * 70)
    print("BENCHMARK YOLO EDGE")
    print("=" * 70)

    # Carica test images
    print("\nCaricamento immagini test...")
    test_images = load_test_images(args.test_images, 10)
    print(f"✓ Caricate {len(test_images)} immagini")

    results = {}

    # Benchmark modello baseline
    if Path(args.baseline).exists():
        print(f"\n{'=' * 70}")
        print("Benchmark PyTorch Baseline")
        print(f"{'=' * 70}")

        platform_baseline = EdgePlatform(
            preprocessor=FastPreprocessor(),
            postprocessor=FastNMS(),
            backend=TorchBackend(args.baseline)
        )

        res = platform_baseline.benchmark(test_images, args.num_runs)
        results['baseline'] = res

        print(f"FPS: {res['fps']:.2f}")
        print(f"Latenza: {res['mean_time'] * 1000:.2f} ms")
        platform_baseline.cleanup()

    # Benchmark modello ottimizzato
    if Path(args.optimized).exists():
        print(f"\n{'=' * 70}")
        print("Benchmark ONNX Ottimizzato")
        print(f"{'=' * 70}")

        platform_opt = EdgePlatform(
            preprocessor=FastPreprocessor(),
            postprocessor=FastNMS(),
            backend=ONNXBackend(args.optimized),
            input_size=(480, 480)
        )

        res = platform_opt.benchmark(test_images, args.num_runs)
        results['optimized'] = res

        print(f"FPS: {res['fps']:.2f}")
        print(f"Latenza: {res['mean_time'] * 1000:.2f} ms")

        if 'baseline' in results:
            speedup = res['fps'] / results['baseline']['fps']
            results['speedup'] = float(speedup)
            print(f"Speed-up: {speedup:.2f}x")

        platform_opt.cleanup()

    # Salva risultati
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Risultati salvati: {output_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    if 'speedup' in results:
        status = "✅" if results['speedup'] >= 1.5 else "❌"
        print(f"{status} Speed-up: {results['speedup']:.2f}x (obiettivo ≥1.5x)")

    from ultralytics import YOLO

    def eval_map(model_path, data_yaml):
        m = YOLO(model_path)
        metrics = m.val(data=data_yaml, imgsz = 480, save=False, save_json=False, plots=False, project="", name="", exist_ok=True,
                        verbose=False, device='cpu')
        return float(metrics.box.map50), float(metrics.box.map)

    # dopo aver creato le piattaforme e misurato FPS:
    data_yaml = './datasets/coco.yaml'
    b_map50, b_map = eval_map(args.baseline, data_yaml)
    q_map50, q_map = eval_map(args.optimized, data_yaml)

    drop50 = b_map50 - q_map50
    drop = b_map - q_map
    print(f"mAP50 baseline={b_map50:.4f} quant={q_map50:.4f} drop={drop50 * 100:.2f} pp")
    print(f"mAP50-95 baseline={b_map:.4f} quant={q_map:.4f} drop={drop * 100:.2f} pp")

    # soglia 3 punti percentuali
    ok = (drop50 <= 0.03) and (drop <= 0.03)
    status = 'OK' if ok else 'FAIL'
    print(f"[{status}] Degrado mAP ≤ 3 pp su mAP50 e mAP50-95")


if __name__ == "__main__":
    main()
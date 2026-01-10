
import argparse
import yaml
from src.optimizer import ModelOptimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./model/yolov8n.pt')
    parser.add_argument('--config', default='./config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Percorso dataset dal config
    dataset_yaml = './' + config['dataset']['yaml_path']

    print("=" * 60)
    print("YOLO OPTIMIZER (MANUAL FIX)")
    print("=" * 60)

    optimizer = ModelOptimizer(args.model, dataset_yaml)


    optimizer.calib_img_dir = "./datasets/coco/images/val2017"

    # Export FP32
    onnx_base = optimizer.export_onnx_base()

    # Quantize INT8
    onnx_int8_dynamic = optimizer.quantize_int8_dynamic(onnx_base)
    onnx_int8_static = optimizer.quantize_int8_static(onnx_base)

    # Stampa i risultati
    print("\nDONE.")
    print(f"Base FP32 ONNX: {onnx_base}")
    print(f"INT8 Dynamic ONNX: {onnx_int8_dynamic}")
    print(f"INT8 Static ONNX: {onnx_int8_static}")


if __name__ == "__main__":
    main()
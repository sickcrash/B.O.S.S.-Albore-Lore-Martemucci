
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
import torch
#import torch_pruning as tp
from copy import deepcopy
import yaml
from src.pruning_utils import replace_c2f_with_c2f_v2
from ultralytics.nn.modules import Detect


from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)
from onnxruntime.quantization.shape_inference import quant_pre_process

class YoloDataReader(CalibrationDataReader):
    """Legge le immagini per la calibrazione"""
    def __init__(self, image_dir: str, input_name: str = 'images', num_imgs: int = 50):
        self.image_paths = list(Path(image_dir).glob('*.jpg'))[:num_imgs]
        self.input_name = input_name
        self.enum_data = iter(self.image_paths)

    def get_next(self):
        try:
            img_path = next(self.enum_data)
            # Preprocessing standard YOLO
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (480, 480))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            return {self.input_name: img}
        except StopIteration:
            return None

class ModelOptimizer:
    def __init__(self, model_path: str, dataset_yaml: str):
        self.model_path = model_path
        self.dataset_yaml = dataset_yaml
        self.model = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')

        self.calib_img_dir = "datasets/coco/images/val2017"

    def export_onnx_base(self) -> str:
        """Esporta il modello base FP32"""
        print(f"1. Exporting Standard ONNX (Float32)...")
        path = self.model.export(
            format='onnx',
            simplify=True,
            opset=13,
            dynamic=False,
            imgsz = 480
        )

        out_path = str(Path(path).with_stem(Path(path).stem + 'base'))
        Path(path).rename(out_path)
        return str(out_path)


    def quantize_int8_dynamic(self, onnx_path: str) -> str:
        """Quantizzazione dinamica INT8"""
        print(f"2. Starting Dynamic INT8 Quantization on {onnx_path}...")

        out_path = str(Path(onnx_path).with_stem(Path(onnx_path).stem + '_int8_dynamic'))

        try:
            quantize_dynamic(
                model_input=onnx_path,
                model_output=out_path,
                weight_type=QuantType.QUInt8)
            print(f"Saved Dynamic INT8 model: {out_path}")
            return out_path

        except Exception as e:
            print(f"Dynamic Quantization Error: {e}")
            return onnx_path

    def quantize_int8_static(self, onnx_path: str) -> str:
        """Quantizzazione statica INT8"""
        print(f"2. Starting Standard Static INT8 Quantization on {onnx_path}...")

        pre_path = str(Path(onnx_path).with_stem(Path(onnx_path).stem + '_pre'))
        out_path = str(Path(onnx_path).with_stem(Path(onnx_path).stem + '_int8_static'))

        try:
            quant_pre_process(onnx_path, pre_path)
            dr = YoloDataReader(self.calib_img_dir)

            quantize_static(
                model_input=pre_path,
                model_output=out_path,
                calibration_data_reader=dr,
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.Entropy,
                per_channel=True,
                op_types_to_quantize=['Conv', 'MatMul', 'Gemm'],
            )

            print(f"Saved Static INT8 model: {out_path}")
            return out_path

        except Exception as e:
            print(f"Static Quantization Error: {e}")
            return onnx_path

'''
    #Pruning method
    def prune(self, amount=0.05):
        """
        Applica PRUNING STRUTTURATO in stile Torch-Pruning su YOLOv8,
        usando la sostituzione C2f -> C2f_v2 come nella repo di VainF/Hyunseok.
        """
        print(f"0. Applicazione Pruning Strutturato (amount={amount})...")


        # 1. Recupera il modello torch interno (già addestrato/finetunato)
        model = self.model.model
        device = next(model.parameters()).device

        # 2. Sostituzione C2f -> C2f_v2 (come nell’issue #147)
        print("Sostituzione blocchi C2f con C2f_v2...")
        replace_c2f_with_c2f_v2(model)  # usa esattamente la versione testata in repo

        # 3. Input di esempio per costruire il grafo
        example_inputs = torch.randn(1, 3, 480, 480).to(device)

        # 4. Ignora la head Detect (per non cambiare nc/reg_max)
        ignored_layers = [m for m in model.modules() if isinstance(m, Detect)]
        print(f"Moduli Detect ignorati: {len(ignored_layers)}")

        # 5. Importanza + pruner (come suggerito da Torch-Pruning per YOLO)
        imp = tp.importance.MagnitudeImportance(p=2)

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            pruning_ratio=amount,
            global_pruning=False,  # pruning locale
            ignored_layers=ignored_layers,
            round_to=8,  # multipli di 8 (puoi toglierlo se serve più libertà)
        )

        print("Analisi Grafo...")
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"   Original: {base_nparams / 1e6:.3f} M Params")

        # (opzionale) debug: quanti gruppi trova?
        groups = list(pruner.DG.get_all_groups(ignored_layers=ignored_layers,
                                               root_module_types=pruner.root_module_types))
        print(f"Gruppi prunabili trovati: {len(groups)}")

        # 6. ESECUZIONE del pruning
        pruner.step()

        # 7. Verifica riduzione
        new_macs, new_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"   Pruned:   {new_nparams / 1e6:.3f} M Params")
        reduction = (1 - new_nparams / base_nparams) * 100
        print(f"   -> Riduzione Reale: {reduction:.2f}%")

        if reduction < 1.0:
            print("⚠️ ATTENZIONE: Riduzione minima. Prova ad alzare l'amount o a modificare la strategia.")

        # 8. Salvataggio Checkpoint
        if not hasattr(self.model, 'ckpt') or self.model.ckpt is None:
            checkpoint = {'model': deepcopy(model), 'epoch': -1, 'train_args': {}}
        else:
            checkpoint = self.model.ckpt
            checkpoint['model'] = deepcopy(model)
            for k in ['optimizer', 'ema', 'updates']:
                checkpoint.pop(k, None)

        pruned_path = str(
            Path(self.model_path).with_stem(
                Path(self.model_path).stem + f'_pruned_struct{int(amount * 100)}'
            )
        )

        torch.save(checkpoint, pruned_path)
        print(f"✓ Modello salvato: {pruned_path}")

        # 9. Ricarica il modello YOLO pruned
        self.model = YOLO(pruned_path)
        return pruned_path

    def prepare_split_datasets(self):
        """
        Divide il dataset val2017 in due metà.
        Restituisce i PERCORSI ASSOLUTI ai file yaml.
        """
        print("Preparazione Split Dataset (2500 Train / 2500 Test)...")

        # 1. Trova le immagini
        with open(self.dataset_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # FIX: Usa .resolve() per ottenere il percorso assoluto della base
        # Così non importa se lanci lo script da 'scripts/' o dalla root
        base_path = Path('./datasets/coco').resolve()

        # Tentativo 1: Path standard
        img_dir = base_path / "images/val2017"

        # Raccogli tutte le immagini
        all_imgs = sorted(list(img_dir.glob("*.jpg")))

        if not all_imgs:
            raise FileNotFoundError(f"Nessuna immagine trovata in {img_dir}")

        # 2. Split 70/30
        split_idx = int(len(all_imgs) * 0.7)
        train_imgs = all_imgs[:split_idx]
        test_imgs = all_imgs[split_idx:]

        # 3. Scrivi i file TXT
        train_txt = base_path / "fine_tune_list.txt"
        test_txt = base_path / "benchmark_list.txt"

        # FIX: Scriviamo i percorsi ASSOLUTI delle immagini nel txt
        # YOLO a volte si confonde con i percorsi relativi nei file txt
        with open(train_txt, 'w', encoding='utf-8') as f:
            for p in train_imgs: f.write(str(p.resolve()) + '\n')

        with open(test_txt, 'w', encoding='utf-8') as f:
            for p in test_imgs: f.write(str(p.resolve()) + '\n')

        # 4. Crea i file YAML
        # Config per Fine Tuning
        ft_config = config.copy()
        ft_config['path'] = str(base_path)
        ft_config['train'] = str(train_txt)
        ft_config['val'] = str(train_txt)

        # Config per Benchmark
        bm_config = config.copy()
        bm_config['path'] = str(base_path)
        bm_config['val'] = str(test_txt)

        # Definiamo i path dei file yaml
        ft_path = base_path / "fine_tune.yaml"
        bm_path = base_path / "benchmark_test.yaml"

        # Salva i nuovi yaml
        with open(ft_path, 'w', encoding='utf-8') as f:
            yaml.dump(ft_config, f)

        with open(bm_path, 'w', encoding='utf-8') as f:
            yaml.dump(bm_config, f)

        print(f"✓ Dataset Splittati:\n  - Fine-Tune: {len(train_imgs)} imgs\n  - Benchmark: {len(test_imgs)} imgs")

        # FIX CRUCIALE: Restituisci stringhe di percorsi ASSOLUTI
        # Questo risolve l'errore "does not exist" che avevi prima.
        return str(ft_path.resolve()), str(bm_path.resolve())


    def fine_tune(self, epochs=3, batch=4, lr=5e-5):
        """Fine-Tuning"""
        print(f"0b. Avvio Fine-Tuning ({epochs} epochs)...")
        results = None
        try:
            results = self.model.train(
                data=self.dataset_yaml,  # Userà il file settato nel main (fine_tune.yaml)
                epochs=epochs,
                imgsz=480,
                batch=batch,
                lr0=lr,
                plots=True,
                verbose=True,
                device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                optimizer='AdamW',
                weight_decay=0.001,
                fliplr=0.5,  # Flip Orizzontale: SI (molto utile)
                flipud=0.0,  # Flip Verticale: NO (le persone non camminano a testa in giù)
                scale=0.5,  # Zoom in/out: +/- 50%
                translate=0.1,  # Spostamento laterale: +/- 10%

                # 3. Augmentations "Distruttive" (MOSAIC)
                # Per il Fine-Tuning post-pruning, riduciamo il Mosaic.
                # Default è 1.0 (sempre attivo). Noi mettiamo 0.5 o 0.0.
                mosaic=0.5,

                # Disattiviamo cose troppo aggressive per il fine-tuning
                mixup=0.0,
                copy_paste=0.0,
                warmup_epochs=0
            )

            best_model_path = self.model.trainer.best
            print(f"✓ Best Fine-Tuned Model saved at: {best_model_path}")
            self.model = YOLO(best_model_path)
            print("✓ Fine-Tuning completato.")

        except Exception as e:
            print(f"⚠️ Errore Fine-Tuning: {e}")

        return results
'''
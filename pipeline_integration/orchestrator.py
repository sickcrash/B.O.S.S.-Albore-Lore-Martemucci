import os
import torch
from source_adapter.file_video_source import FileVideoSource
from source_adapter.image_folder_source import ImageFolderSource
from pipeline_integration.logging_system import LoggingSystem
from dataset_builder import RealBDDImageDataset, RealBDDAVideoDataset
from model_builder import EnhancedTemporalSSD

class PipelineOrchestrator:
    def __init__(self, source_type: str, source_path: str, model_path: str, device: str = "cuda"):
        self.source_type = source_type
        self.source_path = source_path
        self.source = None
        self.device = device
        self.model_path = model_path
        self.logger = LoggingSystem()

    # Configura la sorgente video o immagini
    def setup_source(self):
        if self.source_type == "video":
            if os.path.isfile(self.source_path):
                self.source = FileVideoSource(self.source_path)
                self.source.open()
            else:
                self.source = RealBDDAVideoDataset(root_dir=self.source_path, clip_length=8)
        elif self.source_type == "image_folder":
            if os.path.isdir(self.source_path):
                self.source = ImageFolderSource(self.source_path)
                self.source.open()
            else:
                self.source = RealBDDImageDataset(
                    annotations_path=os.path.join(self.source_path, "annotations.json"),
                    img_root=self.source_path,
                    split="train",
                    is_train=False
                )
        else:
            raise ValueError(f"Tipo di sorgente non supportato: {self.source_type}")

    # Configura il modello
    def setup_model(self):
        try:
            self.model = EnhancedTemporalSSD(
                num_classes=2,
                device=self.device,
                use_adapters=True,
                use_temporal=True,
                use_weather_aug=False
            ).to(self.device)

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print(f"✅ Checkpoint caricato correttamente: {self.model_path}")
            else:
                print("⚠️ Struttura checkpoint non riconosciuta. Uso pesi non addestrati.")
            
            self.model.eval()
        except Exception as e:
            print(f"❌ Errore durante il caricamento del modello: {e}")
            print("⚠️ Uso pesi non addestrati.")

    # Debug
    def debug_pytorch(self):
        print(f"Versione di PyTorch: {torch.__version__}")
        print(f"CUDA disponibile: {torch.cuda.is_available()}")
        print(f"Numero di GPU disponibili: {torch.cuda.device_count()}")

    # Esegue la pipeline
    def run(self):
        try:
            self.debug_pytorch()
            self.setup_source()
            self.setup_model()
            self.logger.start()

            frame_count = 0
            for data in self.source:
                if isinstance(data, dict):  # Sorgenti esterne
                    images = [torch.tensor(data["image"]).permute(2, 0, 1).to(self.device).float() / 255.0]
                    meta = data["meta"]
                    targets = [{}]  # Target vuoto per inferenza
                elif isinstance(data, tuple):
                    clip_rgb, clip_gazemap, target, path = data
                    images = [frame.to(self.device) for frame in clip_rgb]
                    meta = {"source": path}
                    targets = [{}]
                else:
                    raise ValueError(f"Formato dati non supportato: {type(data)}")

                # Inferenza
                detection_results = self.model(images, targets)

                formatted_results = []
                for result in detection_results:
                    formatted_results.append({
                        'boxes': result['boxes'].cpu().detach().numpy(),
                        'scores': result['scores'].cpu().detach().numpy(),
                        'labels': result['labels'].cpu().detach().numpy()
                    })

                # Logging
                self.logger.log(meta, formatted_results)
                frame_count += 1

            fps, total_frames, elapsed_time = self.logger.end()

            print("\n" + "=" * 80)
            print(f"🎯 RISULTATI FINALI DELLA PIPELINE 🎯")
            print(f"📊 FPS: {fps:.2f}")
            print(f"📈 Frame totali elaborati: {total_frames}")
            print(f"⏱️ Tempo totale: {elapsed_time:.2f} secondi")
            print("=" * 80 + "\n")

        except Exception as e:
            print(f"❌ Errore durante l'esecuzione della pipeline: {e}")
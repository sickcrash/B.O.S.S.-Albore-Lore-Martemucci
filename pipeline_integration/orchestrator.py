import torch
import time
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from source_adapter.file_video_source import FileVideoSource
from source_adapter.image_folder_source import ImageFolderSource
from pipeline_integration.logging_system import LoggingSystem
from torchvision.models.detection import ssd300_vgg16

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
            self.source = FileVideoSource(self.source_path)
        elif self.source_type == "image_folder":
            self.source = ImageFolderSource(self.source_path)
        else:
            raise ValueError(f"Tipo di sorgente non supportato: {self.source_type}")

    # Configura il modello
    def setup_model(self):
        try:
            weights = SSD300_VGG16_Weights.COCO_V1
            self.model = ssd300_vgg16(weights=weights).to(self.device)
            
            # Caricamento robusto del checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print(f"✅ Checkpoint caricato correttamente: {self.model_path}")
            else:
                print("⚠️ Struttura checkpoint non riconosciuta. Uso pesi predefiniti.")
        except Exception as e:
            print(f"❌ Errore durante il caricamento del checkpoint: {e}")
            print("⚠️ Uso pesi predefiniti.")
            weights = SSD300_VGG16_Weights.COCO_V1
            self.model = ssd300_vgg16(weights=weights).to(self.device)
        
        self.model.eval()
        print(f"Device utilizzato dal modello: {next(self.model.parameters()).device}")

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
            self.source.open()
            self.setup_model()
            self.logger.start()
            self.start_time = time.time()
            frame_count = 0

            for frame in self.source.frames():
                frame_start_time = time.time()

                # Lettura e pre-elaborazione
                start_time = time.time()
                image_tensor = torch.tensor(frame["image"]).permute(2, 0, 1).to(self.device)
                image_tensor = image_tensor.float() / 255.0
                preproc_time = time.time()

                # Elaborazione del modello
                detection_results = self.model([image_tensor])[0]  # Usa il metodo forward
                model_time = time.time()

                # Formatta l'output per la pipeline
                formatted_results = {
                    'boxes': detection_results['boxes'].cpu().detach().numpy(),
                    'scores': detection_results['scores'].cpu().detach().numpy(),
                    'labels': detection_results['labels'].cpu().detach().numpy()
                }
                self.logger.log(frame["meta"], formatted_results)
                log_time = time.time()

                frame_count += 1

                frame_end_time = time.time()
                print(f"Frame {frame_count}: Tempo totale: {frame_end_time - frame_start_time:.4f}s "
                      f"(Lettura: {preproc_time - start_time:.4f}s, "
                      f"Modello: {model_time - preproc_time:.4f}s, "
                      f"Logging: {log_time - model_time:.4f}s)")

            self.source.close()
            fps, total_frames, elapsed_time = self.logger.end()
            print(f"Pipeline completata. FPS: {fps:.2f} (Frame totali: {total_frames}, Tempo totale: {elapsed_time:.2f}s)")

        except Exception as e:
            print(f"Errore durante l'esecuzione della pipeline: {e}")
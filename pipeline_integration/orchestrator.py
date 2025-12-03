import torch
import time
from source_adapter.file_video_source import FileVideoSource
from source_adapter.image_folder_source import ImageFolderSource
from pipeline_integration.logging_system import LoggingSystem
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
            self.source = FileVideoSource(self.source_path)
        elif self.source_type == "image_folder":
            self.source = ImageFolderSource(self.source_path)
        else:
            raise ValueError(f"Tipo di sorgente non supportato: {self.source_type}")

    # Configura il modello
    def setup_model(self):
        self.model = EnhancedTemporalSSD(num_classes=2, device=self.device).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=False)["model_state_dict"])
        self.model.eval()

        # Debug
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
                detection_results = self.model.forward_single_frame(image_tensor)
                model_time = time.time()

                # Logging
                self.logger.log(frame["meta"], detection_results)
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
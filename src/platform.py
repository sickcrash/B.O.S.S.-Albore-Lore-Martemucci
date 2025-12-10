"""Edge Detection Platform - Main"""
from typing import List, Dict, Tuple
import numpy as np
import time

from .patterns import Preprocessor, PostProcessor, InferenceBackend, HardwareIO


class EdgePlatform:
    """Piattaforma edge detection con pattern integrati"""

    def __init__(self,
                 preprocessor: Preprocessor,
                 postprocessor: PostProcessor,
                 backend: InferenceBackend,
                 input_size: Tuple[int, int] = (480, 480)):
        # Dependency Injection
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.backend = backend
        self.hardware_io = HardwareIO()
        self.input_size = input_size

    def set_backend(self, backend: InferenceBackend):
        """Strategy: cambia backend a runtime"""
        self.backend = backend
        print(f"Backend: {backend.get_name()}")

    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> Tuple[List[Dict], float]:
        """Esegue detection"""
        # Preprocess
        start = time.time()

        processed = self.preprocessor.process(image, self.input_size)

        # Inference
        predictions = self.backend.infer(processed)
        inf_time = time.time() - start

        # Postprocess
        detections = self.postprocessor.process(predictions, conf, iou)

        return detections, inf_time

    def detect_from_file(self, file_path: str) -> Tuple[List[Dict], float]:
        """Detection da file (Facade)"""
        image = self.hardware_io.read_from_file(file_path)
        return self.detect(image) if image is not None else ([], 0.0)

    def benchmark(self, test_images: List[np.ndarray], num_runs: int = 100) -> Dict:
        """Benchmark performance"""
        times = []

        # Warmup
        for _ in range(10):
            self.detect(test_images[0])

        # Benchmark
        for i in range(num_runs):
            img = test_images[i % len(test_images)]
            _, t = self.detect(img)
            times.append(t)

        times = np.array(times)

        return {
            'backend': self.backend.get_name(),
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'fps': float(1.0 / np.mean(times))
        }

    def cleanup(self):
        """Rilascia risorse"""
        self.hardware_io.release()

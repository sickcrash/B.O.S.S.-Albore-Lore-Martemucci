from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
import torch
from ultralytics import YOLO
import onnxruntime as ort
import cv2
from typing import Optional
from pathlib import Path


# ============================================================================
# DEPENDENCY INJECTION - Preprocessors
# ============================================================================

class Preprocessor(ABC):
    """Interfaccia per preprocessing"""

    @abstractmethod
    def process(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        pass


class StandardPreprocessor(Preprocessor):
    """Preprocessing standard"""

    def process(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        img = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)


class FastPreprocessor(Preprocessor):
    """Preprocessing ottimizzato per edge"""

    def process(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        img = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img * (1.0 / 255.0)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)


# ============================================================================
# DEPENDENCY INJECTION - Postprocessors
# ============================================================================

class PostProcessor(ABC):
    """Interfaccia per post-processing (NMS)"""

    @abstractmethod
    def process(self, predictions: np.ndarray, conf: float, iou: float) -> List[Dict]:
        pass


class StandardNMS(PostProcessor):
    """NMS standard"""

    def process(self, predictions: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[Dict]:
        detections = []
        pred = predictions[0] if len(predictions.shape) == 3 else predictions

        mask = pred[:, 4] > conf
        filtered = pred[mask]

        if len(filtered) > 0:
            boxes = filtered[:, :4].tolist()
            scores = filtered[:, 4].tolist()
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)

            for i in indices:
                idx = i if isinstance(i, int) else i[0]
                detections.append({
                    'bbox': filtered[idx, :4].tolist(),
                    'confidence': float(filtered[idx, 4]),
                    'class_id': int(filtered[idx, 5:].argmax()) if filtered.shape[1] > 5 else 0
                })

        return detections


class FastNMS(PostProcessor):
    """NMS veloce per edge"""

    def __init__(self, max_det: int = 100, top_k: int = 20):
        self.max_det = max_det
        self.top_k = top_k

    def process(self, predictions: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[Dict]:
        detections = []
        pred = predictions[0] if len(predictions.shape) == 3 else predictions

        boxes = pred[:, :4]

        # Le classi sono dalla colonna 4 in poi
        scores_matrix = pred[:, 4:]

        # Troviamo il punteggio massimo e l'ID della classe per ogni riga
        class_ids = np.argmax(scores_matrix, axis=1)
        scores = np.max(scores_matrix, axis=1)

        # 1. Filtro per confidenza
        mask = scores > conf
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []


        # OpenCV NMS vuole top_left_x, top_left_y, width, height.
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2

        # 3. NMS di OpenCV
        # NMSBoxes vuole liste, non numpy
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou)

        if len(indices) == 0:
            return []

            # Appiattiamo qualsiasi output (lista di liste, array numpy, ecc) in un array 1D
        indices = np.array(indices).flatten()

        for idx in indices:
            # Ripristina coordinate x1,y1,x2,y2 per il disegno
            box = boxes[idx]
            x, y, w, h = box

            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': float(scores[idx]),
                'class_id': int(class_ids[idx])
            })

        return detections[:self.max_det]

# ============================================================================
# STRATEGY PATTERN - Inference Backends
# ============================================================================

class InferenceBackend(ABC):
    """Interfaccia per backend di inferenza

    CONTRATTO: Tutti i backend devono restituire raw output della rete
    nel formato (batch, num_predictions, num_features) dove:
    - batch: dimensione batch (tipicamente 1)
    - num_predictions: numero di anchor/prediction (es. 8400 per YOLOv8)
    - num_features: [cx, cy, w, h, class_0, class_1, ..., class_N]
    """

    @abstractmethod
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Esegue inferenza e restituisce output grezzo (senza NMS)

        Args:
            input_data: Input preprocessato (batch, channels, height, width)

        Returns:
            np.ndarray: Raw output (batch, num_predictions, num_features)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class TorchBackend(InferenceBackend):
    """Backend PyTorch"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.yolo= YOLO(model_path)
        self.yolo.to(device)

        # Accesso al modello PyTorch sottostante
        self.model = self.yolo.model
        self.device = device

        self.input_shape = (1, 3, 480, 480)

        # Warmup
        self._warmup()

    def _warmup(self):
        """Warmup per allocare memoria GPU/CPU"""
        try:
            dummy = torch.zeros(self.input_shape, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy)
        except Exception as e:
            print(f"Warning: Warmup fallito: {e}")

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Inferenza senza post-processing ultralytics"""

        with torch.no_grad():
            x = torch.from_numpy(input_data).to(self.device)
            outputs = self.model(x)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            raw_output = outputs.cpu().numpy()

            # Trasponi se necessario: (batch, 84, 8400) → (batch, 8400, 84)
            if raw_output.shape[1] < raw_output.shape[2]:
                raw_output = np.transpose(raw_output, (0, 2, 1))

            return raw_output

    def get_name(self) -> str:
        return "PyTorch"

class ONNXBackend(InferenceBackend):
    """Backend ONNX Runtime"""

    def __init__(self, model_path: str, num_threads: int = 4):
        # --- CONFIGURAZIONE PERFORMANCE ---
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Log livello errori (opzionale)
        sess_options.log_severity_level = 3

        # Inizializza sessione
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # Metadati I/O
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Input shape dal modello
        input_shape = self.session.get_inputs()[0].shape
        # Gestisci batch dinamico (-1, 3, 480, 480) → (1, 3, 480, 480)
        self.input_shape = tuple(
            1 if (isinstance(dim, str) or dim == -1) else dim
            for dim in input_shape
        )

        # Rileva dtype richiesto dall'input
        input_type = self.session.get_inputs()[0].type
        if 'float16' in input_type:
            self.input_dtype = np.float16
            print(f"Modello richiede input FP16")
        else:
            self.input_dtype = np.float32
            print(f"Modello richiede input FP32")

        print(f"[ONNX] Threads: {num_threads}, Input shape: {self.input_shape}")

        # Warmup
        self._warmup()

    def _warmup(self):
        """Warmup per ottimizzare allocazioni"""
        try:
            #Usa il dtype corretto per il warmup
            dummy = np.zeros(self.input_shape, dtype=self.input_dtype)
            self.session.run([self.output_name], {self.input_name: dummy})
        except Exception as e:
            print(f"Warning: Warmup fallito: {e}")

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Inferenza ONNX"""
        # Converti al dtype richiesto dal modello
        if input_data.dtype != self.input_dtype:
            input_data = input_data.astype(self.input_dtype)

        # Esegui inferenza
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        output = outputs[0]

        # Trasponi se necessario: (batch, 84, 8400) → (batch, 8400, 84)
        if output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))

        return output

    def get_name(self) -> str:
        return "ONNX Runtime"


class HardwareIO:
    """Facade per gestire I/O hardware (versione ottimizzata)"""

    def __init__(self):
        self.camera = None
        self.current_camera_id = None

    def read_from_camera(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """Legge frame da camera con gestione corretta delle risorse

        Args:
            camera_id: ID della camera (default 0)

        Returns:
            Frame BGR o None se errore
        """
        try:
            # Riapri solo se l'ID è cambiato
            if self.camera is None or self.current_camera_id != camera_id:
                self._release_camera()
                self.camera = cv2.VideoCapture(camera_id)
                self.current_camera_id = camera_id

                if not self.camera.isOpened():
                    raise RuntimeError(f"Impossibile aprire camera {camera_id}")

            ret, frame = self.camera.read()
            if not ret:
                print(f"Warning: Nessun frame disponibile dalla camera {camera_id}")
                return None

            return frame

        except Exception as e:
            print(f"Errore lettura camera: {e}")
            self._release_camera()
            return None

    def read_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """Legge immagine da file con validazione

        Args:
            file_path: Percorso del file

        Returns:
            Immagine BGR o None se errore
        """
        try:
            # Validazione path
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File non trovato: {file_path}")

            if not path.is_file():
                raise ValueError(f"Path non è un file: {file_path}")

            # Leggi immagine
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Impossibile decodificare immagine: {file_path}")

            return image

        except Exception as e:
            print(f"Errore lettura file: {e}")
            return None

    def read_from_buffer(self, buffer: bytes) -> Optional[np.ndarray]:
        """Legge immagine da buffer bytes con validazione

        Args:
            buffer: Buffer contenente immagine codificata (JPEG, PNG, etc.)

        Returns:
            Immagine BGR o None se errore
        """
        try:
            if not buffer:
                raise ValueError("Buffer vuoto")

            if not isinstance(buffer, (bytes, bytearray)):
                raise TypeError(f"Buffer deve essere bytes o bytearray, ricevuto {type(buffer)}")

            # Decodifica
            nparr = np.frombuffer(buffer, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Buffer non contiene un'immagine valida")

            return image

        except Exception as e:
            print(f"Errore decodifica buffer: {e}")
            return None

    def write_to_file(self, image: np.ndarray, file_path: str,
                      quality: int = 95) -> bool:
        """Salva immagine su file con controllo qualità

        Args:
            image: Immagine da salvare (BGR)
            file_path: Percorso di destinazione
            quality: Qualità JPEG (0-100, default 95)

        Returns:
            True se successo, False altrimenti
        """
        try:
            # Validazione input
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Immagine non valida")

            if image.size == 0:
                raise ValueError("Immagine vuota")

            # Crea directory se non esiste
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Parametri di compressione basati sull'estensione
            ext = path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif ext == '.png':
                # PNG compression level: 0-9 (0=no compression, 9=max)
                compression = int((100 - quality) / 11)  # Map 95->0, 0->9
                params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
            else:
                params = []

            # Salva
            success = cv2.imwrite(str(path), image, params)

            if not success:
                raise IOError(f"Impossibile salvare immagine: {file_path}")

            return True

        except Exception as e:
            print(f"Errore scrittura file: {e}")
            return False

    def encode_to_buffer(self, image: np.ndarray,
                         format: str = '.jpg',
                         quality: int = 95) -> Optional[bytes]:
        """Codifica immagine in buffer bytes

        Args:
            image: Immagine da codificare (BGR)
            format: Formato di output ('.jpg', '.png', etc.)
            quality: Qualità di compressione (0-100)

        Returns:
            Buffer bytes o None se errore
        """
        try:
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Immagine non valida")

            # Parametri di encoding
            if format in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif format == '.png':
                compression = int((100 - quality) / 11)
                params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
            else:
                params = []

            # Codifica
            success, buffer = cv2.imencode(format, image, params)

            if not success:
                raise ValueError(f"Impossibile codificare immagine in formato {format}")

            return buffer.tobytes()

        except Exception as e:
            print(f"Errore codifica buffer: {e}")
            return None

    def get_camera_properties(self) -> Optional[dict]:
        """Ottieni proprietà della camera corrente

        Returns:
            Dizionario con proprietà o None se camera non aperta
        """
        if self.camera is None or not self.camera.isOpened():
            return None

        return {
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.camera.get(cv2.CAP_PROP_FPS),
            'backend': self.camera.getBackendName()
        }

    def set_camera_resolution(self, width: int, height: int) -> bool:
        """Imposta risoluzione camera

        Args:
            width: Larghezza desiderata
            height: Altezza desiderata

        Returns:
            True se successo
        """
        if self.camera is None or not self.camera.isOpened():
            print("Camera non aperta")
            return False

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Verifica effettiva risoluzione impostata
        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w != width or actual_h != height:
            print(f"Warning: Risoluzione richiesta {width}x{height}, "
                  f"ottenuta {actual_w}x{actual_h}")

        return True

    def _release_camera(self):
        """Helper privato per rilasciare la camera"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.current_camera_id = None

    def release(self):
        """Rilascia tutte le risorse"""
        self._release_camera()
        cv2.destroyAllWindows()

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup automatico alla fine del context"""
        self.release()
        return False  # Non sopprimere eccezioni

    def __del__(self):
        """Cleanup in caso di distruzione dell'oggetto"""
        self.release()



import cv2
from source_adapter.interface import VideoSourceAdapter
from typing import Iterator, Dict, Any

# Adapter per sorgenti video da file
class FileVideoSource(VideoSourceAdapter):
    def __init__(self, path: str):
        self.path = path
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.width = None
        self.height = None
        self.buffer = []  # pre-carica dei frame

    # Apre il file video e pre-carica i frame
    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"Impossibile aprire il file video: {self.path}")
        
        # Estrai informazioni sul video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Pre-carica i frame nel buffer
        self.buffer = []
        frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = frame_index / self.fps if self.fps else None
            self.buffer.append({
                "image": frame,
                "meta": {
                    "source": self.path,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "resolution": (self.width, self.height),
                    "total_frames": self.frame_count
                }
            })
            frame_index += 1

    # Restituisce un iteratore sui frame del buffer
    def __iter__(self):
        return self.frames()

    def frames(self) -> Iterator[Dict[str, Any]]:
        for frame_data in self.buffer:
            yield frame_data

    # Chiude il file video
    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
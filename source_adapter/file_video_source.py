import cv2
from source_adapter.interface import VideoSourceAdapter
from typing import Iterator, Dict, Any

#Adapter per sorgenti video da file
class FileVideoSource(VideoSourceAdapter):
    def __init__(self, path: str):
        self.path = path
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.width = None
        self.height = None

    #Apre il file video
    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"Impossibile aprire il file video: {self.path}")
        
        # Estrai informazioni sul video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Restituisce un iteratore sui frame del video
    def frames(self) -> Iterator[Dict[str, Any]]:
        if self.cap is None:
            self.open()
        frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = frame_index / self.fps if self.fps else None
            yield {
                "image": frame,
                "meta": {
                    "source": self.path,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "resolution": (self.width, self.height),
                    "total_frames": self.frame_count
                }
            }
            frame_index += 1
    
    #Chiude il file video
    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
import cv2
from source_adapter.interface import VideoSourceAdapter
from typing import Iterator, Dict, Any

'''
    NON USATI
'''

#Adapter per sorgenti video da webcam
class WebcamSource(VideoSourceAdapter):
    def __init__(self, device_idx: int = 0):
        self.device_idx = device_idx
        self.cap = None

    #Apre la webcam
    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.device_idx)
        if not self.cap.isOpened():
            raise ValueError(f"Impossibile accedere alla webcam con indice: {self.device_idx}")

    #Restituisce un iteratore sui frame della webcam
    def frames(self) -> Iterator[Dict[str, Any]]:
        if self.cap is None:
            self.open()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield {"image": frame, "meta": {"device": self.device_idx}}
    
    #Chiude la webcam
    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
import cv2
from source_adapter.interface import VideoSourceAdapter
from typing import Iterator, Dict, Any

'''
    NON USATI
'''

#Adapter per sorgenti video da stream RTSP
class RTSPSource(VideoSourceAdapter):
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None

    #Apre lo stream RTSP
    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise ValueError(f"Impossibile accedere allo stream RTSP: {self.rtsp_url}")

    #Restituisce un iteratore sui frame dello stream RTSP
    def frames(self) -> Iterator[Dict[str, Any]]:
        if self.cap is None:
            self.open()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield {"image": frame, "meta": {"source": self.rtsp_url}}
    
    #Chiude lo stream RTSP
    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
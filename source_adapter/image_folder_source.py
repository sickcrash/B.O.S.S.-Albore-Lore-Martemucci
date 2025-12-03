import os
import cv2
from source_adapter.interface import VideoSourceAdapter
from typing import Iterator, Dict, Any

# Adapter per sorgenti video da cartella di immagini
class ImageFolderSource(VideoSourceAdapter):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.image_files = []
        self.index = 0

    # Legge i file immagine dalla cartella
    def open(self) -> None:
        if not os.path.exists(self.folder_path):
            raise ValueError(f"La cartella non esiste: {self.folder_path}")
        self.image_files = sorted(
            [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        )
        if not self.image_files:
            raise ValueError(f"Nessuna immagine trovata nella cartella: {self.folder_path}")

    # Restituisce un iteratore sui frame della cartella
    def frames(self) -> Iterator[Dict[str, Any]]:
        frame_index = 0
        for image_file in self.image_files:
            image = cv2.imread(image_file)
            yield {
                "image": image,
                "meta": {
                    "file_name": image_file,
                    "frame_index": frame_index
                }
            }
            frame_index += 1
    
    def close(self) -> None:
        self.image_files = []
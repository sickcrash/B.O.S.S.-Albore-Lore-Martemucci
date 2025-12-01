from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any

class VideoSourceAdapter(ABC):
    '''
    Classe astratta per sorgenti video.
    Tutti gli adapter concreti implementeranno questa interfaccia.
    '''
    @abstractmethod
    def open(self) -> None: #apre la sorgente video
        pass

    @abstractmethod
    def frames(self) -> Iterator[Dict[str, Any]]:
        '''
        Restituisce un iteratore sui frame.
        Ogni frame è rappresentato come un dizionario con:
        - 'image': il frame come array NumPy.
        - 'meta': metadati associati al frame.
        '''
        pass

    @abstractmethod
    def close(self) -> None: #chiude la sorgente video
        pass
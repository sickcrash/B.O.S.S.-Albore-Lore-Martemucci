import time

class LoggingSystem:
    def __init__(self, debug=False):  # True per abilitare il debug
        self.start_time = None
        self.frame_count = 0
        self.debug = debug
        self.last_frame_index = -1  # Per rilevare frame drop

    # Inizia il logging
    def start(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_index = -1

    # Registra informazioni sul frame e sui risultati della detection
    def log(self, frame_meta, detection_results):
        self.frame_count += 1
        timestamp = time.time() - self.start_time if self.start_time else None

        # Rilevamento frame drop
        current_frame_index = frame_meta.get("frame_index", -1)
        if self.last_frame_index != -1 and current_frame_index != self.last_frame_index + 1:
            print(f"[WARN] Frame drop rilevato: {current_frame_index} non sequenziale (ultimo frame: {self.last_frame_index}).")
        self.last_frame_index = current_frame_index

        if self.debug:  # Solo se il debug è abilitato
            print(f"Frame {current_frame_index} - Timestamp: {timestamp:.2f}s")
            print(f"Detection Results: {detection_results}")

    # Termina il logging
    def end(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        return fps, self.frame_count, elapsed_time
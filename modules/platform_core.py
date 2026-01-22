import time
import psutil
import os
import csv
import torch
import numpy as np


class BenchmarkPlatform:
    def __init__(self, model_adapter, dataset_loader, device):
        """
        :param model_adapter: Una classe che avvolge il modello (deve avere il metodo predict)
        :param dataset_loader: Una lista di percorsi immagini
        :param device: 'cpu' o 'cuda'
        """
        self.model = model_adapter
        self.dataset = dataset_loader
        self.device = device
        self.results = []
        self.process = psutil.Process(os.getpid())

    def measure_resources(self):
        #Misura RAM attuale in MB
        return self.process.memory_info().rss / (1024 ** 2)

    def run(self, output_csv="results_detailed.csv", output_report="report_overview.txt"):
        print(f"Avvio Benchmark su {len(self.dataset)} immagini con {self.model.name}...")

        print("Caricamento modello...")
        try:
            self.model.load_model(self.device)
        except Exception as e:
            print(f"ERRORE CRITICO nel caricamento modello: {e}")
            return

        total_inference_time = 0
        total_ram_usage = 0

        #Ciclo di predizione
        for i, img_path in enumerate(self.dataset):
            if not os.path.exists(img_path):
                print(f"Immagine non trovata: {img_path}")
                continue

            #Misura risorse (Prima)
            ram_before = self.measure_resources()
            #(perf_counter) cronometro per Python
            start_time = time.perf_counter()

            #Esecuzione modello
            try:
                prediction = self.model.predict(img_path)
            except Exception as e:
                print(f"Errore su immagine {img_path}: {e}")
                prediction = {"class_id": -1, "score": 0.0}

            #Misura Risorse (Dopo)
            end_time = time.perf_counter()
            ram_after = self.measure_resources()

            #Calcoli
            latency = (end_time - start_time) * 1000  #convertiamo secondi in ms
            ram_used = max(0, ram_after - ram_before)  #Delta RAM per questa inferenza

            #Salva il risultato singolo per il CSV
            row = {
                "Image_ID": i,
                "Image_Name": os.path.basename(img_path),
                "Latency_ms": round(latency, 2),
                "RAM_Total_MB": round(ram_after, 2),
                "Prediction_Class": prediction['class_id'],
                "Confidence": prediction['score']
            }
            self.results.append(row)

            total_inference_time += latency
            total_ram_usage += ram_after

            print(f"Img {i + 1}/{len(self.dataset)} - Latency: {latency:.2f}ms | RAM Total: {ram_after:.2f}MB")

        #Salvataggio output
        self._save_csv(output_csv)
        self._save_report(output_report, total_inference_time, total_ram_usage)

    def _save_csv(self, filename):
        if not self.results: return
        keys = self.results[0].keys()
        with open(filename, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.results)
        print(f"--> CSV dettagliato salvato in: {filename}")

    def _save_report(self, filename, total_time, total_ram):
        num_imgs = len(self.results)
        if num_imgs == 0: return

        avg_latency = total_time / num_imgs
        avg_ram = total_ram / num_imgs
        fps = 1000 / avg_latency if avg_latency > 0 else 0

        report = f"""
        ============================================
        B.O.S.S. BENCHMARKING REPORT
        ============================================
        Modello:     {self.model.name}
        Dataset:     {num_imgs} immagini
        Dispositivo: {self.device}

        METRICHE MEDIE:
        --------------------------------------------
        Latenza Media:      {avg_latency:.2f} ms
        Throughput (FPS):   {fps:.2f}
        RAM Media Occupata: {avg_ram:.2f} MB

        TOTALE:
        --------------------------------------------
        Tempo Totale Test:  {total_time / 1000:.2f} secondi
        ============================================
        """
        with open(filename, 'w') as f:
            f.write(report)
        print(f"--> Overview salvata in: {filename}")
        print(report)
#!/usr/bin/env python3
"""
Script di emergenza per scaricare e posizionare le etichette COCO.
"""
import requests
import zipfile
import shutil
import os
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as file, tqdm(
            desc="Scaricamento Labels",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(1024):
            size = file.write(data)
            bar.update(size)


def main():
    print("=" * 60)
    print("FORCE DOWNLOAD COCO LABELS")
    print("=" * 60)

    # 1. Setup Percorsi
    base_dir = Path("../datasets/coco")
    if not base_dir.exists():
        base_dir = Path("datasets/coco")

    # Crea la cartella base se non esiste
    base_dir.mkdir(parents=True, exist_ok=True)

    # Percorso finale dove vogliamo i txt
    dest_labels_dir = base_dir / "labels" / "val2017"

    # Pulizia preventiva: se esiste e vuota o corrotta, rimuovi
    if dest_labels_dir.exists():
        print(f"Pulizia vecchia cartella labels: {dest_labels_dir}")
        shutil.rmtree(base_dir / "labels", ignore_errors=True)

    # 2. Download Zip (46 MB)
    zip_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"
    zip_path = base_dir / "labels_temp.zip"

    print(f"Download da: {zip_url}")
    download_file(zip_url, zip_path)

    # Estrazione in cartella temporanea
    temp_extract_dir = base_dir / "temp_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir()

    print("Estrazione...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)


    # Cerchiamo la cartella 'val2017' che contiene i txt dentro la temp
    found_val_dir = list(temp_extract_dir.rglob("val2017"))

    source_val = None
    for d in found_val_dir:
        # Controlliamo se contiene txt
        if any(d.glob("*.txt")):
            source_val = d
            break

    if source_val:
        print(f"Trovate etichette in: {source_val}")

        # Creiamo la destinazione
        dest_labels_dir.parent.mkdir(parents=True, exist_ok=True)

        # Spostiamo
        if dest_labels_dir.exists(): shutil.rmtree(dest_labels_dir)
        shutil.move(str(source_val), str(dest_labels_dir))

        print(f"Spostate in: {dest_labels_dir}")

        # Facciamo lo stesso per il train (già che ci siamo, pesa poco)
        source_train = source_val.parent / "train2017"
        dest_train = base_dir / "labels" / "train2017"
        if source_train.exists():
            if dest_train.exists(): shutil.rmtree(dest_train)
            shutil.move(str(source_train), str(dest_train))
            print(f"Spostate train labels in: {dest_train}")

    else:
        print("ERRORE")

    # Pulizia file temporanei
    zip_path.unlink()
    shutil.rmtree(temp_extract_dir)

    # RIMUOVIAMO LA CACHE VECCHIA per forzare YOLO a ri-scansionare
    for cache in base_dir.rglob("*.cache"):
        cache.unlink()
        print("Cache rimossa.")

    # Verifica finale
    count = len(list(dest_labels_dir.glob("*.txt")))
    print(f"Operazione conclusa. File .txt in val2017: {count}")



if __name__ == "__main__":
    main()
"""
Script per scaricare e posizionare le etichette COCO.
"""
import requests
import zipfile
import shutil
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
    base_dir = Path("datasets/coco")
    if not base_dir.exists():
        base_dir = Path("../datasets/coco")

    base_dir.mkdir(parents=True, exist_ok=True)

    # Percorso finale dove vogliamo i txt
    dest_labels_dir = base_dir / "labels" / "val2017"

    # Pulizia preventiva
    if dest_labels_dir.exists():
        print(f"Pulizia vecchia cartella labels: {dest_labels_dir}")
        shutil.rmtree(base_dir / "labels", ignore_errors=True)

    # 2. Download Zip (46 MB)
    zip_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"
    zip_path = base_dir / "labels_temp.zip"

    print(f"Download da: {zip_url}")
    download_file(zip_url, zip_path)

    # 3. Estrazione diretta nella cartella base (non in temp)
    print("Estrazione...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)

    # 4. Lo zip estrae in "coco/labels/", quindi dobbiamo spostare
    extracted_labels = base_dir / "coco" / "labels"

    if extracted_labels.exists():
        print(f"Trovate labels estratte in: {extracted_labels}")

        # Sposta labels/train2017 e labels/val2017 nella posizione corretta
        final_labels_dir = base_dir / "labels"

        if final_labels_dir.exists():
            shutil.rmtree(final_labels_dir)

        shutil.move(str(extracted_labels), str(final_labels_dir))
        print(f"Labels spostate in: {final_labels_dir}")

        # Rimuovi la cartella "coco" extra
        extra_coco_dir = base_dir / "coco"
        if extra_coco_dir.exists():
            shutil.rmtree(extra_coco_dir)
            print("Rimossa cartella 'coco' ridondante")

    else:
        print(f"⚠️ ERRORE: Labels non trovate in {extracted_labels}")
        print("Contenuto estratto:")
        for item in base_dir.rglob("*"):
            print(f"  - {item}")

    # 5. Pulizia file temporanei
    zip_path.unlink()

    # 6. Rimuovi cache vecchia
    for cache in base_dir.rglob("*.cache"):
        cache.unlink()
        print(f"Cache rimossa: {cache}")

    # 7. Verifica finale
    final_val = base_dir / "labels" / "val2017"
    final_train = base_dir / "labels" / "train2017"

    if final_val.exists():
        count_val = len(list(final_val.glob("*.txt")))
        print(f"✅ File .txt in val2017: {count_val}")
    else:
        print(f"❌ val2017 non trovata!")

    if final_train.exists():
        count_train = len(list(final_train.glob("*.txt")))
        print(f"✅ File .txt in train2017: {count_train}")
    else:
        print(f"⚠️ train2017 non trovata")




if __name__ == "__main__":
    main()

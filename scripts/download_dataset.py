import argparse
import sys
import yaml
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics.data.utils import check_det_dataset


def download_file(url, destination):
    """Scarica un file con barra di progresso"""
    if destination.exists():
        print(f"File zip già esistente: {destination}")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    print(f"Download: {url}...")
    with open(destination, 'wb') as file, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def unzip_file(zip_path, extract_to):
    """Estrae lo zip"""
    print(f"Estrazione in: {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Estrazione completata.")


def main():
    parser = argparse.ArgumentParser(description='Scarica SOLO Validation Set COCO')
    parser.add_argument('--config', type=str, default='../datasets/coco.yaml',
                        help='Nome del file .yaml')
    args = parser.parse_args()

    print("=" * 60)
    print(f"DOWNLOAD SELECTIVE: COCO VAL2017 ONLY")
    print("=" * 60)

    # 1. Leggiamo il YAML per capire DOVE scaricare i dati
    yaml_path = Path(args.config)
    if not yaml_path.exists():
        print(f"Errore: File {yaml_path} non trovato.")
        return

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Risolviamo il path root del dataset
    # Se nel yaml 'path' è relativo (es ../datasets/coco), lo risolviamo rispetto al yaml
    if 'path' in config:
        root_dir = Path(config['path'])
        if not root_dir.is_absolute():
            # Se è relativo, assumiamo sia relativo alla posizione dello script o del yaml
            # Ultralytics di solito usa path relativi alla root di esecuzione.
            # Qui proviamo a risolvere ../datasets/coco
            root_dir = Path(config['path']).resolve()
    else:
        # Fallback se non c'è 'path'
        root_dir = Path('../datasets/coco').resolve()

    print(f"Cartella destinazione: {root_dir}")
    root_dir.mkdir(parents=True, exist_ok=True)

    # Creiamo sottocartelle
    (root_dir / 'images').mkdir(exist_ok=True)
    (root_dir / 'labels').mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # 2. SCARICA SOLO LE IMMAGINI DI VALIDAZIONE (1GB)
    # ---------------------------------------------------------
    val_url = "http://images.cocodataset.org/zips/val2017.zip"
    val_zip = root_dir / "val2017.zip"

    # Controlla se la cartella estratta esiste già
    if not (root_dir / 'images' / 'val2017').exists():
        download_file(val_url, val_zip)
        unzip_file(val_zip, root_dir / 'images')
        # Pulizia zip
        if val_zip.exists(): val_zip.unlink()
    else:
        print("Immagini val2017 già presenti.")

    labels_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"
    labels_zip = root_dir / "coco2017labels.zip"

    if not (root_dir / 'labels' / 'val2017').exists():
        download_file(labels_url, labels_zip)

        unzip_file(labels_zip, root_dir)

        # Pulizia
        if labels_zip.exists(): labels_zip.unlink()

        # Fix struttura: a volte lo zip crea una cartella 'coco' intermedia
        if (root_dir / 'coco').exists():
            print("System: Spostamento file dalla sottocartella 'coco'...")

    else:
        print("Labels val2017 già presenti.")

    print("\n" + "=" * 60)
    print("VERIFICA STRUTTURA")

    val_path = root_dir / 'images' / 'val2017'
    if val_path.exists() and any(val_path.iterdir()):
        print(f"Validation Set OK: {val_path}")
        print(f"Contiene {len(list(val_path.glob('*.jpg')))} immagini.")
    else:
        print("Qualcosa è andato storto, la cartella val2017 è vuota o assente.")


if __name__ == "__main__":
    main()
import argparse
import glob
import os
import sys
from platform_core import BenchmarkPlatform
from adapters import ResNetAdapter, EfficientNetAdapter

#Configurazione pesi
FILE_RESNET = "resnet.pth"
FILE_EFFICIENT = "efficient_natural.pth"


def main():
    #Configurazione argomenti da linea di comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "efficientnet"],
                        help="Quale modello testare")
    args = parser.parse_args()

    #Controllo Dataset
    dataset_files = glob.glob("dataset_test/*.jpg")
    if not dataset_files:
        print("ERRORE: Nessuna immagine trovata in dataset_test/")
        sys.exit(1)

    #ResNet
    if args.model == "resnet":
        print("\n" + "=" * 40)
        print(" AVVIO TEST ISOLATO: RESNET")
        print("=" * 40)

        resnet_path = os.path.join("./checkpoints", FILE_RESNET)
        adapter = ResNetAdapter(resnet_path)

        platform = BenchmarkPlatform(adapter, dataset_files, device="cpu")
        platform.run("risultati_resnet.csv", "report_resnet.txt")

    #Efficient
    elif args.model == "efficientnet":
        print("\n" + "=" * 40)
        print(" AVVIO TEST ISOLATO: EFFICIENTNET")
        print("=" * 40)

        eff_path = os.path.join("./checkpoints", FILE_EFFICIENT)
        #num_classes=140 come dal training
        adapter = EfficientNetAdapter(eff_path, num_classes=140)

        platform = BenchmarkPlatform(adapter, dataset_files, device="cpu")
        platform.run("risultati_efficientnet.csv", "report_efficientnet.txt")


if __name__ == "__main__":
    main()
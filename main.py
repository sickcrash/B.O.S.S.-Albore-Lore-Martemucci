import argparse
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch.nn as nn

from dataset import CityscapesBinary
from train import run_training


def check_cityscapes_structure(root):
    left = os.path.join(root, "leftImg8bit")
    gt = os.path.join(root, "gtFine")
    if not os.path.isdir(left) or not os.path.isdir(gt):
        print("ERRORE: struttura Cityscapes non trovata.")
        print("Assicurati che la cartella contenga:")
        print("   {}/leftImg8bit/".format(root))
        print("   {}/gtFine/".format(root))
        sys.exit(1)
    print(f"Dataset Cityscapes trovato in: {root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to Cityscapes root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-h", type=int, default=512)
    parser.add_argument("--img-w", type=int, default=1024)
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    # controllo dataset
    check_cityscapes_structure(args.root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    train_ds = CityscapesBinary(args.root, split="train", img_size=(args.img_h, args.img_w))
    val_ds = CityscapesBinary(args.root, split="val", img_size=(args.img_h, args.img_w), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    # carica i pesi pre-addestrati su COCO
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet50(weights=weights, aux_loss=True)

    # sostituiamo l'ultimo layer per la segmentazione binaria (2 classi: sfondo/ostacolo)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    run_training(model, train_loader, val_loader, optimizer, scaler, device, args.epochs, args.ckpt_dir)

if __name__ == "__main__":
    main()
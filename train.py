import os
import time
import numpy as np
import torch
import torch.nn as nn

IGNORE_INDEX = 255

# Metriche
def fast_hist(pred, gt, num_classes=2, ignore_index=IGNORE_INDEX):
    mask = gt != ignore_index
    hist = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist

def compute_miou(model, loader, device):
    model.eval()
    hist = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)["out"]
            preds = out.argmax(1).cpu().numpy()
            gts = targets.cpu().numpy()
            hist += fast_hist(preds, gts)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    return np.nanmean(iu)

# Training loop
def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    running = 0.0
    start = time.time()

    for i, (imgs, targets) in enumerate(loader, 1):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
            out = model(imgs)["out"]
            loss = ce(out, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        if i % 20 == 0:
            print(f"Epoch {epoch} | Iter {i}/{len(loader)} | Loss {running/20:.4f}")
            running = 0.0
            start = time.time()

# Training entrypoint
def run_training(model, train_loader, val_loader, optimizer, scaler, device, epochs, ckpt_dir):
    best_miou = 0.0
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        miou = compute_miou(model, val_loader, device)
        print(f"[Val] Epoch {epoch} | mIoU: {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(ckpt_dir, f"best_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model at {ckpt_path}")

    print(f"Training done. Best mIoU: {best_miou:.4f}")
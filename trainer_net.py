import torch
import logging
from tqdm import tqdm
from torch.amp import GradScaler, autocast

def training_net(model, dataloader, num_epochs, device, optimizer, lr_scheduler, debug_steps=0):
    model.train()
    print("\nInizio addestramento...")

    scaler = GradScaler()

    for epoch in range(num_epochs):
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoca [{epoch+1}/{num_epochs}]")
        
        for i, (images, targets) in enumerate(progress_bar):
            if images is None or targets is None:
                continue
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast(device_type=device.type):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler is not None and isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                lr_scheduler.step()

            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
            
        if lr_scheduler is not None and not isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            lr_scheduler.step()
        
        num_batches_processed = i + 1
        avg_loss = epoch_loss / num_batches_processed
        
        print(f"Fine Epoca [{epoch+1}/{num_epochs}], Perdita media: {avg_loss:.4f}")
        logging.info(f"Epoca [{epoch+1}/{num_epochs}], Perdita media: {avg_loss:.4f}")

    return model
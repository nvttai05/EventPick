import os
import sys
import argparse
import random
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FaceDataset
from model import FaceEmbeddingModel
from losses import ArcFaceLoss
from transforms import train_transform, eval_transform
from utils import save_checkpoint, AverageMeter
from validate import validate


# ============================================================
# 1. ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\Hoctap\CK_KHDL\EventPick_v1\data\train",
        help="Train folder. Structure: train/event/person/images"
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default=r"D:\Hoctap\CK_KHDL\EventPick_v1\data\detected_faces",
        help="Val folder. Structure: detected_faces/Event_Public/person/images"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenetv2"],
        help="Backbone model"
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")

    # Lưu snapshot mỗi N epoch
    parser.add_argument("--save_every", type=int, default=5)

    return parser.parse_args()


# ============================================================
# 2. UTILS
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_dataset_dir(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {name}: {path}")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"{name} không phải folder: {path}")


# ============================================================
# 3. MAIN
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("TRAINING CONFIG")
    print("=" * 80)
    print("Using device:", device)

    if device == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    print("Train dir:", args.data_dir)
    print("Val dir:", args.val_dir)
    print("Model:", args.model_name)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Embedding dim:", args.embedding_dim)
    print("AMP:", args.use_amp)
    print("Save every:", args.save_every)
    print("=" * 80)

    # ------------------------------------------------------------
    # Check folders
    # ------------------------------------------------------------
    check_dataset_dir(args.data_dir, "train_dir")
    check_dataset_dir(args.val_dir, "val_dir")

    # ------------------------------------------------------------
    # Output dirs
    # ------------------------------------------------------------
    run_name = f"{args.model_name}_emb{args.embedding_dim}_bs{args.batch_size}_lr{args.lr}"

    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    log_dir = os.path.join(args.log_dir, run_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    print("\nLoading train dataset...")
    train_dataset = FaceDataset(
        root_dir=args.data_dir,
        transform=train_transform
    )

    print("\nLoading validation dataset...")
    val_dataset = FaceDataset(
        root_dir=args.val_dir,
        transform=eval_transform
    )

    num_classes = len(train_dataset.label_to_idx)

    print("\nDataset summary:")
    print(f"Train images : {len(train_dataset)}")
    print(f"Train classes: {num_classes}")
    print(f"Val images   : {len(val_dataset)}")
    print(f"Val classes  : {len(val_dataset.label_to_idx)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset rỗng.")

    if num_classes < 2:
        raise ValueError("Train dataset cần ít nhất 2 class để train ArcFace.")

    if len(val_dataset) == 0:
        raise ValueError("Val dataset rỗng. Kiểm tra lại val_dir.")

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------
    pin_memory = device == "cuda"
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # ------------------------------------------------------------
    # Model / Loss / Optimizer / Scheduler
    # ------------------------------------------------------------
    model = FaceEmbeddingModel(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        pretrained=True
    ).to(device)

    criterion = ArcFaceLoss(
        embedding_size=args.embedding_dim,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(args.use_amp and device == "cuda")
    )

    # ------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------
    best_val_acc = -1.0
    global_step = 0

    print("\nStart training from scratch...")
    print("=" * 80)

    for epoch in range(args.epochs):
        model.train()
        criterion.train()

        loss_meter = AverageMeter()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            colour="cyan",
            file=sys.stdout
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(args.use_amp and device == "cuda")
            ):
                embeddings = model(images)
                loss = criterion(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), images.size(0))

            current_lr = optimizer.param_groups[0]["lr"]

            progress_bar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "lr": f"{current_lr:.6f}"
            })

            writer.add_scalar("Train/Step_Loss", loss.item(), global_step)
            global_step += 1

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------
        model.eval()

        val_acc = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        scheduler.step()

        epoch_loss = loss_meter.avg
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Loss: {epoch_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
        writer.add_scalar("Train/Learning_Rate", current_lr, epoch)
        writer.add_scalar("Validation/Accuracy", val_acc, epoch)

        # --------------------------------------------------------
        # Save last checkpoint mỗi epoch
        # --------------------------------------------------------
        last_path = os.path.join(checkpoint_dir, "last.pth")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            checkpoint_path=last_path,
            criterion=criterion,
            scheduler=scheduler,
            best_val_acc=best_val_acc,
        )

        print(f"Saved last checkpoint: {last_path}")

        # --------------------------------------------------------
        # Save snapshot mỗi save_every epoch
        # --------------------------------------------------------
        is_snapshot_epoch = (
            (epoch + 1) % args.save_every == 0
            or (epoch + 1) == args.epochs
        )

        if is_snapshot_epoch:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"epoch_{epoch + 1}.pth"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                criterion=criterion,
                scheduler=scheduler,
                best_val_acc=best_val_acc,
            )

            print(f"Saved snapshot checkpoint: {checkpoint_path}")

        # --------------------------------------------------------
        # Save best checkpoint
        # --------------------------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            best_path = os.path.join(checkpoint_dir, "best.pth")

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                checkpoint_path=best_path,
                criterion=criterion,
                scheduler=scheduler,
                best_val_acc=best_val_acc,
            )

            print(f"Saved best checkpoint: {best_path}")

    writer.close()

    print("=" * 80)
    print("Training completed.")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
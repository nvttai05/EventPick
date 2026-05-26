import os
import argparse
import sys

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

# ARGUMENTS
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="D:/Projects/Study/School/Nam3_Ky2/KHDL/data/train")
    parser.add_argument('--val_dir', type=str, default='D:/Projects/Study/School/Nam3_Ky2/KHDL/data/val')
    parser.add_argument('--model_name', type=str, default='resnet50') # resnet50 or mobilenetv2
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')

    return parser.parse_args()

def main():

    args = parse_args()

    # CREATE DIRS
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(args.log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # DATASET
    train_dataset = FaceDataset(root_dir=args.data_dir, transform=train_transform)
    num_classes = len(train_dataset.label_to_idx)

    val_dataset = FaceDataset(root_dir=args.val_dir, transform=eval_transform)

    # DATALOADER
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # MODEL
    model = FaceEmbeddingModel(model_name=args.model_name, embedding_dim=args.embedding_dim, pretrained=False).to(device)

    criterion = ArcFaceLoss(embedding_size=args.embedding_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TRAIN LOOP
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()

        progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}', colour='cyan', file=sys.stdout)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)

            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # UPDATE METRICS
            loss_meter.update(loss.item(), images.size(0))

            progress_bar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': optimizer.param_groups[0]['lr']
            })

            writer.add_scalar('Train/Step_Loss', loss.item(), global_step)
            global_step += 1

        # Validation
        val_acc = validate(model, val_loader, device)

        # SCHEDULER
        scheduler.step()
        epoch_loss = loss_meter.avg
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f'Epoch [{epoch+1}/{args.epochs}] '
            f'Loss: {epoch_loss:.4f}'
            f'Val Accuracy: {val_acc:.4f}'
        )

        writer.add_scalar('Train/Epoch_Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch)

        # SAVE CHECKPOINT
        checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')

        save_checkpoint(model, optimizer, epoch, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')
    writer.close()
    print('Training completed.')

if __name__ == '__main__':
    main()
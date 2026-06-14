import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FaceDataset
from model import FaceEmbeddingModel
from transforms import eval_transform
from validate import validate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["resnet50", "mobilenetv2"]
    )

    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # Hiện tại model của bạn đang train với pretrained=False
    parser.add_argument("--pretrained", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy data_dir: {data_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("EVALUATE CHECKPOINT ON FINAL TEST")
    print("=" * 80)
    print("Data dir        :", data_dir)
    print("Checkpoint      :", checkpoint_path)
    print("Model           :", args.model_name)
    print("Embedding dim   :", args.embedding_dim)
    print("Pretrained flag :", args.pretrained)
    print("Device          :", device)
    print("=" * 80)

    dataset = FaceDataset(
        root_dir=str(data_dir),
        transform=eval_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    model = FaceEmbeddingModel(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = validate(
        model=model,
        loader=loader,
        device=device
    )

    print()
    print("=" * 80)
    print("FINAL TEST RESULT")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key:<15}: {value:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
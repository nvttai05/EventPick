import os
import csv
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import FaceDataset
from model import FaceEmbeddingModel
from transforms import eval_transform


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--val_dir",
        type=str,
        default=r"D:\Hoctap\CK_KHDL\EventPick_v1\data\val_final",
        help="Validation folder. Structure: root/event/person/images"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=r".\checkpoints\resnet50_emb512_bs64_lr5e-05",
        help="Folder chứa best.pth, last.pth, epoch_40.pth"
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best.pth", "last.pth", "epoch_40.pth"],
        help="Danh sách checkpoint cần test"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenetv2"]
    )

    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--gallery_per_person_list",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="Test nhiều số ảnh gallery/người"
    )

    parser.add_argument(
        "--num_trials",
        type=int,
        default=5,
        help="Số lần random split gallery/query cho mỗi checkpoint"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=r".\reports\checkpoint_random_val_results.csv"
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()

    all_embeddings = []
    all_labels = []
    all_paths = []

    dataset = loader.dataset

    for images, labels in tqdm(loader, desc="Extract embeddings", colour="yellow"):
        images = images.to(device, non_blocking=True)

        embeddings = model(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Lấy paths nếu dataset có samples
    if hasattr(dataset, "samples"):
        all_paths = [sample[0] for sample in dataset.samples]
    else:
        all_paths = [""] * len(labels)

    return embeddings, labels, all_paths


# ============================================================
# RANDOM GALLERY / QUERY SPLIT
# ============================================================

def build_gallery_query_split(labels, gallery_per_person, seed):
    rng = random.Random(seed)

    label_to_indices = defaultdict(list)

    labels_np = labels.numpy()

    for idx, label in enumerate(labels_np):
        label_to_indices[int(label)].append(idx)

    gallery_indices = []
    query_indices = []

    skipped_labels = 0

    for label, indices in label_to_indices.items():
        indices = indices.copy()

        if len(indices) <= gallery_per_person:
            skipped_labels += 1
            continue

        rng.shuffle(indices)

        gallery = indices[:gallery_per_person]
        query = indices[gallery_per_person:]

        gallery_indices.extend(gallery)
        query_indices.extend(query)

    if len(gallery_indices) == 0 or len(query_indices) == 0:
        return None

    return {
        "gallery_indices": torch.tensor(gallery_indices, dtype=torch.long),
        "query_indices": torch.tensor(query_indices, dtype=torch.long),
        "skipped_labels": skipped_labels,
        "num_gallery": len(gallery_indices),
        "num_query": len(query_indices),
    }


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def compute_recall_at_k(
    embeddings,
    labels,
    gallery_indices,
    query_indices,
    device,
    ks=(1, 5, 10),
):
    gallery_embeddings = embeddings[gallery_indices].to(device)
    query_embeddings = embeddings[query_indices].to(device)

    gallery_labels = labels[gallery_indices].to(device)
    query_labels = labels[query_indices].to(device)

    similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())

    max_k = min(max(ks), gallery_embeddings.shape[0])

    topk_indices = torch.topk(
        similarity_matrix,
        k=max_k,
        dim=1
    ).indices

    results = {}

    for k in ks:
        k = min(k, max_k)

        retrieved_labels = gallery_labels[topk_indices[:, :k]]

        correct = (retrieved_labels == query_labels.unsqueeze(1)).any(dim=1)

        results[f"recall@{k}"] = float(correct.float().mean().item())

    # similarity stats
    positive_sims = []
    negative_sims = []

    for i in range(len(query_labels)):
        q_label = query_labels[i]
        sims = similarity_matrix[i]

        positive_mask = gallery_labels == q_label
        negative_mask = gallery_labels != q_label

        if positive_mask.any():
            positive_sims.extend(sims[positive_mask].detach().cpu().tolist())

        if negative_mask.any():
            # lấy sample negative để không quá nặng
            neg_values = sims[negative_mask]
            if len(neg_values) > 100:
                neg_values = neg_values[torch.randperm(len(neg_values), device=device)[:100]]
            negative_sims.extend(neg_values.detach().cpu().tolist())

    results["positive_mean"] = float(np.mean(positive_sims)) if positive_sims else 0.0
    results["negative_mean"] = float(np.mean(negative_sims)) if negative_sims else 0.0
    results["positive_std"] = float(np.std(positive_sims)) if positive_sims else 0.0
    results["negative_std"] = float(np.std(negative_sims)) if negative_sims else 0.0

    return results


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_model_from_checkpoint(checkpoint_path, model_name, embedding_dim, device):
    model = FaceEmbeddingModel(
        model_name=model_name,
        embedding_dim=embedding_dim,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", None)
    best_val_acc = checkpoint.get("best_val_acc", None)

    return model, epoch, best_val_acc


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("EVALUATE CHECKPOINTS WITH RANDOM VAL SPLITS")
    print("=" * 80)
    print("Device:", device)
    if device == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0))

    print("Val dir:", args.val_dir)
    print("Checkpoint dir:", args.checkpoint_dir)
    print("Checkpoints:", args.checkpoints)
    print("Gallery per person:", args.gallery_per_person_list)
    print("Trials:", args.num_trials)
    print("=" * 80)

    val_dataset = FaceDataset(
        root_dir=args.val_dir,
        transform=eval_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    rows = []

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    for checkpoint_name in args.checkpoints:
        checkpoint_path = Path(args.checkpoint_dir) / checkpoint_name

        if not checkpoint_path.exists():
            print(f"[SKIP] Không tìm thấy checkpoint: {checkpoint_path}")
            continue

        print()
        print("=" * 80)
        print(f"Đang load checkpoint: {checkpoint_path}")
        print("=" * 80)

        model, checkpoint_epoch, checkpoint_best = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=args.model_name,
            embedding_dim=args.embedding_dim,
            device=device,
        )

        print("Checkpoint epoch:", checkpoint_epoch)
        print("Checkpoint best_val_acc:", checkpoint_best)

        embeddings, labels, paths = extract_embeddings(
            model=model,
            loader=val_loader,
            device=device,
        )

        print(f"Embeddings shape: {tuple(embeddings.shape)}")
        print(f"Labels shape: {tuple(labels.shape)}")

        for gallery_per_person in args.gallery_per_person_list:
            trial_results = []

            print()
            print("-" * 80)
            print(f"Checkpoint={checkpoint_name} | gallery_per_person={gallery_per_person}")
            print("-" * 80)

            for trial in range(args.num_trials):
                split_seed = args.seed + trial

                split = build_gallery_query_split(
                    labels=labels,
                    gallery_per_person=gallery_per_person,
                    seed=split_seed,
                )

                if split is None:
                    print("[WARNING] Không tạo được gallery/query split.")
                    continue

                metrics = compute_recall_at_k(
                    embeddings=embeddings,
                    labels=labels,
                    gallery_indices=split["gallery_indices"],
                    query_indices=split["query_indices"],
                    device=device,
                    ks=(1, 5, 10),
                )

                row = {
                    "checkpoint": checkpoint_name,
                    "checkpoint_epoch": checkpoint_epoch,
                    "checkpoint_best_val_acc": checkpoint_best,
                    "gallery_per_person": gallery_per_person,
                    "trial": trial,
                    "num_gallery": split["num_gallery"],
                    "num_query": split["num_query"],
                    "skipped_labels": split["skipped_labels"],
                    **metrics,
                }

                rows.append(row)
                trial_results.append(metrics)

                print(
                    f"trial={trial} | "
                    f"R@1={metrics['recall@1']:.4f} | "
                    f"R@5={metrics['recall@5']:.4f} | "
                    f"R@10={metrics['recall@10']:.4f} | "
                    f"pos={metrics['positive_mean']:.4f} | "
                    f"neg={metrics['negative_mean']:.4f}"
                )

            if trial_results:
                mean_r1 = np.mean([m["recall@1"] for m in trial_results])
                std_r1 = np.std([m["recall@1"] for m in trial_results])
                mean_r5 = np.mean([m["recall@5"] for m in trial_results])
                mean_r10 = np.mean([m["recall@10"] for m in trial_results])

                print(
                    f"[SUMMARY] {checkpoint_name} | gallery={gallery_per_person} | "
                    f"R@1={mean_r1:.4f}±{std_r1:.4f} | "
                    f"R@5={mean_r5:.4f} | R@10={mean_r10:.4f}"
                )

    if rows:
        fieldnames = list(rows[0].keys())

        with open(output_csv, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print()
        print("=" * 80)
        print("HOÀN TẤT EVALUATE")
        print("=" * 80)
        print(f"Đã lưu CSV: {output_csv}")
        print("=" * 80)


if __name__ == "__main__":
    main()
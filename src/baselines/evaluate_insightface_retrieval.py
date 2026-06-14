import os
import sys

# Tự động tìm và nạp thư mục chứa file DLL của CUDA và cuDNN trong môi trường ảo
venv_base = sys.prefix
cuda_bin = os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin")
cudnn_bin = os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cudnn", "bin")

if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)
if os.path.exists(cudnn_bin):
    os.add_dll_directory(cudnn_bin)
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from insightface import model_zoo


# ============================================================
# 1. ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\Hoctap\CK_KHDL\EventPick_v1\data\public_80_20\val",
        help="Dataset folder. Structure: root/event/person/images"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\Hoctap\CK_KHDL\EventPick_v1\reports\baselines\insightface",
        help="Output report folder"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="buffalo_l",
        help="InsightFace model pack name"
    )

    parser.add_argument(
        "--ctx_id",
        type=int,
        default=0,
        help="0 = GPU, -1 = CPU"
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="0 = use all images. Use small number for quick test."
    )

    return parser.parse_args()


# ============================================================
# 2. DATASET LOADING
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def collect_images(data_dir: Path):
    """
    Expected structure:
        root/event/person/image.jpg

    Label = event/person
    """
    records = []

    for event_dir in sorted(data_dir.iterdir()):
        if not event_dir.is_dir():
            continue

        for person_dir in sorted(event_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            label_name = f"{event_dir.name}/{person_dir.name}"

            for image_path in sorted(person_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    records.append({
                        "image_path": str(image_path),
                        "event": event_dir.name,
                        "person": person_dir.name,
                        "label_name": label_name,
                    })

    return records


def read_image_safe(image_path: str):
    """
    Đọc ảnh an toàn hơn cv2.imread trực tiếp.
    """
    path = Path(image_path)

    if not path.exists():
        return None

    try:
        data = np.fromfile(str(path), dtype=np.uint8)

        if data.size == 0:
            return None

        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        return image

    except Exception:
        return None


# ============================================================
# 3. INSIGHTFACE EMBEDDING
# ============================================================

def build_insightface_recognition_model(model_name: str, ctx_id: int):
    """
    Load trực tiếp InsightFace recognition model.
    Không dùng FaceAnalysis vì FaceAnalysis yêu cầu detection model.
    """

    model_path = (
        Path.home()
        / ".insightface"
        / "models"
        / model_name
        / "w600k_r50.onnx"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy recognition model: {model_path}\n"
            f"Bạn cần có buffalo_l đã download đầy đủ."
        )

    if ctx_id >= 0:
        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        providers = [
            "CPUExecutionProvider",
        ]

    print("=" * 80)
    print("KHỞI TẠO INSIGHTFACE RECOGNITION MODEL TRỰC TIẾP")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"ctx_id: {ctx_id}")
    print(f"Providers: {providers}")
    print("=" * 80)

    rec_model = model_zoo.get_model(
        str(model_path),
        providers=providers,
    )

    rec_model.prepare(ctx_id=ctx_id)

    print("[OK] InsightFace recognition model đã sẵn sàng.")
    print("=" * 80)

    return rec_model


def extract_single_embedding(rec_model, image_bgr: np.ndarray):
    """
    Extract embedding từ ảnh face đã processed.

    InsightFace w600k_r50 nhận input 112x112.
    Data của mình đang 224x224 nên resize về 112x112.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Ảnh rỗng.")

    image_112 = cv2.resize(
        image_bgr,
        (112, 112),
        interpolation=cv2.INTER_AREA,
    )

    feat = rec_model.get_feat(image_112)

    feat = np.asarray(feat)

    if feat.ndim == 2:
        feat = feat[0]

    feat = feat.astype(np.float32)

    norm = np.linalg.norm(feat)

    if norm < 1e-12:
        raise ValueError("Embedding norm quá nhỏ.")

    feat = feat / norm

    return feat


def extract_embeddings(records, rec_model):
    embeddings = []
    kept_records = []
    failed_records = []

    start_time = time.perf_counter()

    for record in tqdm(records, desc="Extract InsightFace embeddings"):
        image_path = record["image_path"]

        try:
            image = read_image_safe(image_path)

            if image is None:
                raise ValueError("Không đọc được ảnh.")

            embedding = extract_single_embedding(
                rec_model=rec_model,
                image_bgr=image,
            )

            embeddings.append(embedding)
            kept_records.append(record)

        except Exception as error:
            failed_record = record.copy()
            failed_record["error"] = str(error)
            failed_records.append(failed_record)

    runtime = time.perf_counter() - start_time

    if len(embeddings) == 0:
        raise RuntimeError("Không extract được embedding nào.")

    embeddings = np.vstack(embeddings).astype(np.float32)

    return embeddings, kept_records, failed_records, runtime


# ============================================================
# 4. RETRIEVAL METRICS
# ============================================================

def encode_labels(records):
    label_names = [record["label_name"] for record in records]

    unique_labels = sorted(set(label_names))
    label_to_idx = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    labels = np.array(
        [label_to_idx[label] for label in label_names],
        dtype=np.int64,
    )

    return labels, label_to_idx


def compute_recall_at_k(embeddings, labels, ks=(1, 5, 10)):
    """
    Leave-one-out retrieval:
    Với mỗi ảnh query, tìm top-k ảnh gần nhất trong cùng tập.
    Nếu trong top-k có ít nhất 1 ảnh cùng label thì đúng.
    """
    n = embeddings.shape[0]

    if n < 2:
        return {f"recall@{k}": 0.0 for k in ks}

    max_k = max(ks)

    if max_k >= n:
        max_k = n - 1

    similarity_matrix = embeddings @ embeddings.T

    # Không cho ảnh query match chính nó
    np.fill_diagonal(similarity_matrix, -np.inf)

    topk_indices = np.argpartition(
        -similarity_matrix,
        kth=max_k - 1,
        axis=1,
    )[:, :max_k]

    row_indices = np.arange(n)[:, None]
    topk_scores = similarity_matrix[row_indices, topk_indices]

    sorted_order = np.argsort(-topk_scores, axis=1)
    topk_indices = topk_indices[row_indices, sorted_order]

    results = {}

    for k in ks:
        actual_k = min(k, max_k)

        correct = 0

        for i in range(n):
            retrieved_indices = topk_indices[i, :actual_k]
            retrieved_labels = labels[retrieved_indices]

            if np.any(retrieved_labels == labels[i]):
                correct += 1

        results[f"recall@{k}"] = correct / n

    return results


def compute_similarity_stats(embeddings, labels, max_pairs=10000, seed=42):
    rng = np.random.default_rng(seed)

    label_to_indices = {}

    for idx, label in enumerate(labels):
        label = int(label)
        label_to_indices.setdefault(label, []).append(idx)

    valid_positive_labels = [
        label for label, indices in label_to_indices.items()
        if len(indices) >= 2
    ]

    all_labels = list(label_to_indices.keys())

    positive_sims = []
    negative_sims = []

    for _ in range(max_pairs):
        if len(valid_positive_labels) == 0:
            break

        label = int(rng.choice(valid_positive_labels))
        indices = label_to_indices[label]

        i, j = rng.choice(indices, size=2, replace=False)

        sim = float(np.dot(embeddings[i], embeddings[j]))
        positive_sims.append(sim)

    for _ in range(max_pairs):
        if len(all_labels) < 2:
            break

        label_a, label_b = rng.choice(all_labels, size=2, replace=False)

        i = rng.choice(label_to_indices[int(label_a)])
        j = rng.choice(label_to_indices[int(label_b)])

        sim = float(np.dot(embeddings[i], embeddings[j]))
        negative_sims.append(sim)

    stats = {
        "positive_mean": float(np.mean(positive_sims)) if positive_sims else 0.0,
        "positive_std": float(np.std(positive_sims)) if positive_sims else 0.0,
        "negative_mean": float(np.mean(negative_sims)) if negative_sims else 0.0,
        "negative_std": float(np.std(negative_sims)) if negative_sims else 0.0,
    }

    return stats


# ============================================================
# 5. SAVE OUTPUTS
# ============================================================

def save_outputs(
    output_dir: Path,
    data_dir: Path,
    model_name: str,
    embeddings: np.ndarray,
    records,
    failed_records,
    labels,
    label_to_idx,
    retrieval_metrics,
    similarity_stats,
    runtime_seconds,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "insightface_embeddings.npy"
    metadata_path = output_dir / "insightface_embedding_metadata.csv"
    failed_path = output_dir / "insightface_failed_images.csv"
    summary_path = output_dir / "insightface_retrieval_summary.json"

    np.save(embeddings_path, embeddings)

    metadata_df = pd.DataFrame(records)
    metadata_df["label_idx"] = labels

    metadata_df.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    failed_df = pd.DataFrame(failed_records)
    failed_df.to_csv(
        failed_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "baseline": "InsightFace pretrained recognition",
        "model_name": model_name,
        "data_dir": str(data_dir),
        "num_images_success": int(len(records)),
        "num_images_failed": int(len(failed_records)),
        "num_classes": int(len(label_to_idx)),
        "embedding_shape": list(embeddings.shape),
        "retrieval_metrics": retrieval_metrics,
        "similarity_stats": similarity_stats,
        "runtime_seconds": float(runtime_seconds),
        "outputs": {
            "embeddings_npy": str(embeddings_path),
            "metadata_csv": str(metadata_path),
            "failed_csv": str(failed_path),
            "summary_json": str(summary_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    return summary


# ============================================================
# 6. MAIN
# ============================================================

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy data_dir: {data_dir}")

    records = collect_images(data_dir)

    if args.max_images > 0:
        records = records[:args.max_images]

    if len(records) == 0:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong: {data_dir}")

    print("=" * 80)
    print("INSIGHTFACE PRETRAINED RETRIEVAL BASELINE")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Total images found: {len(records)}")
    print("=" * 80)

    rec_model = build_insightface_recognition_model(
        model_name=args.model_name,
        ctx_id=args.ctx_id,
    )

    embeddings, kept_records, failed_records, runtime_seconds = extract_embeddings(
        records=records,
        rec_model=rec_model,
    )

    labels, label_to_idx = encode_labels(kept_records)

    retrieval_metrics = compute_recall_at_k(
        embeddings=embeddings,
        labels=labels,
        ks=(1, 5, 10),
    )

    similarity_stats = compute_similarity_stats(
        embeddings=embeddings,
        labels=labels,
        max_pairs=10000,
    )

    summary = save_outputs(
        output_dir=output_dir,
        data_dir=data_dir,
        model_name=args.model_name,
        embeddings=embeddings,
        records=kept_records,
        failed_records=failed_records,
        labels=labels,
        label_to_idx=label_to_idx,
        retrieval_metrics=retrieval_metrics,
        similarity_stats=similarity_stats,
        runtime_seconds=runtime_seconds,
    )

    print()
    print("=" * 80)
    print("HOÀN TẤT INSIGHTFACE BASELINE")
    print("=" * 80)
    print(f"Images success: {summary['num_images_success']}")
    print(f"Images failed : {summary['num_images_failed']}")
    print(f"Classes       : {summary['num_classes']}")
    print()
    print("Retrieval metrics:")
    print(f"Recall@1 : {retrieval_metrics['recall@1']:.4f}")
    print(f"Recall@5 : {retrieval_metrics['recall@5']:.4f}")
    print(f"Recall@10: {retrieval_metrics['recall@10']:.4f}")
    print()
    print("Similarity stats:")
    print(f"Positive mean: {similarity_stats['positive_mean']:.4f}")
    print(f"Negative mean: {similarity_stats['negative_mean']:.4f}")
    print()
    print(f"Summary JSON: {summary['outputs']['summary_json']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_distances

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

from src.cluster_refinement.config import (
    PROJECT_ROOT,
    TEST_EVENT_VERSION,
    TEST_CLUSTER_NAME,
    TEST_CLUSTER_DIR,
    EMBEDDING_OUTPUT_DIR,
    FACENET_EMBEDDINGS_NPY_PATH,
    FACENET_EMBEDDING_METADATA_CSV_PATH,
    REPORT_OUTPUT_DIR,
    FACENET_EMBEDDING_SUMMARY_JSON_PATH,
    LOG_OUTPUT_DIR,
    FACENET_EMBEDDING_LOG_PATH,
    FACENET_PRETRAINED,
    FACENET_IMAGE_SIZE,
    FACENET_NORMALIZE_MEAN,
    FACENET_NORMALIZE_STD,
    SUPPORTED_IMAGE_EXTENSIONS,
)


# ============================================================
# 1. LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    """
    Tạo logger:
    - In ra terminal
    - Ghi toàn bộ vào file .log
    """
    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("facenet_embedding_extraction_one_cluster")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        FACENET_EMBEDDING_LOG_PATH,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# 2. PATH UTILITIES
# ============================================================

def to_project_relative_path(path: Path) -> str:
    """
    Chuyển absolute path về relative path tính từ project root.
    """
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


# ============================================================
# 3. COLLECT INPUT IMAGES
# ============================================================

def collect_cluster_images(cluster_dir: Path) -> list[Path]:
    """
    Lấy toàn bộ ảnh trong folder person_0.
    """
    image_paths: list[Path] = []

    for path in sorted(cluster_dir.iterdir()):
        if path.is_file() and path.suffix in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.append(path)

    return image_paths


# ============================================================
# 4. LOAD FACENET MODEL
# ============================================================

def load_facenet_model(
    device: torch.device,
    logger: logging.Logger,
) -> InceptionResnetV1:
    """
    Load đúng model FaceNet cũ:
    InceptionResnetV1(pretrained='vggface2')
    """
    logger.info("Đang load FaceNet InceptionResnetV1 pretrained=%s ...", FACENET_PRETRAINED)

    model = InceptionResnetV1(
        pretrained=FACENET_PRETRAINED
    ).eval().to(device)

    logger.info("[OK] FaceNet model đã sẵn sàng.")
    return model


# ============================================================
# 5. TRANSFORM GIỐNG CODE CLUSTERING CŨ
# ============================================================

def build_facenet_transform() -> transforms.Compose:
    """
    Transform giống code clustering cũ:
    - Resize 160x160
    - ToTensor
    - Normalize mean/std = 0.5
    """
    return transforms.Compose([
        transforms.Resize(FACENET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=FACENET_NORMALIZE_MEAN,
            std=FACENET_NORMALIZE_STD,
        ),
    ])


# ============================================================
# 6. EMBEDDING NORMALIZATION
# ============================================================

def l2_normalize_embedding(
    embedding: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Chuẩn hóa embedding:
        embedding = embedding / ||embedding||

    Trả về:
    - embedding đã normalize
    - norm trước normalize
    - norm sau normalize
    """
    embedding = embedding.astype(np.float32).flatten()

    raw_norm = float(np.linalg.norm(embedding))

    if raw_norm <= 0:
        raise ValueError("Embedding có L2 norm <= 0, không thể normalize.")

    normalized_embedding = embedding / raw_norm
    normalized_norm = float(np.linalg.norm(normalized_embedding))

    return normalized_embedding, raw_norm, normalized_norm


# ============================================================
# 7. DISTANCE STATS
# ============================================================

def compute_pairwise_cosine_distance_stats(
    embeddings: np.ndarray,
) -> dict[str, Any]:
    """
    Tính thống kê khoảng cách cosine nội cluster.

    Lưu ý:
    - Đường chéo distance(i, i) = 0
    - Khi thống kê nội cluster, ta loại đường chéo.
    """
    if embeddings.shape[0] < 2:
        return {
            "num_pairs": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
            "std": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }

    dist_matrix = cosine_distances(embeddings)

    upper_triangle_values = dist_matrix[
        np.triu_indices_from(dist_matrix, k=1)
    ]

    return {
        "num_pairs": int(len(upper_triangle_values)),
        "min": float(np.min(upper_triangle_values)),
        "mean": float(np.mean(upper_triangle_values)),
        "median": float(np.median(upper_triangle_values)),
        "max": float(np.max(upper_triangle_values)),
        "std": float(np.std(upper_triangle_values)),
        "p05": float(np.percentile(upper_triangle_values, 5)),
        "p25": float(np.percentile(upper_triangle_values, 25)),
        "p75": float(np.percentile(upper_triangle_values, 75)),
        "p95": float(np.percentile(upper_triangle_values, 95)),
    }


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def extract_facenet_embeddings_one_cluster() -> None:
    logger = setup_logging()
    total_start = time.perf_counter()

    logger.info("=" * 90)
    logger.info("BẮT ĐẦU EXTRACT FACENET EMBEDDINGS CHO 1 CLUSTER")
    logger.info("=" * 90)
    logger.info("Event version: %s", TEST_EVENT_VERSION)
    logger.info("Cluster name: %s", TEST_CLUSTER_NAME)
    logger.info("Cluster path: %s", TEST_CLUSTER_DIR)

    if not TEST_CLUSTER_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy folder cluster: {TEST_CLUSTER_DIR}"
        )

    # Tạo output dirs
    EMBEDDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_paths = collect_cluster_images(TEST_CLUSTER_DIR)

    logger.info("Tổng ảnh tìm thấy trong cluster: %d", len(image_paths))

    if len(image_paths) == 0:
        raise ValueError("Folder cluster không có ảnh hợp lệ.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Torch device: %s", device)

    if device.type == "cuda":
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA available: True")
    else:
        logger.warning("CUDA không khả dụng, script sẽ chạy CPU.")

    # Model + transform
    model = load_facenet_model(device, logger)
    transform = build_facenet_transform()

    # Extraction containers
    embeddings: list[np.ndarray] = []
    metadata_records: list[dict[str, Any]] = []

    success_count = 0
    error_count = 0

    inference_times_ms: list[float] = []
    raw_norms: list[float] = []
    normalized_norms: list[float] = []
    widths: list[int] = []
    heights: list[int] = []

    logger.info("Bắt đầu xử lý từng ảnh...")

    for image_index, image_path in enumerate(
        tqdm(image_paths, desc="Extracting FaceNet embeddings"),
        start=0,
    ):
        record: dict[str, Any] = {
            "embedding_index": "",
            "image_index_in_cluster": image_index,
            "image_name": image_path.name,
            "image_path": to_project_relative_path(image_path),
            "width": "",
            "height": "",
            "embedding_dim": "",
            "raw_embedding_l2_norm": "",
            "normalized_embedding_l2_norm": "",
            "inference_time_ms": "",
            "status": "pending",
            "error": "",
        }

        try:
            # Đọc ảnh giống code cũ
            img = Image.open(image_path).convert("RGB")

            width, height = img.size
            widths.append(int(width))
            heights.append(int(height))

            # Preprocess
            tensor = transform(img).unsqueeze(0).to(device)

            # Inference timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_inference = time.perf_counter()

            with torch.no_grad():
                emb_tensor = model(tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_inference = time.perf_counter()

            inference_time_ms = (end_inference - start_inference) * 1000.0

            # To numpy
            emb = emb_tensor.cpu().numpy()[0]

            # L2 normalize giống code clustering cũ
            normalized_emb, raw_norm, normalized_norm = l2_normalize_embedding(emb)

            embedding_index = len(embeddings)
            embeddings.append(normalized_emb)

            success_count += 1
            inference_times_ms.append(inference_time_ms)
            raw_norms.append(raw_norm)
            normalized_norms.append(normalized_norm)

            record.update({
                "embedding_index": embedding_index,
                "width": int(width),
                "height": int(height),
                "embedding_dim": int(normalized_emb.shape[0]),
                "raw_embedding_l2_norm": raw_norm,
                "normalized_embedding_l2_norm": normalized_norm,
                "inference_time_ms": inference_time_ms,
                "status": "success",
                "error": "",
            })

        except Exception as error:
            error_count += 1

            record.update({
                "status": "error",
                "error": str(error),
            })

            logger.error(
                "Lỗi xử lý ảnh %s | %s",
                image_path.name,
                str(error),
            )

        metadata_records.append(record)

    if len(embeddings) == 0:
        raise RuntimeError("Không extract thành công embedding nào.")

    # ============================================================
    # 9. SAVE EMBEDDINGS + METADATA
    # ============================================================

    embeddings_array = np.asarray(embeddings, dtype=np.float32)

    np.save(FACENET_EMBEDDINGS_NPY_PATH, embeddings_array)

    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(
        FACENET_EMBEDDING_METADATA_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 10. STATS
    # ============================================================

    distance_stats = compute_pairwise_cosine_distance_stats(embeddings_array)

    total_runtime_seconds = time.perf_counter() - total_start

    summary = {
        "test_scope": "single_cluster_embedding_extraction",
        "event_version": TEST_EVENT_VERSION,
        "cluster_name": TEST_CLUSTER_NAME,
        "cluster_dir": str(TEST_CLUSTER_DIR),

        "embedding_model": {
            "model": "facenet_pytorch.InceptionResnetV1",
            "pretrained": FACENET_PRETRAINED,
            "input_size": list(FACENET_IMAGE_SIZE),
            "normalization_mean": FACENET_NORMALIZE_MEAN,
            "normalization_std": FACENET_NORMALIZE_STD,
            "embedding_postprocess": "L2 normalization",
        },

        "device": {
            "torch_device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },

        "input_images": {
            "num_images_found": len(image_paths),
            "num_images_success": success_count,
            "num_images_error": error_count,
            "width_stats": {
                "min": min(widths) if widths else None,
                "mean": mean(widths) if widths else None,
                "max": max(widths) if widths else None,
            },
            "height_stats": {
                "min": min(heights) if heights else None,
                "mean": mean(heights) if heights else None,
                "max": max(heights) if heights else None,
            },
        },

        "embeddings": {
            "shape": list(embeddings_array.shape),
            "num_embeddings": int(embeddings_array.shape[0]),
            "embedding_dim": int(embeddings_array.shape[1]),
            "raw_l2_norm_stats": {
                "min": min(raw_norms) if raw_norms else None,
                "mean": mean(raw_norms) if raw_norms else None,
                "max": max(raw_norms) if raw_norms else None,
            },
            "normalized_l2_norm_stats": {
                "min": min(normalized_norms) if normalized_norms else None,
                "mean": mean(normalized_norms) if normalized_norms else None,
                "max": max(normalized_norms) if normalized_norms else None,
            },
        },

        "performance": {
            "total_runtime_seconds": total_runtime_seconds,
            "inference_time_ms": {
                "min": min(inference_times_ms) if inference_times_ms else None,
                "mean": mean(inference_times_ms) if inference_times_ms else None,
                "median": median(inference_times_ms) if inference_times_ms else None,
                "max": max(inference_times_ms) if inference_times_ms else None,
            },
        },

        "pairwise_cosine_distance_in_cluster": distance_stats,

        "outputs": {
            "embeddings_npy": str(FACENET_EMBEDDINGS_NPY_PATH),
            "embedding_metadata_csv": str(FACENET_EMBEDDING_METADATA_CSV_PATH),
            "summary_json": str(FACENET_EMBEDDING_SUMMARY_JSON_PATH),
            "log_file": str(FACENET_EMBEDDING_LOG_PATH),
        },
    }

    with open(
        FACENET_EMBEDDING_SUMMARY_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    # ============================================================
    # 11. FINAL LOG
    # ============================================================

    logger.info("=" * 90)
    logger.info("HOÀN TẤT EXTRACT FACENET EMBEDDINGS CHO %s", TEST_CLUSTER_NAME)
    logger.info("=" * 90)
    logger.info("Tổng ảnh tìm thấy: %d", len(image_paths))
    logger.info("Thành công: %d", success_count)
    logger.info("Lỗi: %d", error_count)
    logger.info("Embedding shape: %s", embeddings_array.shape)
    logger.info("Tổng thời gian chạy: %.4f giây", total_runtime_seconds)

    if inference_times_ms:
        logger.info(
            "Inference time (ms) | min=%.4f | mean=%.4f | median=%.4f | max=%.4f",
            min(inference_times_ms),
            mean(inference_times_ms),
            median(inference_times_ms),
            max(inference_times_ms),
        )

    if raw_norms:
        logger.info(
            "Raw embedding L2 norm | min=%.6f | mean=%.6f | max=%.6f",
            min(raw_norms),
            mean(raw_norms),
            max(raw_norms),
        )

    if normalized_norms:
        logger.info(
            "Normalized embedding L2 norm | min=%.6f | mean=%.6f | max=%.6f",
            min(normalized_norms),
            mean(normalized_norms),
            max(normalized_norms),
        )

    logger.info(
        "Pairwise cosine distance | pairs=%s | min=%s | mean=%s | median=%s | max=%s | p95=%s",
        distance_stats["num_pairs"],
        distance_stats["min"],
        distance_stats["mean"],
        distance_stats["median"],
        distance_stats["max"],
        distance_stats["p95"],
    )

    logger.info("Đã lưu embeddings: %s", FACENET_EMBEDDINGS_NPY_PATH)
    logger.info("Đã lưu metadata: %s", FACENET_EMBEDDING_METADATA_CSV_PATH)
    logger.info("Đã lưu summary: %s", FACENET_EMBEDDING_SUMMARY_JSON_PATH)
    logger.info("Đã lưu log: %s", FACENET_EMBEDDING_LOG_PATH)
    logger.info("=" * 90)


if __name__ == "__main__":
    extract_facenet_embeddings_one_cluster()
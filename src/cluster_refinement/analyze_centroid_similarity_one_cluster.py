from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean, median

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cluster_refinement.config import (
    PROJECT_ROOT,
    TEST_EVENT_VERSION,
    TEST_CLUSTER_NAME,
    REPORT_OUTPUT_DIR,
    LOG_OUTPUT_DIR,
    FACENET_EMBEDDINGS_NPY_PATH,
    FACENET_EMBEDDING_METADATA_CSV_PATH,
)


# ============================================================
# 1. OUTPUT PATHS
# ============================================================

CENTROID_ANALYSIS_DIR = REPORT_OUTPUT_DIR / "centroid_analysis"
CENTROID_ANALYSIS_CSV_PATH = (
    CENTROID_ANALYSIS_DIR / f"{TEST_CLUSTER_NAME}_centroid_similarity_analysis.csv"
)
CENTROID_SUMMARY_JSON_PATH = (
    CENTROID_ANALYSIS_DIR / f"{TEST_CLUSTER_NAME}_centroid_similarity_summary.json"
)

VISUALIZATION_DIR = CENTROID_ANALYSIS_DIR / "visualizations"

SIMILARITY_HISTOGRAM_PATH = (
    VISUALIZATION_DIR / f"{TEST_CLUSTER_NAME}_centroid_similarity_histogram.png"
)

TOP_CLOSEST_MONTAGE_PATH = (
    VISUALIZATION_DIR / f"{TEST_CLUSTER_NAME}_top_closest_to_centroid.jpg"
)

TOP_FARTHEST_MONTAGE_PATH = (
    VISUALIZATION_DIR / f"{TEST_CLUSTER_NAME}_top_farthest_from_centroid.jpg"
)

CENTROID_ANALYSIS_LOG_PATH = (
    LOG_OUTPUT_DIR / "centroid_similarity_analysis.log"
)


# ============================================================
# 2. CONFIG FOR REVIEW
# ============================================================

TOP_K_REVIEW = 30

MONTAGE_THUMB_SIZE = (160, 160)
MONTAGE_COLUMNS = 6


# ============================================================
# 3. LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("centroid_similarity_analysis_one_cluster")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        CENTROID_ANALYSIS_LOG_PATH,
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
# 4. UTILITIES
# ============================================================

def resolve_project_path(path_value: str) -> Path:
    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm <= 0:
        raise ValueError("Vector có norm <= 0, không thể normalize.")

    return vector / norm


def compute_centroid_similarity(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input embeddings đã L2-normalized.
    Tính centroid, normalize centroid, rồi tính cosine similarity.

    Vì cả embedding và centroid đều normalized:
        cosine_similarity = dot_product
    """
    centroid_raw = embeddings.mean(axis=0)
    centroid = l2_normalize(centroid_raw.astype(np.float32))

    cosine_similarities = embeddings @ centroid
    cosine_distances = 1.0 - cosine_similarities

    return centroid, cosine_similarities, cosine_distances


# ============================================================
# 5. MONTAGE CREATION
# ============================================================

def create_face_montage(
    rows: pd.DataFrame,
    output_path: Path,
    score_column: str,
    score_title: str,
) -> None:
    """
    Tạo montage nhiều ảnh mặt.
    Mỗi ô hiển thị ảnh + giá trị score.
    """

    thumb_w, thumb_h = MONTAGE_THUMB_SIZE
    cols = MONTAGE_COLUMNS
    rows_count = int(np.ceil(len(rows) / cols))

    label_height = 34

    canvas_h = rows_count * (thumb_h + label_height)
    canvas_w = cols * thumb_w

    canvas = np.full(
        (canvas_h, canvas_w, 3),
        255,
        dtype=np.uint8,
    )

    for idx, (_, row) in enumerate(rows.iterrows()):
        grid_row = idx // cols
        grid_col = idx % cols

        x = grid_col * thumb_w
        y = grid_row * (thumb_h + label_height)

        image_path = resolve_project_path(row["image_path"])
        image = cv2.imread(str(image_path))

        if image is None:
            thumb = np.full((thumb_h, thumb_w, 3), 200, dtype=np.uint8)
        else:
            thumb = cv2.resize(
                image,
                (thumb_w, thumb_h),
                interpolation=cv2.INTER_AREA,
            )

        canvas[y:y + thumb_h, x:x + thumb_w] = thumb

        score = float(row[score_column])
        image_name = str(row["image_name"])

        short_name = image_name[:11]

        cv2.putText(
            canvas,
            f"{score_title}: {score:.3f}",
            (x + 5, y + thumb_h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            short_name,
            (x + 5, y + thumb_h + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), canvas)

    if not success:
        raise IOError(f"Không thể ghi montage: {output_path}")


# ============================================================
# 6. HISTOGRAM
# ============================================================

def save_similarity_histogram(
    similarities: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=30)
    plt.xlabel("Cosine similarity to cluster centroid")
    plt.ylabel("Number of faces")
    plt.title(f"Centroid similarity distribution - {TEST_CLUSTER_NAME}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# 7. MAIN ANALYSIS
# ============================================================

def analyze_centroid_similarity_one_cluster() -> None:
    logger = setup_logging()

    logger.info("=" * 90)
    logger.info("BẮT ĐẦU PHÂN TÍCH CENTROID SIMILARITY CHO %s", TEST_CLUSTER_NAME)
    logger.info("=" * 90)

    if not FACENET_EMBEDDINGS_NPY_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy embeddings npy: {FACENET_EMBEDDINGS_NPY_PATH}"
        )

    if not FACENET_EMBEDDING_METADATA_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy embedding metadata CSV: {FACENET_EMBEDDING_METADATA_CSV_PATH}"
        )

    CENTROID_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(FACENET_EMBEDDINGS_NPY_PATH)
    metadata_df = pd.read_csv(
        FACENET_EMBEDDING_METADATA_CSV_PATH,
        keep_default_na=False,
    )

    success_df = metadata_df[
        metadata_df["status"] == "success"
    ].copy()

    if len(success_df) != embeddings.shape[0]:
        raise ValueError(
            "Số dòng metadata success không khớp với số embedding. "
            f"metadata success={len(success_df)}, embeddings={embeddings.shape[0]}"
        )

    logger.info("Embedding shape: %s", embeddings.shape)
    logger.info("Metadata success rows: %d", len(success_df))

    centroid, similarities, distances = compute_centroid_similarity(embeddings)

    success_df["cosine_similarity_to_centroid"] = similarities
    success_df["cosine_distance_to_centroid"] = distances

    # Rank: 1 = gần centroid nhất
    success_df = success_df.sort_values(
        by="cosine_similarity_to_centroid",
        ascending=False,
    ).reset_index(drop=True)

    success_df["centroid_similarity_rank_desc"] = np.arange(
        1,
        len(success_df) + 1,
    )

    success_df.to_csv(
        CENTROID_ANALYSIS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    similarity_stats = {
        "min": float(np.min(similarities)),
        "mean": float(np.mean(similarities)),
        "median": float(np.median(similarities)),
        "max": float(np.max(similarities)),
        "std": float(np.std(similarities)),
        "p05": float(np.percentile(similarities, 5)),
        "p25": float(np.percentile(similarities, 25)),
        "p75": float(np.percentile(similarities, 75)),
        "p95": float(np.percentile(similarities, 95)),
    }

    distance_stats = {
        "min": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
        "std": float(np.std(distances)),
        "p05": float(np.percentile(distances, 5)),
        "p25": float(np.percentile(distances, 25)),
        "p75": float(np.percentile(distances, 75)),
        "p95": float(np.percentile(distances, 95)),
    }

    top_closest = success_df.head(TOP_K_REVIEW).copy()
    top_farthest = success_df.tail(TOP_K_REVIEW).sort_values(
        by="cosine_similarity_to_centroid",
        ascending=True,
    ).copy()

    create_face_montage(
        rows=top_closest,
        output_path=TOP_CLOSEST_MONTAGE_PATH,
        score_column="cosine_similarity_to_centroid",
        score_title="sim",
    )

    create_face_montage(
        rows=top_farthest,
        output_path=TOP_FARTHEST_MONTAGE_PATH,
        score_column="cosine_similarity_to_centroid",
        score_title="sim",
    )

    save_similarity_histogram(
        similarities=similarities,
        output_path=SIMILARITY_HISTOGRAM_PATH,
    )

    summary = {
        "test_scope": "single_cluster_centroid_similarity_analysis",
        "event_version": TEST_EVENT_VERSION,
        "cluster_name": TEST_CLUSTER_NAME,
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "centroid_l2_norm": float(np.linalg.norm(centroid)),
        "cosine_similarity_to_centroid_stats": similarity_stats,
        "cosine_distance_to_centroid_stats": distance_stats,
        "review_config": {
            "top_k_closest": TOP_K_REVIEW,
            "top_k_farthest": TOP_K_REVIEW,
        },
        "outputs": {
            "analysis_csv": str(CENTROID_ANALYSIS_CSV_PATH),
            "summary_json": str(CENTROID_SUMMARY_JSON_PATH),
            "histogram": str(SIMILARITY_HISTOGRAM_PATH),
            "top_closest_montage": str(TOP_CLOSEST_MONTAGE_PATH),
            "top_farthest_montage": str(TOP_FARTHEST_MONTAGE_PATH),
            "log_file": str(CENTROID_ANALYSIS_LOG_PATH),
        },
    }

    with open(CENTROID_SUMMARY_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    logger.info("Cosine similarity to centroid:")
    logger.info(
        "min=%.6f | mean=%.6f | median=%.6f | max=%.6f | std=%.6f",
        similarity_stats["min"],
        similarity_stats["mean"],
        similarity_stats["median"],
        similarity_stats["max"],
        similarity_stats["std"],
    )
    logger.info(
        "p05=%.6f | p25=%.6f | p75=%.6f | p95=%.6f",
        similarity_stats["p05"],
        similarity_stats["p25"],
        similarity_stats["p75"],
        similarity_stats["p95"],
    )

    logger.info("Cosine distance to centroid:")
    logger.info(
        "min=%.6f | mean=%.6f | median=%.6f | max=%.6f",
        distance_stats["min"],
        distance_stats["mean"],
        distance_stats["median"],
        distance_stats["max"],
    )

    logger.info("Đã lưu analysis CSV: %s", CENTROID_ANALYSIS_CSV_PATH)
    logger.info("Đã lưu summary JSON: %s", CENTROID_SUMMARY_JSON_PATH)
    logger.info("Đã lưu histogram: %s", SIMILARITY_HISTOGRAM_PATH)
    logger.info("Đã lưu top closest montage: %s", TOP_CLOSEST_MONTAGE_PATH)
    logger.info("Đã lưu top farthest montage: %s", TOP_FARTHEST_MONTAGE_PATH)
    logger.info("=" * 90)
    logger.info("HOÀN TẤT PHÂN TÍCH CENTROID SIMILARITY")
    logger.info("=" * 90)


if __name__ == "__main__":
    analyze_centroid_similarity_one_cluster()
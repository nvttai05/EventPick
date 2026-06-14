from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

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
# 1. CONFIG
# ============================================================

LINKAGES = ["complete", "average"]

DISTANCE_THRESHOLDS = [
    0.12,
    0.16,
    0.20,
    0.24,
    0.28,
    0.32,
]

TOP_N_CLUSTERS_TO_VISUALIZE = 8
MAX_IMAGES_PER_MONTAGE = 30
MONTAGE_COLUMNS = 6
THUMB_SIZE = (160, 160)


# ============================================================
# 2. OUTPUT PATHS
# ============================================================

SUBCLUSTERING_ROOT_DIR = (
    REPORT_OUTPUT_DIR
    / "subclustering_agglomerative_sweep"
)

SWEEP_SUMMARY_CSV_PATH = (
    SUBCLUSTERING_ROOT_DIR
    / f"{TEST_CLUSTER_NAME}_agglomerative_sweep_summary.csv"
)

SWEEP_SUMMARY_JSON_PATH = (
    SUBCLUSTERING_ROOT_DIR
    / f"{TEST_CLUSTER_NAME}_agglomerative_sweep_summary.json"
)

SUBCLUSTERING_LOG_PATH = (
    LOG_OUTPUT_DIR
    / "agglomerative_subclustering_sweep.log"
)


# ============================================================
# 3. LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("agglomerative_subclustering_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        SUBCLUSTERING_LOG_PATH,
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
# 4. PATH HELPERS
# ============================================================

def resolve_project_path(path_value: str) -> Path:
    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def threshold_to_folder_name(threshold: float) -> str:
    return str(threshold).replace(".", "p")


# ============================================================
# 5. CLUSTERING
# ============================================================

def run_agglomerative(
    embeddings: np.ndarray,
    linkage: str,
    distance_threshold: float,
) -> np.ndarray:
    """
    Chạy Agglomerative clustering trên embedding đã L2-normalized.
    Dùng cosine distance.
    """

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage=linkage,
        compute_full_tree=True,
    )

    labels = model.fit_predict(embeddings)
    return labels


# ============================================================
# 6. METRICS
# ============================================================

def safe_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    """
    Tính silhouette nếu:
    - Có ít nhất 2 cluster
    - Không phải mỗi điểm là một cluster
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return None

    if len(unique_labels) >= len(labels):
        return None

    try:
        return float(
            silhouette_score(
                embeddings,
                labels,
                metric="cosine",
            )
        )
    except Exception:
        return None


def summarize_labels(labels: np.ndarray) -> dict:
    counter = Counter(labels.tolist())

    sizes_desc = sorted(
        counter.values(),
        reverse=True,
    )

    return {
        "num_subclusters": int(len(counter)),
        "largest_cluster_size": int(sizes_desc[0]) if sizes_desc else 0,
        "smallest_cluster_size": int(sizes_desc[-1]) if sizes_desc else 0,
        "cluster_sizes_desc": [int(x) for x in sizes_desc],
    }


# ============================================================
# 7. MONTAGE
# ============================================================

def create_cluster_montage(
    cluster_df: pd.DataFrame,
    output_path: Path,
    cluster_label: int,
) -> None:
    """
    Tạo montage ảnh của một subcluster.
    """
    cluster_df = cluster_df.head(MAX_IMAGES_PER_MONTAGE)

    thumb_w, thumb_h = THUMB_SIZE
    cols = MONTAGE_COLUMNS
    rows = int(np.ceil(len(cluster_df) / cols))

    label_height = 34

    canvas = np.full(
        (
            rows * (thumb_h + label_height),
            cols * thumb_w,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    for idx, (_, row) in enumerate(cluster_df.iterrows()):
        r = idx // cols
        c = idx % cols

        x = c * thumb_w
        y = r * (thumb_h + label_height)

        image_path = resolve_project_path(row["image_path"])
        image = cv2.imread(str(image_path))

        if image is None:
            thumb = np.full(
                (thumb_h, thumb_w, 3),
                200,
                dtype=np.uint8,
            )
        else:
            thumb = cv2.resize(
                image,
                THUMB_SIZE,
                interpolation=cv2.INTER_AREA,
            )

        canvas[y:y + thumb_h, x:x + thumb_w] = thumb

        image_name = str(row["image_name"])
        short_name = image_name[:11]

        cv2.putText(
            canvas,
            f"subcluster: {cluster_label}",
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
# 8. SAVE ONE CONFIG RESULT
# ============================================================

def save_one_config_result(
    metadata_df: pd.DataFrame,
    labels: np.ndarray,
    linkage: str,
    threshold: float,
    embeddings: np.ndarray,
) -> dict:
    """
    Lưu assignments + summary + montage cho một cấu hình.
    """

    threshold_name = threshold_to_folder_name(threshold)

    config_dir = (
        SUBCLUSTERING_ROOT_DIR
        / f"{linkage}_thr_{threshold_name}"
    )

    montages_dir = config_dir / "montages"

    config_dir.mkdir(parents=True, exist_ok=True)
    montages_dir.mkdir(parents=True, exist_ok=True)

    result_df = metadata_df.copy()
    result_df["subcluster_label"] = labels

    label_counts = (
        result_df["subcluster_label"]
        .value_counts()
        .sort_values(ascending=False)
    )

    # Ghi assignments
    assignments_csv_path = (
        config_dir
        / f"{TEST_CLUSTER_NAME}_{linkage}_thr_{threshold_name}_assignments.csv"
    )

    result_df.to_csv(
        assignments_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Tạo montage cho các cluster lớn nhất
    top_labels = label_counts.head(TOP_N_CLUSTERS_TO_VISUALIZE).index.tolist()

    montage_paths = []

    for rank, label in enumerate(top_labels, start=1):
        cluster_df = result_df[
            result_df["subcluster_label"] == label
        ].copy()

        montage_path = (
            montages_dir
            / f"rank_{rank:02d}_subcluster_{int(label):03d}_size_{len(cluster_df):04d}.jpg"
        )

        create_cluster_montage(
            cluster_df=cluster_df,
            output_path=montage_path,
            cluster_label=int(label),
        )

        montage_paths.append(str(montage_path))

    label_summary = summarize_labels(labels)
    silhouette = safe_silhouette_score(embeddings, labels)

    summary = {
        "event_version": TEST_EVENT_VERSION,
        "cluster_name": TEST_CLUSTER_NAME,
        "method": "AgglomerativeClustering",
        "metric": "cosine",
        "linkage": linkage,
        "distance_threshold": threshold,
        "num_faces": int(len(labels)),
        "num_subclusters": label_summary["num_subclusters"],
        "largest_cluster_size": label_summary["largest_cluster_size"],
        "smallest_cluster_size": label_summary["smallest_cluster_size"],
        "cluster_sizes_desc": label_summary["cluster_sizes_desc"],
        "silhouette_score_cosine": silhouette,
        "outputs": {
            "assignments_csv": str(assignments_csv_path),
            "montages_dir": str(montages_dir),
            "montage_files": montage_paths,
        },
    }

    summary_json_path = (
        config_dir
        / f"{TEST_CLUSTER_NAME}_{linkage}_thr_{threshold_name}_summary.json"
    )

    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    summary["outputs"]["summary_json"] = str(summary_json_path)

    return summary


# ============================================================
# 9. MAIN
# ============================================================

def test_agglomerative_subclustering_one_cluster() -> None:
    logger = setup_logging()

    logger.info("=" * 100)
    logger.info("BẮT ĐẦU AGGLOMERATIVE SUB-CLUSTERING SWEEP CHO %s", TEST_CLUSTER_NAME)
    logger.info("=" * 100)

    if not FACENET_EMBEDDINGS_NPY_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy embeddings: {FACENET_EMBEDDINGS_NPY_PATH}"
        )

    if not FACENET_EMBEDDING_METADATA_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy metadata: {FACENET_EMBEDDING_METADATA_CSV_PATH}"
        )

    SUBCLUSTERING_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(FACENET_EMBEDDINGS_NPY_PATH)

    metadata_df = pd.read_csv(
        FACENET_EMBEDDING_METADATA_CSV_PATH,
        keep_default_na=False,
    )

    metadata_df = metadata_df[
        metadata_df["status"] == "success"
    ].copy().reset_index(drop=True)

    if len(metadata_df) != embeddings.shape[0]:
        raise ValueError(
            "Số dòng metadata success không khớp embeddings. "
            f"metadata={len(metadata_df)}, embeddings={embeddings.shape[0]}"
        )

    logger.info("Cluster: %s", TEST_CLUSTER_NAME)
    logger.info("Embedding shape: %s", embeddings.shape)
    logger.info("Số cấu hình cần test: %d", len(LINKAGES) * len(DISTANCE_THRESHOLDS))

    all_summaries: list[dict] = []
    summary_rows: list[dict] = []

    for linkage in LINKAGES:
        for threshold in DISTANCE_THRESHOLDS:
            logger.info("-" * 100)
            logger.info(
                "Đang chạy | linkage=%s | distance_threshold=%.2f",
                linkage,
                threshold,
            )

            labels = run_agglomerative(
                embeddings=embeddings,
                linkage=linkage,
                distance_threshold=threshold,
            )

            summary = save_one_config_result(
                metadata_df=metadata_df,
                labels=labels,
                linkage=linkage,
                threshold=threshold,
                embeddings=embeddings,
            )

            all_summaries.append(summary)

            row = {
                "linkage": linkage,
                "distance_threshold": threshold,
                "num_faces": summary["num_faces"],
                "num_subclusters": summary["num_subclusters"],
                "largest_cluster_size": summary["largest_cluster_size"],
                "smallest_cluster_size": summary["smallest_cluster_size"],
                "silhouette_score_cosine": summary["silhouette_score_cosine"],
                "cluster_sizes_desc": str(summary["cluster_sizes_desc"]),
                "summary_json": summary["outputs"]["summary_json"],
                "montages_dir": summary["outputs"]["montages_dir"],
            }

            summary_rows.append(row)

            logger.info(
                "Kết quả | subclusters=%d | largest=%d | smallest=%d | silhouette=%s",
                summary["num_subclusters"],
                summary["largest_cluster_size"],
                summary["smallest_cluster_size"],
                str(summary["silhouette_score_cosine"]),
            )

    sweep_df = pd.DataFrame(summary_rows)

    sweep_df.to_csv(
        SWEEP_SUMMARY_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with open(SWEEP_SUMMARY_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(all_summaries, file, ensure_ascii=False, indent=4)

    logger.info("=" * 100)
    logger.info("HOÀN TẤT AGGLOMERATIVE SUB-CLUSTERING SWEEP")
    logger.info("=" * 100)
    logger.info("Đã lưu sweep summary CSV: %s", SWEEP_SUMMARY_CSV_PATH)
    logger.info("Đã lưu sweep summary JSON: %s", SWEEP_SUMMARY_JSON_PATH)
    logger.info("Đã lưu log: %s", SUBCLUSTERING_LOG_PATH)
    logger.info("=" * 100)


if __name__ == "__main__":
    test_agglomerative_subclustering_one_cluster()
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"/")

FACES_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "faces_metadata.csv"
IMAGES_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "images_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "visualizations" / "face_statistics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILS
# ============================================================

def save_fig(filename):
    output_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Đã lưu: {output_path}")


def add_value_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)


# ============================================================
# MAIN
# ============================================================

def main():
    if not FACES_METADATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy: {FACES_METADATA_PATH}")

    if not IMAGES_METADATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy: {IMAGES_METADATA_PATH}")

    faces_df = pd.read_csv(FACES_METADATA_PATH)
    images_df = pd.read_csv(IMAGES_METADATA_PATH)

    print("=" * 80)
    print("BẮT ĐẦU TẠO VISUALIZATION FACE STATISTICS")
    print("=" * 80)
    print(f"Faces metadata : {FACES_METADATA_PATH}")
    print(f"Images metadata: {IMAGES_METADATA_PATH}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print("=" * 80)

    # ========================================================
    # 1. Detection status ở mức ảnh
    # ========================================================
    if "detection_status" in images_df.columns:
        counts = images_df["detection_status"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="bar", ax=ax)

        ax.set_title("Phân bố trạng thái phát hiện khuôn mặt ở mức ảnh")
        ax.set_xlabel("Detection status")
        ax.set_ylabel("Số lượng ảnh")
        ax.tick_params(axis="x", rotation=0)
        add_value_labels(ax)

        save_fig("01_image_detection_status.png")

    # ========================================================
    # 2. Phân bố số mặt trên mỗi ảnh
    # ========================================================
    if "face_count" in images_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))

        images_df["face_count"].plot(
            kind="hist",
            bins=50,
            ax=ax
        )

        ax.set_title("Phân bố số khuôn mặt trên mỗi ảnh")
        ax.set_xlabel("Số khuôn mặt trong một ảnh")
        ax.set_ylabel("Số lượng ảnh")

        save_fig("02_face_count_distribution.png")

    # ========================================================
    # 3. Top ảnh đông người nhất
    # ========================================================
    if "face_count" in images_df.columns:
        top_images = images_df.sort_values(
            by="face_count",
            ascending=False
        ).head(10)

        if "image_id" in top_images.columns:
            labels = top_images["image_id"].astype(str)
        else:
            labels = [f"image_{i}" for i in range(len(top_images))]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(labels, top_images["face_count"])

        ax.set_title("Top 10 ảnh có nhiều khuôn mặt nhất")
        ax.set_xlabel("Image ID")
        ax.set_ylabel("Số khuôn mặt")
        ax.tick_params(axis="x", rotation=45)
        add_value_labels(ax)

        save_fig("03_top_crowded_images.png")

    # ========================================================
    # 4. Detection status ở mức face
    # ========================================================
    if "detection_status" in faces_df.columns:
        counts = faces_df["detection_status"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="bar", ax=ax)

        ax.set_title("Phân bố trạng thái detection ở mức khuôn mặt")
        ax.set_xlabel("Detection status")
        ax.set_ylabel("Số lượng khuôn mặt")
        ax.tick_params(axis="x", rotation=0)
        add_value_labels(ax)

        save_fig("04_face_detection_status.png")

    # ========================================================
    # 5. Confidence histogram
    # ========================================================
    if "confidence" in faces_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))

        faces_df["confidence"].dropna().plot(
            kind="hist",
            bins=40,
            ax=ax
        )

        ax.set_title("Phân bố confidence score của detector")
        ax.set_xlabel("Confidence score")
        ax.set_ylabel("Số lượng khuôn mặt")

        save_fig("05_confidence_distribution.png")

    # ========================================================
    # 6. Crop status
    # ========================================================
    if "crop_status" in faces_df.columns:
        counts = faces_df["crop_status"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="bar", ax=ax)

        ax.set_title("Phân bố trạng thái crop khuôn mặt")
        ax.set_xlabel("Crop status")
        ax.set_ylabel("Số lượng khuôn mặt")
        ax.tick_params(axis="x", rotation=0)
        add_value_labels(ax)

        save_fig("06_crop_status.png")

    # ========================================================
    # 7. Alignment status
    # ========================================================
    if "alignment_status" in faces_df.columns:
        counts = faces_df["alignment_status"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="bar", ax=ax)

        ax.set_title("Phân bố trạng thái alignment khuôn mặt")
        ax.set_xlabel("Alignment status")
        ax.set_ylabel("Số lượng khuôn mặt")
        ax.tick_params(axis="x", rotation=0)
        add_value_labels(ax)

        save_fig("07_alignment_status.png")

    # ========================================================
    # 8. Số face theo event
    # ========================================================
    if "event_id" in faces_df.columns:
        event_counts = faces_df["event_id"].value_counts().head(15)

        fig, ax = plt.subplots(figsize=(12, 6))
        event_counts.plot(kind="bar", ax=ax)

        ax.set_title("Top 15 event có nhiều khuôn mặt nhất")
        ax.set_xlabel("Event")
        ax.set_ylabel("Số lượng khuôn mặt")
        ax.tick_params(axis="x", rotation=45)
        add_value_labels(ax)

        save_fig("08_faces_by_event_top15.png")

    # ========================================================
    # 9. Tổng quan crop/align success
    # ========================================================
    summary_items = {}

    summary_items["Detected faces"] = len(faces_df)

    if "crop_status" in faces_df.columns:
        summary_items["Crop success"] = int((faces_df["crop_status"] == "success").sum())

    if "alignment_status" in faces_df.columns:
        summary_items["Align success"] = int((faces_df["alignment_status"] == "success").sum())

    if len(summary_items) > 1:
        fig, ax = plt.subplots(figsize=(9, 5))

        labels = list(summary_items.keys())
        values = list(summary_items.values())

        ax.bar(labels, values)
        ax.set_title("Tổng quan số lượng face detect / crop / align")
        ax.set_xlabel("Loại thống kê")
        ax.set_ylabel("Số lượng khuôn mặt")
        ax.tick_params(axis="x", rotation=0)
        add_value_labels(ax)

        save_fig("09_detect_crop_align_summary.png")

    print("=" * 80)
    print("HOÀN TẤT VISUALIZATION")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
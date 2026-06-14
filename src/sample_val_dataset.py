from pathlib import Path
import random
import shutil
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

SOURCE_VAL_DIR = Path(
    r"D:\Hoctap\CK_KHDL\EventPick_v1\data\val_processed"
)

OUTPUT_VAL_DIR = Path(
    r"D:\Hoctap\CK_KHDL\EventPick_v1\data\val_processed_sampled"
)

MIN_IMAGES_PER_CLASS = 5
MAX_IMAGES_PER_CLASS = 15

RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# MAIN
# ============================================================

def collect_class_folders(root_dir: Path):
    """
    Dataset structure:
        root/event/person/images
    """
    class_folders = []

    for event_dir in sorted(root_dir.iterdir()):
        if not event_dir.is_dir():
            continue

        for person_dir in sorted(event_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            class_folders.append((event_dir.name, person_dir.name, person_dir))

    return class_folders


def get_images(folder: Path):
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def main():
    random.seed(RANDOM_SEED)

    if not SOURCE_VAL_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy SOURCE_VAL_DIR: {SOURCE_VAL_DIR}")

    if OUTPUT_VAL_DIR.exists():
        print(f"Output đã tồn tại, đang xóa: {OUTPUT_VAL_DIR}")
        shutil.rmtree(OUTPUT_VAL_DIR)

    OUTPUT_VAL_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = collect_class_folders(SOURCE_VAL_DIR)

    total_input_images = 0
    total_output_images = 0
    kept_classes = 0
    skipped_classes = 0

    class_stats = []

    print("=" * 80)
    print("BẮT ĐẦU SAMPLE VALIDATION DATASET")
    print("=" * 80)
    print(f"Source: {SOURCE_VAL_DIR}")
    print(f"Output: {OUTPUT_VAL_DIR}")
    print(f"Min images/class: {MIN_IMAGES_PER_CLASS}")
    print(f"Max images/class: {MAX_IMAGES_PER_CLASS}")
    print("=" * 80)

    for event_name, person_name, person_dir in class_folders:
        images = get_images(person_dir)
        total_input_images += len(images)

        if len(images) < MIN_IMAGES_PER_CLASS:
            skipped_classes += 1
            continue

        selected_images = images.copy()

        if len(selected_images) > MAX_IMAGES_PER_CLASS:
            selected_images = random.sample(selected_images, MAX_IMAGES_PER_CLASS)

        selected_images = sorted(selected_images)

        output_person_dir = OUTPUT_VAL_DIR / event_name / person_name
        output_person_dir.mkdir(parents=True, exist_ok=True)

        for image_path in selected_images:
            output_path = output_person_dir / image_path.name
            shutil.copy2(image_path, output_path)

        kept_classes += 1
        total_output_images += len(selected_images)

        class_stats.append({
            "event": event_name,
            "person": person_name,
            "input_images": len(images),
            "output_images": len(selected_images),
        })

    print()
    print("=" * 80)
    print("HOÀN TẤT SAMPLE VALIDATION DATASET")
    print("=" * 80)
    print(f"Tổng class input: {len(class_folders)}")
    print(f"Class giữ lại: {kept_classes}")
    print(f"Class bỏ qua: {skipped_classes}")
    print(f"Tổng ảnh input: {total_input_images}")
    print(f"Tổng ảnh output: {total_output_images}")
    print(f"Output folder: {OUTPUT_VAL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
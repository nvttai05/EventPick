from pathlib import Path
import random
import shutil
import csv
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Hoctap\CK_KHDL\EventPick_v1")

SOURCE_VAL_FINAL_DIR = PROJECT_ROOT / "data" / "val_final" / "Event_Public"

OUTPUT_TEST_FINAL_DIR = PROJECT_ROOT / "data" / "test_final" / "Event_Public"

REPORT_PATH = PROJECT_ROOT / "reports" / "test_final_split_report.csv"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# Chọn 30% identity từ val_final để copy sang test_final
TEST_IDENTITY_RATIO = 0.55

# Giới hạn số ảnh mỗi người trong test.
# Nếu muốn lấy toàn bộ ảnh mỗi người thì để None.
MAX_IMAGES_PER_IDENTITY = None
# Ví dụ nhẹ hơn:
# MAX_IMAGES_PER_IDENTITY = 30

RANDOM_SEED = 42

OVERWRITE_OUTPUT = True


# ============================================================
# UTILS
# ============================================================

def get_images(folder: Path):
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def copy_identity_folder(source_dir: Path, output_dir: Path, rng: random.Random):
    images = get_images(source_dir)

    if MAX_IMAGES_PER_IDENTITY is not None and len(images) > MAX_IMAGES_PER_IDENTITY:
        selected_images = rng.sample(images, MAX_IMAGES_PER_IDENTITY)
        selected_images = sorted(selected_images)
    else:
        selected_images = images

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in selected_images:
        shutil.copy2(image_path, output_dir / image_path.name)

    return len(images), len(selected_images)


# ============================================================
# MAIN
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)

    if not SOURCE_VAL_FINAL_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy SOURCE_VAL_FINAL_DIR: {SOURCE_VAL_FINAL_DIR}")

    output_root = OUTPUT_TEST_FINAL_DIR.parent

    if output_root.exists() and OVERWRITE_OUTPUT:
        print(f"Đang xóa test_final cũ: {output_root}")
        shutil.rmtree(output_root)

    OUTPUT_TEST_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    person_dirs = [
        p for p in sorted(SOURCE_VAL_FINAL_DIR.iterdir())
        if p.is_dir() and len(get_images(p)) > 0
    ]

    rng.shuffle(person_dirs)

    num_test = max(1, int(len(person_dirs) * TEST_IDENTITY_RATIO))
    test_persons = person_dirs[:num_test]

    total_source_images = 0
    total_copied_images = 0
    report_rows = []

    print("=" * 80)
    print("COPY VAL_FINAL → TEST_FINAL")
    print("=" * 80)
    print(f"Source val_final : {SOURCE_VAL_FINAL_DIR}")
    print(f"Output test_final: {OUTPUT_TEST_FINAL_DIR}")
    print(f"Total identities in val_final: {len(person_dirs)}")
    print(f"Selected test identities     : {len(test_persons)}")
    print(f"TEST_IDENTITY_RATIO          : {TEST_IDENTITY_RATIO}")
    print(f"MAX_IMAGES_PER_IDENTITY      : {MAX_IMAGES_PER_IDENTITY}")
    print("=" * 80)

    for person_dir in tqdm(test_persons, desc="Copy identities to test_final"):
        output_person_dir = OUTPUT_TEST_FINAL_DIR / person_dir.name

        num_source, num_copied = copy_identity_folder(
            source_dir=person_dir,
            output_dir=output_person_dir,
            rng=rng
        )

        total_source_images += num_source
        total_copied_images += num_copied

        report_rows.append({
            "person_id": person_dir.name,
            "source_dir": str(person_dir),
            "output_dir": str(output_person_dir),
            "num_source_images": num_source,
            "num_copied_images": num_copied,
        })

    with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "person_id",
                "source_dir",
                "output_dir",
                "num_source_images",
                "num_copied_images",
            ]
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print("=" * 80)
    print("HOÀN TẤT COPY TEST_FINAL")
    print("=" * 80)
    print(f"Test identities     : {len(test_persons)}")
    print(f"Source images       : {total_source_images}")
    print(f"Copied images       : {total_copied_images}")
    print(f"Output test_final   : {OUTPUT_TEST_FINAL_DIR.parent}")
    print(f"Report              : {REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
from pathlib import Path
import random
import shutil
import csv
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Hoctap\CK_KHDL\EventPick_v1")

SOURCE_PUBLIC_DIR = PROJECT_ROOT / "data" / "val_processed" / "Event_Public"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "public_80_20"
TRAIN_OUTPUT_DIR = OUTPUT_ROOT / "train" / "Event_Public"
VAL_OUTPUT_DIR = OUTPUT_ROOT / "val" / "Event_Public"

SPLIT_REPORT_CSV = OUTPUT_ROOT / "public_80_20_split_report.csv"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

MIN_IMAGES_PER_CLASS = 5

# None = giữ toàn bộ ảnh mỗi người
MAX_IMAGES_PER_CLASS = None

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

OVERWRITE_OUTPUT = True


# ============================================================
# UTILS
# ============================================================

def get_images(folder: Path):
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def copy_identity_folder(source_dir: Path, output_dir: Path, images):
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(image_path, output_dir / image_path.name)


# ============================================================
# MAIN
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)

    if not SOURCE_PUBLIC_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy SOURCE_PUBLIC_DIR: {SOURCE_PUBLIC_DIR}")

    if OUTPUT_ROOT.exists() and OVERWRITE_OUTPUT:
        print(f"Đang xóa output cũ: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    identity_dirs = [
        p for p in sorted(SOURCE_PUBLIC_DIR.iterdir())
        if p.is_dir()
    ]

    valid_identities = []

    for person_dir in identity_dirs:
        images = get_images(person_dir)

        if len(images) < MIN_IMAGES_PER_CLASS:
            continue

        valid_identities.append({
            "person_name": person_dir.name,
            "person_dir": person_dir,
            "num_images": len(images),
            "images": images,
        })

    rng.shuffle(valid_identities)

    num_train = int(len(valid_identities) * TRAIN_RATIO)

    train_identities = valid_identities[:num_train]
    val_identities = valid_identities[num_train:]

    report_rows = []

    train_images_count = 0
    val_images_count = 0

    print("=" * 80)
    print("BẮT ĐẦU SPLIT PUBLIC 80/20 THEO IDENTITY")
    print("=" * 80)
    print(f"Source: {SOURCE_PUBLIC_DIR}")
    print(f"Train output: {TRAIN_OUTPUT_DIR}")
    print(f"Val output: {VAL_OUTPUT_DIR}")
    print(f"Total valid identities: {len(valid_identities)}")
    print(f"Train identities: {len(train_identities)}")
    print(f"Val identities: {len(val_identities)}")
    print("=" * 80)

    for item in tqdm(train_identities, desc="Copy train identities"):
        person_name = item["person_name"]
        images = item["images"]

        if MAX_IMAGES_PER_CLASS is not None and len(images) > MAX_IMAGES_PER_CLASS:
            images = rng.sample(images, MAX_IMAGES_PER_CLASS)
            images = sorted(images)

        output_dir = TRAIN_OUTPUT_DIR / person_name
        copy_identity_folder(item["person_dir"], output_dir, images)

        train_images_count += len(images)

        report_rows.append({
            "split": "train",
            "person_name": person_name,
            "original_images": item["num_images"],
            "copied_images": len(images),
            "output_dir": str(output_dir),
        })

    for item in tqdm(val_identities, desc="Copy val identities"):
        person_name = item["person_name"]
        images = item["images"]

        if MAX_IMAGES_PER_CLASS is not None and len(images) > MAX_IMAGES_PER_CLASS:
            images = rng.sample(images, MAX_IMAGES_PER_CLASS)
            images = sorted(images)

        output_dir = VAL_OUTPUT_DIR / person_name
        copy_identity_folder(item["person_dir"], output_dir, images)

        val_images_count += len(images)

        report_rows.append({
            "split": "val",
            "person_name": person_name,
            "original_images": item["num_images"],
            "copied_images": len(images),
            "output_dir": str(output_dir),
        })

    with open(SPLIT_REPORT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "person_name",
                "original_images",
                "copied_images",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print("=" * 80)
    print("HOÀN TẤT SPLIT PUBLIC 80/20")
    print("=" * 80)
    print(f"Train identities: {len(train_identities)}")
    print(f"Train images: {train_images_count}")
    print(f"Val identities: {len(val_identities)}")
    print(f"Val images: {val_images_count}")
    print(f"Report CSV: {SPLIT_REPORT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
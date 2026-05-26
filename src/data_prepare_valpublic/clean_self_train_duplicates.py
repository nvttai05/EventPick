from pathlib import Path
import shutil
import hashlib
from PIL import Image
import imagehash
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Hoctap\CK_KHDL\EventPick_v1")

SOURCE_TRAIN_DIR = PROJECT_ROOT / "data" / "train"
OUTPUT_TRAIN_CLEAN_DIR = PROJECT_ROOT / "data" / "train_clean"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class ít hơn số này thì bỏ
MIN_IMAGES_PER_CLASS = 5

# Giới hạn ảnh mỗi người sau lọc lặp
MAX_IMAGES_PER_CLASS = 50

# Perceptual hash threshold:
# 0 = gần như giống hệt
# 3-5 = lọc khá mạnh ảnh gần giống
# 8+ = rất mạnh, có thể xóa nhầm ảnh khác biểu cảm nhẹ
PHASH_DISTANCE_THRESHOLD = 5

OVERWRITE_OUTPUT = True


# ============================================================
# UTILS
# ============================================================

def get_file_md5(path: Path) -> str:
    hash_md5 = hashlib.md5()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_images(folder: Path):
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def compute_phash(path: Path):
    try:
        image = Image.open(path).convert("RGB")
        return imagehash.phash(image)
    except Exception:
        return None


def is_too_similar(new_hash, kept_hashes, threshold: int) -> bool:
    for old_hash in kept_hashes:
        if old_hash is None:
            continue

        distance = new_hash - old_hash

        if distance <= threshold:
            return True

    return False


def select_diverse_images(images):
    """
    Lọc ảnh trùng/gần trùng trong một folder người.
    """
    selected = []
    selected_phashes = []
    seen_md5 = set()

    for image_path in images:
        try:
            md5 = get_file_md5(image_path)

            if md5 in seen_md5:
                continue

            seen_md5.add(md5)

            p_hash = compute_phash(image_path)

            if p_hash is None:
                continue

            if is_too_similar(
                new_hash=p_hash,
                kept_hashes=selected_phashes,
                threshold=PHASH_DISTANCE_THRESHOLD,
            ):
                continue

            selected.append(image_path)
            selected_phashes.append(p_hash)

            if len(selected) >= MAX_IMAGES_PER_CLASS:
                break

        except Exception:
            continue

    return selected


# ============================================================
# MAIN
# ============================================================

def main():
    if not SOURCE_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy SOURCE_TRAIN_DIR: {SOURCE_TRAIN_DIR}")

    if OUTPUT_TRAIN_CLEAN_DIR.exists() and OVERWRITE_OUTPUT:
        print(f"Đang xóa output cũ: {OUTPUT_TRAIN_CLEAN_DIR}")
        shutil.rmtree(OUTPUT_TRAIN_CLEAN_DIR)

    OUTPUT_TRAIN_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    person_folders = []

    for event_dir in sorted(SOURCE_TRAIN_DIR.iterdir()):
        if not event_dir.is_dir():
            continue

        for person_dir in sorted(event_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            person_folders.append((event_dir.name, person_dir.name, person_dir))

    total_input_images = 0
    total_output_images = 0
    kept_classes = 0
    skipped_classes = 0

    print("=" * 80)
    print("BẮT ĐẦU LỌC LẶP TRAIN TỰ THU THẬP")
    print("=" * 80)
    print(f"Input : {SOURCE_TRAIN_DIR}")
    print(f"Output: {OUTPUT_TRAIN_CLEAN_DIR}")
    print(f"MIN_IMAGES_PER_CLASS: {MIN_IMAGES_PER_CLASS}")
    print(f"MAX_IMAGES_PER_CLASS: {MAX_IMAGES_PER_CLASS}")
    print(f"PHASH_DISTANCE_THRESHOLD: {PHASH_DISTANCE_THRESHOLD}")
    print("=" * 80)

    for event_name, person_name, person_dir in tqdm(person_folders, desc="Cleaning train folders"):
        images = get_images(person_dir)
        total_input_images += len(images)

        if len(images) < MIN_IMAGES_PER_CLASS:
            skipped_classes += 1
            continue

        selected_images = select_diverse_images(images)

        if len(selected_images) < MIN_IMAGES_PER_CLASS:
            skipped_classes += 1
            continue

        output_person_dir = OUTPUT_TRAIN_CLEAN_DIR / event_name / person_name
        output_person_dir.mkdir(parents=True, exist_ok=True)

        for image_path in selected_images:
            shutil.copy2(image_path, output_person_dir / image_path.name)

        kept_classes += 1
        total_output_images += len(selected_images)

    print()
    print("=" * 80)
    print("HOÀN TẤT LỌC LẶP TRAIN")
    print("=" * 80)
    print(f"Tổng class input : {len(person_folders)}")
    print(f"Class giữ lại    : {kept_classes}")
    print(f"Class bỏ qua     : {skipped_classes}")
    print(f"Tổng ảnh input   : {total_input_images}")
    print(f"Tổng ảnh output  : {total_output_images}")
    print(f"Output folder    : {OUTPUT_TRAIN_CLEAN_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
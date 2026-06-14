from pathlib import Path
import random
import shutil
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Hoctap\CK_KHDL\EventPick_v1")

SOURCE_PUBLIC_DIR = PROJECT_ROOT / "data" / "val_processed" / "Event_Public"

PUBLIC_TRAIN_DIR = PROJECT_ROOT / "data" / "public_split" / "train" / "Event_Public"
PUBLIC_VAL_DIR = PROJECT_ROOT / "data" / "public_split" / "val" / "Event_Public"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_IDENTITY_RATIO = 0.7

MIN_IMAGES_PER_CLASS = 12
MAX_IMAGES_PUBLIC_TRAIN_PER_CLASS = 60
MAX_IMAGES_PUBLIC_VAL_PER_CLASS = 35

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


def sample_images(images, max_images, rng: random.Random):
    if len(images) <= max_images:
        return sorted(images)

    return sorted(rng.sample(images, max_images))


def copy_identity_folder(person_dir: Path, output_dir: Path, max_images: int, rng: random.Random):
    images = get_images(person_dir)

    if len(images) < MIN_IMAGES_PER_CLASS:
        return 0

    selected = sample_images(images, max_images, rng)

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in selected:
        shutil.copy2(image_path, output_dir / image_path.name)

    return len(selected)


# ============================================================
# MAIN
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)

    if not SOURCE_PUBLIC_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy SOURCE_PUBLIC_DIR: {SOURCE_PUBLIC_DIR}")

    output_root = PROJECT_ROOT / "data" / "public_split"

    if output_root.exists() and OVERWRITE_OUTPUT:
        print(f"Đang xóa output cũ: {output_root}")
        shutil.rmtree(output_root)

    PUBLIC_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_VAL_DIR.mkdir(parents=True, exist_ok=True)

    person_dirs = [
        p for p in sorted(SOURCE_PUBLIC_DIR.iterdir())
        if p.is_dir()
    ]

    valid_person_dirs = [
        p for p in person_dirs
        if len(get_images(p)) >= MIN_IMAGES_PER_CLASS
    ]

    rng.shuffle(valid_person_dirs)

    num_train = int(len(valid_person_dirs) * TRAIN_IDENTITY_RATIO)

    train_persons = valid_person_dirs[:num_train]
    val_persons = valid_person_dirs[num_train:]

    train_images = 0
    val_images = 0

    print("=" * 80)
    print("BẮT ĐẦU SPLIT PUBLIC IDENTITIES")
    print("=" * 80)
    print(f"Source public: {SOURCE_PUBLIC_DIR}")
    print(f"Public train : {PUBLIC_TRAIN_DIR}")
    print(f"Public val   : {PUBLIC_VAL_DIR}")
    print(f"Total valid identities: {len(valid_person_dirs)}")
    print(f"Train identities: {len(train_persons)}")
    print(f"Val identities  : {len(val_persons)}")
    print("=" * 80)

    for person_dir in tqdm(train_persons, desc="Copy public train"):
        out_dir = PUBLIC_TRAIN_DIR / person_dir.name
        train_images += copy_identity_folder(
            person_dir=person_dir,
            output_dir=out_dir,
            max_images=MAX_IMAGES_PUBLIC_TRAIN_PER_CLASS,
            rng=rng,
        )

    for person_dir in tqdm(val_persons, desc="Copy public val"):
        out_dir = PUBLIC_VAL_DIR / person_dir.name
        val_images += copy_identity_folder(
            person_dir=person_dir,
            output_dir=out_dir,
            max_images=MAX_IMAGES_PUBLIC_VAL_PER_CLASS,
            rng=rng,
        )

    print()
    print("=" * 80)
    print("HOÀN TẤT SPLIT PUBLIC")
    print("=" * 80)
    print(f"Public train identities: {len(train_persons)}")
    print(f"Public train images    : {train_images}")
    print(f"Public val identities  : {len(val_persons)}")
    print(f"Public val images      : {val_images}")
    print("=" * 80)


if __name__ == "__main__":
    main()
import os

from PIL import Image
from torch.utils.data import Dataset

MIN_IMAGES_PER_CLASS = 5


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.label_to_idx = {}

        current_label = 0
        skipped_classes = 0

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Không tìm thấy dataset folder: {root_dir}")

        events = sorted(os.listdir(root_dir))

        for event_name in events:
            event_path = os.path.join(root_dir, event_name)

            if not os.path.isdir(event_path):
                continue

            persons = sorted(os.listdir(event_path))

            for person_name in persons:
                person_path = os.path.join(event_path, person_name)

                if not os.path.isdir(person_path):
                    continue

                images = [
                    f for f in os.listdir(person_path)
                    if f.lower().endswith((
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp",
                        ".bmp"
                    ))
                ]

                if len(images) < MIN_IMAGES_PER_CLASS:
                    skipped_classes += 1
                    continue

                global_person_id = f"{event_name}_{person_name}"

                if global_person_id not in self.label_to_idx:
                    self.label_to_idx[global_person_id] = current_label
                    current_label += 1

                label = self.label_to_idx[global_person_id]

                for image_name in images:
                    image_path = os.path.join(person_path, image_name)
                    self.samples.append((image_path, label))

        print(f"Dataset root : {self.root_dir}")
        print(f"Total images : {len(self.samples)}")
        print(f"Total classes: {len(self.label_to_idx)}")
        print(f"Skipped classes with < {MIN_IMAGES_PER_CLASS} images: {skipped_classes}")

        if len(self.samples) == 0:
            raise ValueError(
                f"Không load được ảnh nào từ {self.root_dir}. "
                "Cấu trúc cần là root/event/person/image."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
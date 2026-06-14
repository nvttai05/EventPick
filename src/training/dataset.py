import os

from PIL import Image
from torch.utils.data import Dataset

MIN_IMAGES_PER_CLASS = 5
# MAX_IMAGES_PER_CLASS = 50 # thử train với lệch số lượng thử xem kết quả thế nào, nếu không
# ổn thì thêm MAX này cho các classes đỡ lệch nhiều.

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.label_to_idx = {}

        current_label = 0

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
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                if len(images) < MIN_IMAGES_PER_CLASS:
                    continue

                global_person_id = f"{event_name}_{person_name}"

                if global_person_id not in self.label_to_idx:
                    self.label_to_idx[global_person_id] = current_label
                    current_label += 1

                label = self.label_to_idx[global_person_id]

                for image_name in images:
                    image_path = os.path.join(person_path, image_name)

                    self.samples.append((image_path, label))

        print(f"Total images : {len(self.samples)}")
        print(f"Total classes: {len(self.label_to_idx)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
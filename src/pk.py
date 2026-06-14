import random
from collections import defaultdict

from torch.utils.data import Sampler

class PKSampler(Sampler):
    def __init__(self, dataset, p=16, k=4):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.batch_size = p * k

        # label -> list indices
        self.label_to_indices = defaultdict(list)

        for idx, (_, label) in enumerate(dataset.samples):
            self.label_to_indices[label].append(idx)

        # chỉ giữ class đủ K ảnh
        self.valid_labels = [
            label
            for label, indices in self.label_to_indices.items()
            if len(indices) >= k
        ]

        if len(self.valid_labels) < p:
            raise ValueError(
                f"Không đủ class để sample P={p}"
            )

    def __iter__(self):
        labels = self.valid_labels.copy()
        random.shuffle(labels)

        batch = []

        while len(labels) >= self.p:
            selected_labels = labels[:self.p]
            labels = labels[self.p:]

            for label in selected_labels:
                indices = self.label_to_indices[label]
                sampled = random.sample(indices, self.k)
                batch.extend(sampled)

            yield batch

            batch = []

    def __len__(self):
        return len(self.valid_labels) // self.p
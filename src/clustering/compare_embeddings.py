import os
import cv2
import numpy as np

from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

from tqdm import tqdm

import torch

# =========================================================
# CONFIG
# =========================================================
CLUSTER_DIR = "./clustered/Event_Trai_v1/Nam"

THRESHOLD = 0.35

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device:", device)

# =========================================================
# FACENET
# =========================================================
model = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

# =========================================================
# TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# =========================================================
# GET FOLDER EMBEDDINGS
# =========================================================
folder_embeddings = {}

folders = [
    f for f in os.listdir(CLUSTER_DIR)
    if os.path.isdir(
        os.path.join(CLUSTER_DIR, f)
    )
]

pg_bar = tqdm(
    folders,
    desc="Processing folders",
    colour="cyan"
)

for idx, folder in enumerate(pg_bar):

    folder_path = os.path.join(
        CLUSTER_DIR,
        folder
    )

    pg_bar.set_description(
        f"Index: {idx}"
    )

    embeddings = []

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ))
    ]

    for file in files:

        path = os.path.join(
            folder_path,
            file
        )

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # =================================================
        # preprocess
        # =================================================
        tensor = transform(img)\
            .unsqueeze(0)\
            .to(device)

        # =================================================
        # embedding
        # =================================================
        with torch.no_grad():

            emb = model(tensor)

        emb = emb.cpu().numpy()[0]

        # normalize
        emb = emb / np.linalg.norm(emb)

        embeddings.append(emb)

    if len(embeddings) == 0:
        continue

    # =====================================================
    # mean embedding
    # =====================================================
    mean_emb = np.mean(
        embeddings,
        axis=0
    )

    mean_emb = mean_emb / np.linalg.norm(mean_emb)

    folder_embeddings[folder] = mean_emb

print(
    "\nTotal folders:",
    len(folder_embeddings)
)

# =========================================================
# COMPARE FOLDERS
# =========================================================
print("\nPossible duplicate folders:\n")

for f1, f2 in combinations(
    folder_embeddings.keys(),
    2
):

    emb1 = folder_embeddings[f1]
    emb2 = folder_embeddings[f2]

    dist = cosine_distances(
        [emb1],
        [emb2]
    )[0][0]

    if dist < THRESHOLD:

        print(
            f"{f1} <-> {f2}"
        )

        print(
            f"Distance: {dist:.4f}\n"
        )
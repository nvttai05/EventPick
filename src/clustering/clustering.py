import os
import shutil
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# ==========================================
# CONFIG
# ==========================================
INPUT_DIR_1 = "./clustered/Event_Trai_v1/from_unknown/unknown"
# INPUT_DIR_2 = "./cleaned_faces/Event_SV/"
# INPUT_DIR_3 = "./cleaned_faces/Event_Trai/"
OUTPUT_DIR = "./clustered/Event_Trai_v1/from_unknown/from_unknown"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device:", device)

# ==========================================
# FACENET MODEL
# ==========================================
model = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

# ==========================================
# TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),

    # FaceNet normalize
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ==========================================
# LOAD FILES
# ==========================================
files_1 = [
    f for f in os.listdir(INPUT_DIR_1)
    if f.lower().endswith((
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ))
]
# files_2 = [
#     f for f in os.listdir(INPUT_DIR_2)
#     if f.lower().endswith((
#         ".jpg",
#         ".jpeg",
#         ".png",
#         ".webp"
#     ))
# ]
# files_3 = [
#     f for f in os.listdir(INPUT_DIR_3)
#     if f.lower().endswith((
#         ".jpg",
#         ".jpeg",
#         ".png",
#         ".webp"
#     ))
# ]
all_files = []

for f in files_1:
    all_files.append(
        os.path.join(INPUT_DIR_1, f)
    )

# for f in files_2:
#     all_files.append(
#         os.path.join(INPUT_DIR_2, f)
#     )
#
# for f in files_3:
#     all_files.append(
#         os.path.join(INPUT_DIR_3, f)
#     )

print("Total images:", len(all_files))

embeddings = []
paths = []

# ==========================================
# EXTRACT EMBEDDINGS
# ==========================================
for path in tqdm(all_files, desc="Extracting embeddings"):
    try:
        img = Image.open(path).convert("RGB")
    except:
        print("Cannot open:", path)
        continue

    # --------------------------------------
    # preprocess
    # --------------------------------------
    tensor = transform(img)\
        .unsqueeze(0)\
        .to(device)

    # --------------------------------------
    # embedding
    # --------------------------------------
    with torch.no_grad():
        emb = model(tensor)

    emb = emb.cpu().numpy()[0]

    # --------------------------------------
    # normalize embedding
    # IMPORTANT
    # --------------------------------------
    emb = emb / np.linalg.norm(emb)

    embeddings.append(emb)
    paths.append(path)

# ==========================================
# CHECK
# ==========================================
if len(embeddings) == 0:
    print("No embeddings extracted.")
    exit()

embeddings = np.array(embeddings)

print("Embedding shape:", embeddings.shape)

# ==========================================
# DEBUG DISTANCE
# ==========================================
dist_matrix = cosine_distances(embeddings)

print("\nDistance statistics:")
print("Min :", dist_matrix.min())
print("Max :", dist_matrix.max())
print("Mean:", dist_matrix.mean())

# ==========================================
# CLUSTER
# ==========================================
print("\nClustering...")

cluster = DBSCAN(
    eps=0.01,
    min_samples=3,
    metric='cosine'
)

labels = cluster.fit_predict(embeddings)

# ==========================================
# STATS
# ==========================================
unique_labels = set(labels)

num_clusters = len(unique_labels)

if -1 in unique_labels:
    num_clusters -= 1

print(f"\nClusters found: {num_clusters}")

for label in unique_labels:

    if label == -1:
        continue

    count = np.sum(labels == label)

    print(f"Cluster {label}: {count} images")

unknown_count = np.sum(labels == -1)

print(f"Unknown: {unknown_count}")

# ==========================================
# SAVE RESULTS
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for path, label in tqdm(
    list(zip(paths, labels)),
    desc="Saving clustered images"
):

    # --------------------------------------
    # folder
    # --------------------------------------
    if label == -1:
        folder = os.path.join(
            OUTPUT_DIR,
            "unknown"
        )
    else:
        folder = os.path.join(
            OUTPUT_DIR,
            f"person_{label}"
        )

    os.makedirs(folder, exist_ok=True)

    # --------------------------------------
    # save
    # --------------------------------------
    save_path = os.path.join(
        folder,
        os.path.basename(path)
    )

    shutil.copy(path, save_path)

print("\nDONE")
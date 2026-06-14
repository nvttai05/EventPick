# import random
# import sys
# from collections import defaultdict
#
# import numpy as np
# import torch
#
# from tqdm import tqdm
#
#
# # EXTRACT EMBEDDINGS
# @torch.no_grad()
# def extract_embeddings(model,loader,device):
#     model.eval()
#
#     all_embeddings = []
#     all_labels = []
#
#     for images, labels in tqdm(loader, desc="Extract embeddings", colour="yellow", file=sys.stdout):
#         images = images.to(device, non_blocking=True)
#         embeddings = model(images)
#         all_embeddings.append(embeddings.cpu())
#         all_labels.append(labels)
#
#     if len(all_embeddings) == 0:
#         return None, None
#
#     all_embeddings = torch.cat(all_embeddings, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
#
#     return all_embeddings, all_labels
#
#
# # SPLIT GALLERY / QUERY
# def build_gallery_query_sets(embeddings, labels, gallery_per_person=1):
#     label_to_indices = defaultdict(list)
#     labels_np = labels.numpy()
#     for idx, label in enumerate(labels_np):
#         label_to_indices[int(label)].append(idx)
#
#     gallery_indices = []
#     query_indices = []
#
#     for label, indices in label_to_indices.items():
#         if len(indices) < 2:
#             continue
#         random.shuffle(indices)
#
#         gallery = indices[:gallery_per_person]
#         query = indices[gallery_per_person:]
#
#         if len(query) == 0:
#             continue
#
#         gallery_indices.extend(gallery)
#         query_indices.extend(query)
#
#     if len(gallery_indices) == 0:
#         return None
#
#     gallery_embeddings = embeddings[gallery_indices]
#     gallery_labels = labels[gallery_indices]
#
#     query_embeddings = embeddings[query_indices]
#     query_labels = labels[query_indices]
#
#     return {
#         "gallery_embeddings": gallery_embeddings,
#         "gallery_labels": gallery_labels,
#         "query_embeddings": query_embeddings,
#         "query_labels": query_labels,
#     }
#
# # RETRIEVAL METRICS
# @torch.no_grad()
# def compute_recall_at_k(query_embeddings, query_labels, gallery_embeddings, gallery_labels, ks=(1, 5, 10)):
#     results = {}
#     similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())
#     max_k = max(ks)
#
#     # top-k nearest gallery
#     topk_indices = torch.topk(similarity_matrix, k=max_k, dim=1).indices
#
#     for k in ks:
#         correct = 0
#         for i in range(len(query_labels)):
#             query_label = query_labels[i]
#
#             retrieved_indices = topk_indices[i, :k]
#
#             retrieved_labels = gallery_labels[retrieved_indices]
#
#             # nếu trong top-k có đúng person
#             if (retrieved_labels == query_label).any():
#                 correct += 1
#
#         recall_k = correct / len(query_labels)
#         results[f"recall@{k}"] = float(recall_k)
#     return results
#
# # SIMILARITY STATS
# @torch.no_grad()
# def compute_similarity_stats(
#     query_embeddings,
#     query_labels,
#     gallery_embeddings,
#     gallery_labels
# ):
#
#     similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())
#
#     positive_similarities = []
#     negative_similarities = []
#
#     for i in range(len(query_labels)):
#         query_label = query_labels[i]
#         similarities = similarity_matrix[i]
#         positive_mask = (gallery_labels == query_label)
#         negative_mask = (gallery_labels != query_label)
#         positive_sims = similarities[positive_mask]
#         negative_sims = similarities[negative_mask]
#
#         if len(positive_sims) > 0:
#             positive_similarities.extend(positive_sims.cpu().tolist())
#
#         if len(negative_sims) > 0:
#             negative_similarities.extend(negative_sims.cpu().tolist())
#     positive_mean = float(np.mean(positive_similarities))
#     negative_mean = float(np.mean(negative_similarities))
#     positive_std = float(np.std(positive_similarities))
#     negative_std = float(np.std(negative_similarities))
#     return {
#         "positive_mean": positive_mean,
#         "negative_mean": negative_mean,
#         "positive_std": positive_std,
#         "negative_std": negative_std,
#     }
#
# # MAIN VALIDATE
# @torch.no_grad()
# def validate(model, loader, device, gallery_per_person=1, ks=(1, 5, 10)):
#     # EXTRACT EMBEDDINGS
#     embeddings, labels = extract_embeddings(model=model, loader=loader, device=device)
#
#     if embeddings is None:
#         print("[VALIDATION WARNING] No embeddings extracted.")
#         return {"recall@1": 0.0}
#
#     # BUILD GALLERY / QUERY
#     split_data = build_gallery_query_sets(
#         embeddings=embeddings,
#         labels=labels,
#         gallery_per_person=gallery_per_person
#     )
#
#     if split_data is None:
#         print("[VALIDATION WARNING] Cannot build gallery/query sets.")
#         return {"recall@1": 0.0}
#
#     gallery_embeddings = split_data["gallery_embeddings"].to(device)
#     gallery_labels = split_data["gallery_labels"].to(device)
#     query_embeddings = split_data["query_embeddings"].to(device)
#     query_labels = split_data["query_labels"].to(device)
#
#     # COMPUTE RETRIEVAL METRICS
#     metrics = compute_recall_at_k(
#         query_embeddings=query_embeddings,
#         query_labels=query_labels,
#         gallery_embeddings=gallery_embeddings,
#         gallery_labels=gallery_labels,
#         ks=ks
#     )
#
#     similarity_stats = compute_similarity_stats(
#         query_embeddings=query_embeddings,
#         query_labels=query_labels,
#         gallery_embeddings=gallery_embeddings,
#         gallery_labels=gallery_labels,
#     )
#     metrics.update(similarity_stats)
#
#     print("\n========== VALIDATION ==========")
#     for k, value in metrics.items():
#         print(
#             f"{k.upper():<12}: {value:.4f}")
#     print("================================\n")
#
#     return metrics

import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm


# =========================================================
# 1. EXTRACT EMBEDDINGS
# =========================================================
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extract embeddings", file=sys.stdout):
        images = images.to(device, non_blocking=True)

        embeddings = model(images)

        # normalize embedding (VERY IMPORTANT for cosine similarity)
        embeddings = F.normalize(embeddings, dim=1)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    if len(all_embeddings) == 0:
        return None, None

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_embeddings, all_labels


# =========================================================
# 2. BUILD PROTOTYPE GALLERY (MEAN EMBEDDING PER PERSON)
# =========================================================
@torch.no_grad()
def build_prototype_gallery(embeddings, labels):
    label_to_embs = defaultdict(list)

    labels_np = labels.numpy()

    for idx, label in enumerate(labels_np):
        label_to_embs[int(label)].append(embeddings[idx])

    gallery_embeddings = []
    gallery_labels = []

    for label, embs in label_to_embs.items():

        if len(embs) == 0:
            continue

        embs = torch.stack(embs, dim=0)

        # mean embedding (prototype)
        proto = embs.mean(dim=0)

        # normalize again (VERY IMPORTANT)
        proto = F.normalize(proto, dim=0)

        gallery_embeddings.append(proto)
        gallery_labels.append(label)

    if len(gallery_embeddings) == 0:
        return None

    gallery_embeddings = torch.stack(gallery_embeddings, dim=0)
    gallery_labels = torch.tensor(gallery_labels)

    return {
        "gallery_embeddings": gallery_embeddings,
        "gallery_labels": gallery_labels
    }


# =========================================================
# 3. RETRIEVAL METRICS (Recall@K)
# =========================================================
@torch.no_grad()
def compute_recall_at_k(query_embeddings, query_labels, gallery_embeddings, gallery_labels, ks=(1, 5, 10)):

    similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())

    max_k = max(ks)
    topk_indices = torch.topk(similarity_matrix, k=max_k, dim=1).indices

    results = {}

    for k in ks:
        correct = 0

        for i in range(len(query_labels)):
            q_label = query_labels[i]

            retrieved_labels = gallery_labels[topk_indices[i, :k]]

            if (retrieved_labels == q_label).any():
                correct += 1

        results[f"recall@{k}"] = float(correct / len(query_labels))

    return results


# =========================================================
# 4. SIMILARITY STATISTICS (POS / NEG GAP)
# =========================================================
@torch.no_grad()
def compute_similarity_stats(query_embeddings, query_labels, gallery_embeddings, gallery_labels):

    similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())

    pos_sims = []
    neg_sims = []

    for i in range(len(query_labels)):

        sims = similarity_matrix[i]
        q_label = query_labels[i]

        pos_mask = (gallery_labels == q_label)
        neg_mask = (gallery_labels != q_label)

        pos = sims[pos_mask]
        neg = sims[neg_mask]

        if len(pos) > 0:
            pos_sims.extend(pos.cpu().tolist())

        if len(neg) > 0:
            neg_sims.extend(neg.cpu().tolist())

    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)

    return {
        "positive_mean": float(pos_sims.mean()) if len(pos_sims) else 0.0,
        "negative_mean": float(neg_sims.mean()) if len(neg_sims) else 0.0,
        "positive_std": float(pos_sims.std()) if len(pos_sims) else 0.0,
        "negative_std": float(neg_sims.std()) if len(neg_sims) else 0.0,
        "margin": float(pos_sims.mean() - neg_sims.mean()) if len(pos_sims) and len(neg_sims) else 0.0
    }


# =========================================================
# 5. VALIDATION PIPELINE
# =========================================================
@torch.no_grad()
def validate(model, loader, device, ks=(1, 5, 10)):

    # 1. extract embeddings
    embeddings, labels = extract_embeddings(model, loader, device)

    if embeddings is None:
        print("[VALIDATION] No embeddings found.")
        return {}

    # 2. build prototype gallery
    gallery = build_prototype_gallery(embeddings, labels)

    if gallery is None:
        print("[VALIDATION] Cannot build gallery.")
        return {}

    gallery_embeddings = gallery["gallery_embeddings"].to(device)
    gallery_labels = gallery["gallery_labels"].to(device)

    # 3. query = all embeddings (standard retrieval eval)
    query_embeddings = embeddings.to(device)
    query_labels = labels.to(device)

    # 4. metrics
    metrics = compute_recall_at_k(
        query_embeddings,
        query_labels,
        gallery_embeddings,
        gallery_labels,
        ks
    )

    stats = compute_similarity_stats(
        query_embeddings,
        query_labels,
        gallery_embeddings,
        gallery_labels
    )

    metrics.update(stats)

    # 5. print
    print("\n========== VALIDATION ==========")
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.4f}")
    print("================================\n")

    return metrics
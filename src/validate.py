import random
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# ============================================================
# 1. EXTRACT EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """
    Trích xuất embedding cho toàn bộ validation set.

    Output:
    - embeddings: numpy array shape (N, embedding_dim), đã L2 normalize
    - labels: numpy array shape (N,)
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    for images, targets in tqdm(loader, desc="Validating - extracting embeddings"):
        images = images.to(device, non_blocking=True)

        embeddings = model(images)

        # Model của bạn đã normalize trong forward,
        # nhưng normalize lại lần nữa để đảm bảo an toàn.
        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(targets.numpy())

    if len(all_embeddings) == 0:
        return np.array([]), np.array([])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_embeddings, all_labels


# ============================================================
# 2. PAIR SAMPLING
# ============================================================

def build_label_to_indices(labels):
    """
    Tạo mapping:
        label -> danh sách index ảnh thuộc label đó
    """
    label_to_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        label_to_indices[int(label)].append(idx)

    return label_to_indices


def sample_positive_pairs(label_to_indices, num_pairs):
    """
    Positive pair = 2 ảnh cùng class/người.
    """
    pairs = []

    valid_labels = [
        label for label, indices in label_to_indices.items()
        if len(indices) >= 2
    ]

    if len(valid_labels) == 0:
        return pairs

    for _ in range(num_pairs):
        label = random.choice(valid_labels)
        i, j = random.sample(label_to_indices[label], 2)
        pairs.append((i, j, 1))

    return pairs


def sample_negative_pairs(label_to_indices, num_pairs):
    """
    Negative pair = 2 ảnh khác class/người.
    """
    pairs = []

    labels = list(label_to_indices.keys())

    if len(labels) < 2:
        return pairs

    for _ in range(num_pairs):
        label_a, label_b = random.sample(labels, 2)

        i = random.choice(label_to_indices[label_a])
        j = random.choice(label_to_indices[label_b])

        pairs.append((i, j, 0))

    return pairs


def sample_validation_pairs(
    labels,
    num_positive_pairs=10000,
    num_negative_pairs=10000,
):
    """
    Lấy mẫu pairwise validation thay vì so toàn bộ N^2 cặp.
    """
    label_to_indices = build_label_to_indices(labels)

    positive_pairs = sample_positive_pairs(
        label_to_indices,
        num_positive_pairs
    )

    negative_pairs = sample_negative_pairs(
        label_to_indices,
        num_negative_pairs
    )

    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)

    return pairs


# ============================================================
# 3. METRICS
# ============================================================

def compute_pairwise_scores(embeddings, pairs):
    """
    Tính cosine similarity cho các cặp đã sample.

    Vì embeddings đã L2-normalized:
        cosine similarity = dot product
    """
    similarities = []
    ground_truths = []

    for i, j, gt_same in pairs:
        sim = float(np.dot(embeddings[i], embeddings[j]))

        similarities.append(sim)
        ground_truths.append(gt_same)

    similarities = np.array(similarities, dtype=np.float32)
    ground_truths = np.array(ground_truths, dtype=np.int64)

    return similarities, ground_truths


def evaluate_at_threshold(similarities, ground_truths, threshold):
    """
    Tính accuracy tại một threshold cụ thể.
    """
    predictions = similarities >= threshold
    accuracy = np.mean(predictions == ground_truths)

    return float(accuracy)


def find_best_threshold(
    similarities,
    ground_truths,
    thresholds=None,
):
    """
    Tìm threshold cho accuracy tốt nhất.
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.91, 0.05)

    best_acc = 0.0
    best_threshold = None

    threshold_results = []

    for threshold in thresholds:
        acc = evaluate_at_threshold(
            similarities,
            ground_truths,
            threshold
        )

        threshold_results.append({
            "threshold": float(threshold),
            "accuracy": float(acc),
        })

        if acc > best_acc:
            best_acc = float(acc)
            best_threshold = float(threshold)

    return best_acc, best_threshold, threshold_results


def compute_extra_stats(similarities, ground_truths):
    """
    Tính thêm thống kê similarity cho positive/negative pairs.
    Dùng để hiểu model đang học tốt hay chưa.
    """
    positive_sims = similarities[ground_truths == 1]
    negative_sims = similarities[ground_truths == 0]

    stats = {}

    if len(positive_sims) > 0:
        stats["positive_mean"] = float(np.mean(positive_sims))
        stats["positive_median"] = float(np.median(positive_sims))
        stats["positive_min"] = float(np.min(positive_sims))
        stats["positive_max"] = float(np.max(positive_sims))
    else:
        stats["positive_mean"] = None
        stats["positive_median"] = None
        stats["positive_min"] = None
        stats["positive_max"] = None

    if len(negative_sims) > 0:
        stats["negative_mean"] = float(np.mean(negative_sims))
        stats["negative_median"] = float(np.median(negative_sims))
        stats["negative_min"] = float(np.min(negative_sims))
        stats["negative_max"] = float(np.max(negative_sims))
    else:
        stats["negative_mean"] = None
        stats["negative_median"] = None
        stats["negative_min"] = None
        stats["negative_max"] = None

    return stats


# ============================================================
# 4. MAIN VALIDATE FUNCTION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    thresholds=None,
    num_positive_pairs=10000,
    num_negative_pairs=10000,
):
    """
    Validation cho face embedding model.

    Cách đánh giá:
    - Extract embedding cho validation set
    - Sample positive pairs: cùng người
    - Sample negative pairs: khác người
    - Tính cosine similarity
    - Tìm threshold tốt nhất
    - Trả về best accuracy

    Hàm này trả về 1 số float để tương thích với train.py:
        val_acc = validate(...)
    """

    embeddings, labels = extract_embeddings(
        model=model,
        loader=loader,
        device=device
    )

    if len(embeddings) == 0:
        print("[VALIDATION WARNING] Không có embedding nào.")
        return 0.0

    pairs = sample_validation_pairs(
        labels=labels,
        num_positive_pairs=num_positive_pairs,
        num_negative_pairs=num_negative_pairs,
    )

    if len(pairs) == 0:
        print("[VALIDATION WARNING] Không tạo được validation pairs.")
        return 0.0

    similarities, ground_truths = compute_pairwise_scores(
        embeddings=embeddings,
        pairs=pairs,
    )

    best_acc, best_threshold, threshold_results = find_best_threshold(
        similarities=similarities,
        ground_truths=ground_truths,
        thresholds=thresholds,
    )

    stats = compute_extra_stats(
        similarities=similarities,
        ground_truths=ground_truths,
    )

    print(
        "Validation | "
        f"best_acc={best_acc:.4f} | "
        f"best_threshold={best_threshold:.2f} | "
        f"pairs={len(pairs)} | "
        f"pos_mean={stats['positive_mean']:.4f} | "
        f"neg_mean={stats['negative_mean']:.4f}"
    )

    return best_acc
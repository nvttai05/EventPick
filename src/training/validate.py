import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

@torch.no_grad()
def validate(model, loader, device, threshold=0.7):
    model.eval()

    embeddings = []
    labels = []

    for images, target in loader:
        images = images.to(device)
        emb = model(images)
        embeddings.append(emb.cpu().numpy())
        labels.append(target.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # PAIRWISE VALIDATION
    correct = 0
    total = 0

    n = len(embeddings)

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[j]]
            )[0][0]

            pred_same = sim >= threshold
            gt_same = labels[i] == labels[j]

            if pred_same == gt_same:
                correct += 1

            total += 1

    accuracy = correct / total
    return accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5):
        super().__init__()

        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # embeddings = F.normalize(embeddings) # normalize trong model rồi
        weights = F.normalize(self.weight)
        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = one_hot * target_logits + (1 - one_hot) * cosine
        logits *= self.s
        loss = F.cross_entropy(logits, labels)

        return loss
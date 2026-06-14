# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class ArcFaceLoss(nn.Module):
#     def __init__(self, embedding_size, num_classes, s=64.0, m=0.5):
#         super().__init__()
#
#         self.s = s
#         self.m = m
#
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
#         nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, embeddings, labels):
#         # embeddings = F.normalize(embeddings) # normalize trong model rồi
#         weights = F.normalize(self.weight)
#         cosine = F.linear(embeddings, weights)
#         cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
#         theta = torch.acos(cosine)
#         target_logits = torch.cos(theta + self.m)
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, labels.view(-1, 1), 1)
#         logits = one_hot * target_logits + (1 - one_hot) * cosine
#         logits *= self.s
#         loss = F.cross_entropy(logits, labels)
#
#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_classes,
        s=30.0,
        m=0.3,
        easy_margin=False,
    ):
        super().__init__()

        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size)
        )
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(torch.pi - m))
        self.mm = torch.sin(torch.tensor(torch.pi - m)) * m

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m.to(cosine.device) - sine * self.sin_m.to(cosine.device)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(
                cosine > self.th.to(cosine.device),
                phi,
                cosine - self.mm.to(cosine.device)
            )

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s

        loss = F.cross_entropy(logits, labels)

        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet50,
    mobilenet_v2,

    ResNet50_Weights,
    MobileNet_V2_Weights
)

class FaceEmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name='resnet50',
        embedding_dim=512,
        pretrained=True
    ):
        super().__init__()
        self.model_name = model_name

        if model_name == 'resnet50':
            weights = (ResNet50_Weights.DEFAULT if pretrained else None)
            backbone = resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            in_features = 2048

        elif model_name == 'mobilenetv2':
            weights = (MobileNet_V2_Weights.DEFAULT if pretrained else None)
            backbone = mobilenet_v2(weights=weights)
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 1280

        else:
            raise ValueError(f'Unsupported model: {model_name}')

        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        if self.model_name == 'resnet50':
            x = self.backbone(x)

        elif self.model_name == 'mobilenetv2':
            x = self.backbone(x)
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)

        return x
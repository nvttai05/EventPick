# from torchvision import transforms
#
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
#
# eval_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

from torchvision import transforms


# Normalize chuẩn cho ResNet50 / MobileNetV2 pretrained ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Lật ngang: bình thường và rất hay dùng cho face
    transforms.RandomHorizontalFlip(p=0.5),

    # Aug màu nhẹ, không phá màu da/mặt quá nhiều
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.10,
        hue=0.02
    ),

    transforms.RandomRotation(degrees=5),

    # transforms.RandomAffine(
    #     degrees=0,
    #     translate=(0.02, 0.02),
    #     scale=(0.98, 1.02)
    # ),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
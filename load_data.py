import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from model import AlexNet

model = AlexNet()

# -------------------- 1 / common constants --------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -------------------- 2 / TRAIN transforms + loader --------------------
train_tf = T.Compose(
    [
        T.Resize(256),  # short side = 256 px
        T.RandomCrop(224),  # random translation
        T.RandomHorizontalFlip(),  # left–right reflection
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_data(batch_size=128, num_workers=8, root="/path/to/imagenet"):
    train_transform = T.Compose(
        [
            T.Resize(256),  # short side = 256 px
            T.RandomCrop(224),  # random translation
            T.RandomHorizontalFlip(),  # left–right reflection
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_set = ImageNet(root=root, split="train", transform=train_transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    test_transform = T.Compose(
        [
            T.Resize(256),
            T.TenCrop(224),  # 5 crops + 5 flips
            T.Lambda(
                lambda crops: torch.stack(
                    [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(c)) for c in crops]
                )
            ),  # (10, 3, 224, 224)
        ]
    )
    val_set = ImageNet(root=root, split="val", transform=test_transform)

    val_loader = DataLoader(
        val_set,
        batch_size=32,  # effective batch = 32 × 10 crops
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

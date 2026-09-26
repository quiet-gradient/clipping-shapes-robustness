from __future__ import annotations

from collections.abc import Sequence

import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
def cifar_transform(
    train: bool,
    mean: Sequence[float],
    std: Sequence[float],
    augmentation: bool = True,
) -> T.Compose:
    transforms: list = []
    if train and augmentation:
        transforms.extend(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
            ]
        )
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return T.Compose(transforms)

import os

import torch
from torchvision.datasets import ImageFolder


class CustomImageFolder(ImageFolder):
    def __init__(
        self, root, centers_dict, transform=None, target_transform=None
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.centers_dict = centers_dict

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        path = self.imgs[index][0]
        image_name = os.path.basename(path)

        centers_for_image = (
            self.centers_dict.get(image_name)
            if isinstance(self.centers_dict, dict)
            else self.centers_dict
        )

        if centers_for_image is None:
            raise KeyError(f"Centers not found for image {image_name}")

        centers_tensor = torch.tensor(centers_for_image)

        return img, target, image_name, centers_tensor

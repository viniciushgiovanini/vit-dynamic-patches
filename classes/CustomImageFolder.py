from torchvision.datasets import ImageFolder
import os


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        path = self.imgs[index][0]

        image_name = os.path.basename(path)

        return img, target, image_name 
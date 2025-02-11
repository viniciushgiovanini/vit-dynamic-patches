import matplotlib.pyplot as plt
import string
import random
import matplotlib.patches as patches
import numpy as np
import torch


class PatchVisualizer:

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def gerar_string_aleatoria(self, tamanho):
        caracteres = string.ascii_letters + string.digits
        return ''.join(random.choice(caracteres) for _ in range(tamanho))

    def denormalize(self, image_tensor, mean, std):
        mean = torch.tensor(mean).view(1, 1, 3)
        std = torch.tensor(std).view(1, 1, 3)

        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * std.numpy() + mean.numpy())

        image_np = np.clip(image_np, 0, 1)

        return image_np

    def visualize_patches_with_tensor(self, patch_array, cols=14, rows=14):

        num_images = len(patch_array)
        # print(num_images)
        # print("-----------------\n\n\n\n\n\n\n\n\n\n\n\n")

        if rows * cols < num_images:
            rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

        if rows > 1:
            axes = axes.flatten()

        for i, tensor in enumerate(patch_array):
            tensor_np = tensor.cpu().numpy()

            if tensor_np.shape[0] == 3:
                tensor_np = tensor_np.transpose(1, 2, 0)

            axes[i].imshow(tensor_np)
            axes[i].axis('off')

        for j in range(num_images, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

        plt.savefig("figs/patches_extraidos.png",
                    bbox_inches='tight', pad_inches=0)
        plt.show()

        plt.close(fig)

    def visualize_patch_centers(self, image, centers, patch_size, img_name):
        image_np = image.permute(1, 2, 0).cpu().numpy()

        image_np = self.denormalize(image, mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])

        fig, ax = plt.subplots(1)
        ax.imshow(image_np)

        patch_height, patch_width = patch_size

        for (h, w) in centers:
            rect = patches.Rectangle(
                (w - patch_width // 2, h - patch_height // 2),
                patch_width, patch_height,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            plt.plot(w, h, marker='v', color='r')

        ax.set_xlim([0, image_np.shape[1]])
        ax.set_ylim([image_np.shape[0], 0])
        plt.axis('off')
        # plt.savefig(f"figs/log_img/{self.gerar_string_aleatoria(10)}.jpg", bbox_inches='tight', pad_inches=0)
        plt.savefig(f"figs/log_img/{img_name}",
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

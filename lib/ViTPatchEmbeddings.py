import pickle
from operator import itemgetter

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel

from lib.dynamic_patches import DynamicPatches
from lib.patch_visualizer import PatchVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomVITPatchEmbeddings(nn.Module):
    def __init__(self, model_data: dict):
        super(CustomVITPatchEmbeddings, self).__init__()

        (
            input_size,
            patch_size,
            num_patches,
            embed_dim,
            is_visualizer,
            projection_type,
            abordagem_selecionada,
        ) = itemgetter(
            "input_size",
            "patch_size",
            "num_patches",
            "embed_dim",
            "is_visualizer",
            "projection_type",
            "abordagem_selecionada",
        )(
            model_data
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.is_visualizer = is_visualizer
        self.abordagem_selecionada = abordagem_selecionada
        self.patch_generator = DynamicPatches()

        if projection_type == "linear":
            self.projection = nn.Linear(
                patch_size[0] * patch_size[1] * input_size[0], embed_dim
            )
            self.patch_function = projection_type
        elif projection_type == "conv":
            self.projection = nn.Conv2d(
                in_channels=input_size[0],
                out_channels=embed_dim,
                kernel_size=patch_size[0],
                stride=patch_size[0],
            )
            self.patch_function = projection_type

        if is_visualizer:
            self.visualizer = PatchVisualizer(patch_size)

    def forward(self, x, **kwargs):
        img_name, centers_images = getattr(self, "current_centers_image", None)
        centers_images = centers_images.tolist()

        if isinstance(self.projection, nn.Linear):
            return self.default_extract(x, img_name, centers_images)
        else:
            return self.convolutional_strategy(x, img_name, centers_images)

    def convolutional_strategy(self, x, img_name: dict, centers: list):
        batch_size, channels, height, width = x.size()

        each_image = {}

        for b in range(batch_size):

            if self.abordagem_selecionada == "grid":
                patch_centers = self.patch_generator.generate_patch_centers(
                    height, width, self.patch_size
                )
            elif self.abordagem_selecionada == "sr":
                patch_centers = (
                    self.patch_generator.generate_random_patch_centers(
                        height, width, self.patch_size, self.num_patches
                    )
                )
            else:
                patch_centers = centers[b]

            h_indices = [int(h) for h, _ in patch_centers]
            w_indices = [int(w) for _, w in patch_centers]

            patches = []

            for h_idx, w_idx in zip(h_indices, w_indices):

                start_h = h_idx - self.patch_size[0] // 2
                start_w = w_idx - self.patch_size[1] // 2

                end_h = start_h + self.patch_size[0]
                end_w = start_w + self.patch_size[1]

                if (
                    0 <= start_h
                    and start_h + self.patch_size[0] <= height
                    and 0 <= start_w
                    and start_w + self.patch_size[1] <= width
                ):

                    patch = x[b, :, start_h:end_h, start_w:end_w]
                    patches.append(patch)

                else:
                    print(
                        f"Patch fora dos limites: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}"
                    )

            patches_tensor = torch.stack(patches)
            embeddings = self.projection(patches_tensor)
            each_image[img_name[b]] = embeddings.view(embeddings.size(0), -1)

        all_images = torch.stack(list(each_image.values()))
        return all_images

    def default_extract(self, x, img_name: dict, centers: list):
        # X -> Tensor de entrada (batch_size, channels, height, width)

        batch_size, channels, height, width = x.size()

        all_patches = []

        ################################################
        #                   Print de Log               #
        ################################################
        # print("Iniciou um loop de batch\n")
        # print(f"Printando de dentro do CustomPatchEmbedding: {image_names_dict}")

        for b in range(batch_size):

            if self.abordagem_selecionada == "grid":
                patch_centers = DynamicPatches().generate_patch_centers(
                    height, width, self.patch_size
                )
            elif self.abordagem_selecionada == "sr":
                patch_centers = DynamicPatches().generate_random_patch_centers(
                    height, width, self.patch_size, self.num_patches
                )
            else:
                patch_centers = centers[b]

            patches = []
            h_indices = [int(h) for h, _ in patch_centers]
            w_indices = [int(w) for _, w in patch_centers]

            for h_idx, w_idx in zip(h_indices, w_indices):

                start_h = h_idx - self.patch_size[0] // 2
                start_w = w_idx - self.patch_size[1] // 2

                end_h = start_h + self.patch_size[0]
                end_w = start_w + self.patch_size[1]

                if (
                    0 <= start_h
                    and start_h + self.patch_size[0] <= height
                    and 0 <= start_w
                    and start_w + self.patch_size[1] <= width
                ):

                    patch = x[b, :, start_h:end_h, start_w:end_w]
                    patches.append(patch)
                else:
                    print(
                        f"Patch fora dos limites: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}"
                    )

            if len(patches) < self.num_patches:
                print("ERRO: Gerando patch preto\n\n\n\n\n\n\n\n\n")
                missing_patches = self.num_patches - len(patches)
                patches += [
                    torch.zeros(
                        channels,
                        self.patch_size[0],
                        self.patch_size[1],
                        device=device,
                    )
                ] * missing_patches

            ##################################
            # Visualização do Patch Tensor
            ##################################
            # if self.is_visualizer:
            # self.visualizer.visualize_patches_with_tensor(patches)

            # self.visualizer.visualize_patch_centers(
            #     x[b], centers, self.patch_size, image_names_dict[b])

            patches = torch.stack(patches)

            patches = patches.flatten(start_dim=1)

            patches = self.projection(patches)

            all_patches.append(patches)

        all_patches = torch.stack(all_patches).to(device)

        return all_patches

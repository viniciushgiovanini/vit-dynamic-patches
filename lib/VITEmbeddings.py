import math
from operator import itemgetter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomViTEmbeddings(nn.Module):
    def __init__(
        self,
        config: ViTConfig,
        embeddings_dict: dict,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()

        CustomViTPatchEmbeddings, DefaultEmbeddings = itemgetter(
            "CustomPatchEmbeddings", "DefaultEmbeddings"
        )(embeddings_dict)

        self.cls_token = DefaultEmbeddings.cls_token

        self.mask_token = DefaultEmbeddings.mask_token

        self.patch_embeddings = CustomViTPatchEmbeddings

        self.position_embeddings = DefaultEmbeddings.position_embeddings

        self.interpolate_pos_encoding = (
            DefaultEmbeddings.interpolate_pos_encoding
        )

        self.dropout = DefaultEmbeddings.dropout

        self.config = config

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:

        _, centers_images = getattr(self, "current_centers_image_PE", None)

        batch_size, _1, height, width = pixel_values.shape

        embeddings = self.patch_embeddings(pixel_values)

        if bool_masked_pos is not None and self.mask_token is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings,
                height,
                width,
            )
        else:
            interpolated_pos = self.my_interpolate(
                self.position_embeddings, centers_images
            )

            embeddings = embeddings + interpolated_pos

        embeddings = self.dropout(embeddings)
        return embeddings

    def reorder_position_embbeddings(
        self,
        model,
        centers_with_idx: list,
        device=None,
    ):
        pos_embed = self.position_embeddings.data.clone()
        cls_pos_embed = pos_embed[:, 0:1, :]
        patch_pos_embed = pos_embed[:, 1:, :]

        ordered_indices = [idx for idx, _ in centers_with_idx]

        indices_tensor = torch.tensor(
            ordered_indices,
            dtype=torch.long,
            device=patch_pos_embed.device if device is None else device,
        )

        reordered_patches = patch_pos_embed[:, indices_tensor, :]

        new_pos_embed = torch.cat([cls_pos_embed, reordered_patches], dim=1)

        self.position_embeddings.data.copy_(new_pos_embed)

        return model

    def my_interpolate(self, pos_embeddings, centers, image_size=(224, 224)):
        H_img, W_img = image_size
        cls_token = pos_embeddings[:, 0:1, :]
        pos_only = pos_embeddings[:, 1:, :]

        B = centers.shape[0]
        D = pos_only.shape[-1]
        num_patches = pos_only.shape[1]

        h = w = int(math.sqrt(num_patches))
        assert (
            h * w == num_patches
        ), "pos_embeddings deve corresponder a um grid quadrado"

        pos_grid = pos_only.reshape(1, h, w, D).permute(0, 3, 1, 2)
        pos_grid = pos_grid.expand(B, -1, -1, -1)

        centers = centers.to(dtype=pos_grid.dtype, device=pos_grid.device)
        centers_grid = self.pixels_to_grid_coords(centers, H_img, W_img, h, w)

        x_grid = centers_grid[..., 0]
        y_grid = centers_grid[..., 1]

        x_norm = (x_grid / (w - 1)) * 2.0 - 1.0
        y_norm = (y_grid / (h - 1)) * 2.0 - 1.0

        centers_norm = torch.stack([x_norm, y_norm], dim=-1)

        grid = centers_norm.unsqueeze(1)
        grid = grid.to(pos_grid.device)

        interpolated = F.grid_sample(
            pos_grid, grid, align_corners=True, mode="bilinear"
        )

        interpolated = interpolated.squeeze(2).permute(0, 2, 1)

        cls_expand = cls_token.expand(B, -1, -1)
        interpolated_with_cls = torch.cat([cls_expand, interpolated], dim=1)

        return interpolated_with_cls

    def pixels_to_grid_coords(self, centers, H_img, W_img, h_grid, w_grid):
        centers = centers.to(dtype=torch.float32)
        x_pix = centers[..., 0]
        y_pix = centers[..., 1]

        x_grid = x_pix * ((w_grid - 1) / float(W_img - 1))
        y_grid = y_pix * ((h_grid - 1) / float(H_img - 1))

        return torch.stack([x_grid, y_grid], dim=-1)

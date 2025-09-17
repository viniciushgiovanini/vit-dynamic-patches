import math
from operator import itemgetter
from typing import Optional

import torch
import torch.nn as nn
from transformers import ViTConfig


class CustomViTEmbeddings(nn.Module):
    def __init__(
        self,
        config: ViTConfig,
        embeddings_dict: dict,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()

        CustomViTEmbeddings, DefaultEmbeddings = itemgetter(
            "CustomViTEmbeddings", "DefaultEmbeddings"
        )(embeddings_dict)

        self.cls_token = DefaultEmbeddings.cls_token

        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            if use_mask_token
            else None
        )

        self.patch_embeddings = CustomViTEmbeddings

        self.position_embeddings = DefaultEmbeddings.position_embeddings

        self.dropout = DefaultEmbeddings.dropout

        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(
                h0 / math.sqrt(num_positions),
                w0 / math.sqrt(num_positions),
            ),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

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
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings

    def reoorder_position_embbeddings(
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

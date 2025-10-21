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
                height, width, embeddings
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

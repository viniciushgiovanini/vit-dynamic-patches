import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import ViTConfig


class ViTSelfAttentionCustom(nn.Module):
    def __init__(self, config: ViTConfig, defaultSelfAttention: tuple):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = (
            config.hidden_size // config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        defaultSelfAttention = defaultSelfAttention[1]
        self.query = defaultSelfAttention.query
        self.key = defaultSelfAttention.key
        self.value = defaultSelfAttention.value
        self.dropout = defaultSelfAttention.dropout

        self.temperature = nn.Parameter(torch.ones(1))

        self.mask_value = -1e9

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # [B, H, N, D]

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )

        temp = torch.clamp(self.temperature, min=1e-4)
        attention_scores = attention_scores / (
            math.sqrt(self.attention_head_size) * temp
        )

        n_tokens = attention_scores.size(-1)
        eye = torch.eye(
            n_tokens, device=attention_scores.device, dtype=torch.bool
        )
        attention_scores = attention_scores.masked_fill(
            eye.unsqueeze(0).unsqueeze(0), self.mask_value
        )

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size(0),
            context_layer.size(1),
            self.all_head_size,
        )

        outputs = (
            (context_layer, attention_probs)
            if output_attentions
            else (context_layer,)
        )
        return outputs

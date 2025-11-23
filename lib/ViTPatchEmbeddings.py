from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.is_visualizer = is_visualizer
        self.abordagem_selecionada = abordagem_selecionada

        # self.shift_pixels = 1
        # self.spt = SPTBlock(input_size[0], shift_k=8, mode="8way")
        # self.spt = SPTPreConv(input_size[0], shift_k=1, mode="8way")

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

        # x = self.shifted_patch_tokenization(x, shift_pixels=self.shift_pixels)
        # x = self.spt(x)

        return self.patch_extract(x, img_name=img_name, centers=centers_images)

    def _extract_patches_from_centers(self, x: torch.Tensor, centers_list):
        B, C, H, W = x.shape
        ph, pw = self.patch_size

        N = self.num_patches
        centers_tensor = torch.zeros(
            (B, N, 2), dtype=torch.long, device=x.device
        )

        for b in range(B):
            centers_b = centers_list[b]
            chosen = centers_b[:N]
            centers_tensor[b] = torch.tensor(
                chosen, dtype=torch.long, device=x.device
            )

        centers_h = centers_tensor[..., 0]
        centers_w = centers_tensor[..., 1]

        dh = torch.arange(ph, device=x.device, dtype=torch.long) - (ph // 2)
        dw = torch.arange(pw, device=x.device, dtype=torch.long) - (pw // 2)

        grid_h = centers_h.unsqueeze(-1) + dh.unsqueeze(0).unsqueeze(0)
        grid_w = centers_w.unsqueeze(-1) + dw.unsqueeze(0).unsqueeze(0)

        grid_h = grid_h.clamp(0, H - 1)
        grid_w = grid_w.clamp(0, W - 1)

        grid_h_exp = grid_h.unsqueeze(-1).expand(-1, -1, -1, pw)
        grid_w_exp = grid_w.unsqueeze(-2).expand(-1, -1, ph, -1)

        linear_idx = (grid_h_exp * W + grid_w_exp).view(B, N * ph * pw)

        x_flat = x.view(B, C, H * W)

        idx_exp = linear_idx.unsqueeze(1).expand(-1, C, -1)

        patches_flat = torch.gather(x_flat, 2, idx_exp)

        patches = patches_flat.view(B, C, N, ph, pw)

        patches = patches.permute(0, 2, 1, 3, 4).contiguous()

        return patches

    def patch_extract(self, x, img_name: dict, centers: list):
        B, C, H, W = x.size()

        if centers is None:
            patch_centers = self.patch_generator.generate_patch_centers(
                H, W, self.patch_size
            )
            centers = [patch_centers for _ in range(B)]

        patches = self._extract_patches_from_centers(x, centers)
        B, N, C, ph, pw = patches.shape

        patches_conv_in = patches.view(B * N, C, ph, pw)

        if isinstance(self.projection, nn.Conv2d):
            emb = self.projection(patches_conv_in)
            emb = emb.view(B, N, -1)
        else:
            patches_lin = patches_conv_in.view(B * N, -1)
            emb = self.projection(patches_lin)
            emb = emb.view(B, N, -1)

        if img_name is None:
            return emb

        if isinstance(img_name, str):
            img_name = [img_name]

        each_image = {}

        B = emb.shape[0]
        if len(img_name) != B:
            raise ValueError(
                f"Inconsistência: img_name tem {len(img_name)} itens, "
                f"mas emb tem batch size {B}"
            )

        for i, name in enumerate(img_name):
            each_image[name] = emb[i]

        return torch.stack([each_image[n] for n in img_name])


class SPTPreConv(nn.Module):
    def __init__(self, in_ch, shift_k=1, mode="4way"):
        super().__init__()
        self.spt = SPTBlock(in_ch, shift_k=shift_k, mode=mode)

        C_total = in_ch * len(self.spt.shift_offsets)

        self.compress = nn.Conv2d(C_total, in_ch, kernel_size=1)

    def forward(self, x):
        x_spt = self.spt.forward(x)
        return self.compress(x_spt)


class SPTBlock(nn.Module):
    def __init__(self, in_ch, shift_k=1, mode="4way"):
        super().__init__()
        self.k = shift_k

        if mode == "4way":
            self.shift_offsets = [
                (0, 0),
                (-self.k, 0),
                (self.k, 0),
                (0, -self.k),
                (0, self.k),
            ]

        elif mode == "8way":
            self.shift_offsets = [
                (0, 0),
                (-self.k, 0),
                (self.k, 0),
                (0, -self.k),
                (0, self.k),
                (-self.k, -self.k),
                (-self.k, self.k),
                (self.k, -self.k),
                (self.k, self.k),
            ]
        else:
            raise ValueError("Mode inválido")

        self.num_shifts = len(self.shift_offsets)
        in_ch_total = in_ch * self.num_shifts

        self.dw = nn.Conv2d(
            in_ch_total,
            in_ch_total,
            kernel_size=3,
            padding=1,
            groups=in_ch_total,
            bias=False,
        )

        self.pw = nn.Conv2d(in_ch_total, in_ch, kernel_size=1, bias=False)

    def shift_by(self, x, dy, dx):
        B, C, H, W = x.shape

        pad_left = max(dx, 0)
        pad_right = max(-dx, 0)
        pad_top = max(dy, 0)
        pad_bottom = max(-dy, 0)

        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return x_pad[:, :, pad_top : pad_top + H, pad_left : pad_left + W]

    def forward(self, x):
        shifted = [self.shift_by(x, dy, dx) for (dy, dx) in self.shift_offsets]

        x_cat = torch.cat(shifted, dim=1)

        y = self.pw(self.dw(x_cat))

        return x + y

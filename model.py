"""
This is Vision Transformer (ViT) – built from scratch.
"""
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Type, Union

from einops import rearrange
from einops.layers.torch import Rearrange

ImgSize = Union[int, Tuple[int, int]]

def _hw(img_size: ImgSize) -> Tuple[int, int]:
    return img_size, img_size if isinstance(img_size, int) else tuple(img_size)

class Patchifier(nn.Module):
    def __init__(self, img_size: ImgSize, patch_size:int, c_in:int, dim:int):
        super().__init__()
        H, W = _hw(img_size=img_size)
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.dim = dim
        self.net = nn.Sequential(
            # Convolution to create patches
            nn.Conv2d(c_in, dim, patch_size, patch_size),
            # rearrange
            Rearrange('b d h w -> b (h w) d')
        )

    def forward(self, x:torch.tensor) -> torch.Tensor:
        return self.net(x)

class InputEmbedding(nn.Module):
    def __init__(self, d_model, data_size):
        super().__init__()
        self.d_model = d_model
        self.data_size =data_size
        self.embedding = nn.Embedding(data_size, d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class RoPE_2D(nn.Module):
    def __init__(self, head_dim:int, grid_height, grid_width):
        super().__init__()
        assert head_dim % 4 == 0 , "head_dim must be divisible by 4"
        self.head_dim = head_dim
        self.grid_h = grid_height
        self.grid_w = grid_width

        half_dim = head_dim // 2

        freq = 1.0 / (
            10000 ** (
                torch.arange(0, half_dim, 2).float()
            )
        )
        row_pos = torch.arange(self.grid_h).float()
        col_pos = torch.arange(self.grid_w).float()

        row_theta = torch.outer(row_pos, freq)
        col_theta = torch.outer(col_pos, freq)

        self.register_buffer("row_cos", torch.cos(row_theta))
        self.register_buffer("row_sin", torch.sin(row_theta))

        self.register_buffer("col_cos", torch.cos(col_theta))
        self.register_buffer("col_sin", torch.sin(col_theta))

    def rotate(self, x, cos, sin):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.stack(
            (rotated_x1, rotated_x2),
            dim = -1
        ).flatten(-2)

    def forward(self, q, k):
        # q/k shape:
        # (batch, heads, num_patches, head_dim)

        B, H, N, D = q.shape

        assert N == self.grid_h + self.grid_w

        positions = torch.arange(N, device=q.device)

        rows = positions // self.grid_w
        cols = positions % self.grid_h

        # splitting the q/k into row and column parts
        q_row, q_col = q[..., : D // 2], q[..., D // 2:]
        k_row, k_col = k[..., : D // 2], k[..., D // 2:]

        # get sin/cos corresponding to each patch's row and column
        row_cos = self.row_cos[rows]
        row_sin = self.row_sin[rows]

        col_cos = self.col_cos[cols]
        col_sin = self.col_sin[cols]

        # Apply RoPE

        q_row = self.rotate(q_row, row_cos, row_sin)
        k_row = self.rotate(k_row, row_cos, row_sin)

        q_col = self.rotate(q_col, col_cos, col_sin)
        k_col = self.rotate(k_col, col_cos, col_sin)

        q = torch.cat((q_row, q_col), dim=-1)
        k = torch.cat((k_row, k_col), dim=-1)

        return q, k
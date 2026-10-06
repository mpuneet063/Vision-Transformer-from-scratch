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
    def __init__(self, img_size: ImgSize, patch_size:int, c_in:int, dim:int) -> None:
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
    def __init__(self, d_model, data_size) -> None:
        super().__init__()
        self.d_model = d_model
        self.data_size =data_size
        self.embedding = nn.Embedding(data_size, d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class RoPE_2D(nn.Module):
    def __init__(self, head_dim:int, grid_height, grid_width) -> None:
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

class RMSNorm(nn.Module):
    def __init__(self, epsilon:float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.epsilon = epsilon

    def rms(self, x:torch.Tensor):
        mean = torch.mean(x**2, dim=-1, keepdim=True)
        return torch.sqrt(mean + self.epsilon)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        rms = self.rms(x)
        return self.gamma * x / rms

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert self.d_model % self.num_heads == 0, "Model dimension must be divisible by the number of heads"
        self.d_k = self.d_model // self.num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # output layer
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        if dropout is not None:
            attn_probs = dropout(attn_probs)
        out = attn_probs @ value
        return out, attn_probs

    def forward(self, q, k, v, mask=None) -> torch.Tensor:
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # split q, k, v into multiple heads 
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)

        x, self.attn = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)
        return self.w_o(x)

class ResidualConnectionBlock(nn.Module):
    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm()

    def forward(self, x:torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.attn_block = self_attention_block
        self.ff_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(2)])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.attn_block(x,x,x))
        x = self.residual_connections[1](x, self.ff_block)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderBlock() for _ in range(num_layers))
        self.norm = RMSNorm()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attn_block = self_attention_block
        self.cross_attn_block = cross_attention_block
        self.ff_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(3)])

    def forward(self, x:torch.Tensor, encoder_output) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attn_block(x,x,x))
        x = self.residual_connections[1](x, lambda x: self.cross_attn_block(x, encoder_output, encoder_output))
        x = self.residual_connections[2](x, self.ff_block)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(DecoderBlock() for _ in range(num_layers))
        self.norm = RMSNorm()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, ):
        super().__init__()
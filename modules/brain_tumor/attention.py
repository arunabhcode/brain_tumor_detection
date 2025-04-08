#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from logger import print  # Import custom print


class SelfAttention(torch.nn.Module):
    """
    Class for a single self-attention head.
    Input: (B, seq_length, head_dim)
    Output: (B, seq_length, head_dim)
    """

    def __init__(self, head_dim=64):  # Example: 768 embed_dim / 12 heads = 64
        super(SelfAttention, self).__init__()
        self.head_dim = head_dim
        # Linear projections for Q, K, V
        self.W_q = torch.nn.Linear(
            head_dim, head_dim, bias=False
        )  # Bias often False in attention
        self.W_k = torch.nn.Linear(head_dim, head_dim, bias=False)
        self.W_v = torch.nn.Linear(head_dim, head_dim, bias=False)

    def forward(self, x):
        # x shape: (B, seq_length, head_dim)
        batch_size, seq_length, _ = x.shape

        # Project to Q, K, V
        q = self.W_q(x)  # (B, seq_length, head_dim)
        k = self.W_k(x)  # (B, seq_length, head_dim)
        v = self.W_v(x)  # (B, seq_length, head_dim)

        # Calculate attention scores (scaled dot-product)
        # (B, seq_length, head_dim) @ (B, head_dim, seq_length) -> (B, seq_length, seq_length)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(
            attn_scores, dim=-1
        )  # (B, seq_length, seq_length)

        # Apply attention weights to V
        # (B, seq_length, seq_length) @ (B, seq_length, head_dim) -> (B, seq_length, head_dim)
        attention_output = torch.matmul(attention_weights, v)

        return attention_output


class MultiHeadAttention(torch.nn.Module):
    """
    Class for the multi-head attention layer.
    Input: (B, seq_length, embed_dim)
    Output: (B, seq_length, embed_dim)
    """

    def __init__(self, embed_dim=768, num_heads=12):
        super(MultiHeadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combine Q, K, V projections for efficiency
        self.qkv_proj = torch.nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # Output projection
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        # --- Original Approach (using ModuleList of SelfAttention) --- #
        # Kept for reference, but the combined projection is more standard.
        # self.self_attention_layers = torch.nn.ModuleList(
        #     [SelfAttention(self.head_dim) for _ in range(num_heads)]
        # )
        # self.W_o = torch.nn.Linear(embed_dim, embed_dim)
        # --- End Original Approach --- #

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        batch_size, seq_length, _ = x.shape

        # --- Combined QKV Projection Approach --- #
        qkv = self.qkv_proj(x)  # (B, seq_length, embed_dim * 3)

        # Split into Q, K, V for each head
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (B, num_heads, seq_length, 3 * head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each is (B, num_heads, seq_length, head_dim)

        # Calculate attention scores (scaled dot-product)
        # (B, num_heads, seq_length, head_dim) @ (B, num_heads, head_dim, seq_length) -> (B, num_heads, seq_length, seq_length)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(
            attn_scores, dim=-1
        )  # (B, num_heads, seq_length, seq_length)

        # Apply attention weights to V
        # (B, num_heads, seq_length, seq_length) @ (B, num_heads, seq_length, head_dim) -> (B, num_heads, seq_length, head_dim)
        attention_output = torch.matmul(attention_weights, v)

        # Reshape and combine heads
        attention_output = attention_output.permute(
            0, 2, 1, 3
        )  # (B, seq_length, num_heads, head_dim)
        attention_output = attention_output.reshape(
            batch_size, seq_length, self.embed_dim
        )  # (B, seq_length, embed_dim)

        # Apply output projection
        output = self.out_proj(attention_output)
        # --- End Combined QKV Projection Approach --- #

        # --- Original Approach Forward Pass --- #
        # head_outputs = []
        # x_split = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        # for i, layer in enumerate(self.self_attention_layers):
        #     head_i = x_split[:, :, i, :] # Get i-th head (B, seq_length, head_dim)
        #     head_output = layer(head_i)  # Apply attention (B, seq_length, head_dim)
        #     head_outputs.append(head_output)
        # # Concatenate all head outputs -> (B, seq_length, embed_dim)
        # concat_heads = torch.cat(head_outputs, dim=-1)
        # # Apply output projection
        # output = self.W_o(concat_heads)
        # --- End Original Approach Forward Pass --- #

        return output

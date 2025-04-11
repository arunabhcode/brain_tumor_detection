#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from mlp import FeedForward
from attention import MultiHeadAttentionQKVProjection


class Encoder(torch.nn.Module):
    """
    Standard encoder class for a transformer
    """

    def __init__(self, embed_dim, num_heads, dropout):
        super(Encoder, self).__init__()

        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.multi_head_attention = MultiHeadAttentionQKVProjection(
            embed_dim, num_heads
        )
        self.dropout = torch.nn.Dropout(dropout)  # Added dropout after MHA and MLP
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, dropout=dropout)

    def forward(self, tokens):
        # MHA + Dropout + Add & Norm
        attn_input = self.layer_norm1(tokens)
        attn_output = self.multi_head_attention(
            attn_input
        )  # (B, num_patches, embed_dim)
        attn_output = self.dropout(attn_output)
        x = tokens + attn_output  # Residual connection

        # FeedForward + Dropout + Add & Norm
        ff_input = self.layer_norm2(x)
        ff_output = self.feed_forward(ff_input)  # (B, num_patches, embed_dim)
        # Note: Dropout is applied inside FeedForward in the provided mlp.py
        encoded_output = (
            x + ff_output
        )  # Residual connection (B, num_patches, embed_dim)
        return encoded_output

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from logger import print  # Import custom print


class PatchEmbeddingFull(torch.nn.Module):
    """
    Class for the patch embedding layer using Conv2d.
    Input: (B, C, H, W)
    Output: (B, num_patches, embed_dim)
    """

    def __init__(self, in_channels=3, out_channels=768, patch_size=(16, 16)):
        super(PatchEmbeddingFull, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        # Define the convolutional layer for patch embedding
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,  # Assuming no padding needed
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.conv(x)  # Output shape: (B, embed_dim, H/patch_h, W/patch_w)
        # Flatten the spatial dimensions and transpose
        x = x.flatten(2)  # Output shape: (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Output shape: (B, num_patches, embed_dim)
        return x


class PatchEmbedding(torch.nn.Module):
    """
    Class for the patch embedding layer using a Linear layer.
    Assumes input is already flattened per patch.
    Input: (B, num_patches, patch_h * patch_w * C)
    Output: (B, num_patches, embed_dim)
    """

    def __init__(self, in_channels=16 * 16 * 3, out_channels=768):
        super(PatchEmbedding, self).__init__()
        # Define the linear layer for patch embedding
        self.embed_matrix = torch.nn.Linear(
            in_channels,
            out_channels,
        )

    def forward(self, x):
        # x shape: (B, num_patches, patch_h * patch_w * C)
        # print(x.shape)
        x = self.embed_matrix(x)
        # Output shape: (B, num_patches, embed_dim)
        return x


class PositionEmbedding(torch.nn.Module):
    """
    Class for the sinusoidal position embedding layer.
    Adds positional embeddings to the input tensor.
    Input: (B, seq_length, embed_dim)
    Output: (B, seq_length, embed_dim)
    """

    def __init__(
        self, seq_length=197, embed_dim=768
    ):  # Default seq_length assumes 196 patches + 1 CLS token
        super(PositionEmbedding, self).__init__()

        pe = torch.zeros(seq_length, embed_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_length, embed_dim)

        # Register the positional embedding matrix as a non-trainable buffer
        self.register_buffer("pe", pe)
        # If you need it to be trainable (like in original ViT paper for learned embeddings):
        # self.pe = torch.nn.Parameter(pe, requires_grad=True)
        # Note: The original implementation in train.py used Parameter with requires_grad=False,
        # which is equivalent to register_buffer unless assigned later.
        # Using register_buffer is clearer for non-trainable parameters.

    def forward(self, x):
        # x shape: (B, seq_length, embed_dim)
        # Add positional embedding. We slice `pe` in case `x`'s seq_length is shorter
        # (e.g., during inference with different image sizes, though not typical for ViT)
        # or if the PositionEmbedding was initialized with a larger seq_length.
        x = x + self.pe[:, : x.size(1)]
        return x

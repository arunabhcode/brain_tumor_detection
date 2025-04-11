#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from embedding import PatchEmbedding, PositionEmbedding
from encoder import Encoder


class ViTEncoderMean(torch.nn.Module):
    """
    Class for a simplified Vision Transformer Encoder model with a classification head.
    Uses MEAN POOLING of output tokens for classification.
    Input: Assumes input x is already patch-embedded and flattened (B, num_patches, patch_input_dim)
           where patch_input_dim = patch_h * patch_w * C
    Output: Logits (B, 1) for binary classification.
    """

    def __init__(
        self,
        num_patches=196,  # e.g., (224/16) * (224/16)
        patch_input_dim=768,  # e.g., 16 * 16 * 3
        embed_dim=768,
        num_heads=12,
        num_classes=1,  # Outputting 1 logit for BCEWithLogitsLoss
        dropout=0.1,  # Added dropout parameter
    ):
        super(ViTEncoderMean, self).__init__()
        self.seq_length = num_patches  # Sequence length is number of patches

        # Embedding layer (Linear projection for already flattened patches)
        self.patch_embedding = PatchEmbedding(
            in_channels=patch_input_dim, out_channels=embed_dim
        )
        # Positional embedding (Sinusoidal)
        self.position_embedding = PositionEmbedding(self.seq_length, embed_dim)

        self.transformer_encoder = Encoder(embed_dim, num_heads, dropout)

        # Classification head (using mean pooling)
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

        self._init_weights()  # Add weight initialization

    def _init_weights(self):
        # Simple initialization: normal for linear weights, zeros for biases
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (batch_size, num_patches, patch_input_dim)
        # 1. Patch + Position Embedding
        embedded_x = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        # Note: PositionEmbedding expects (B, seq_len, embed_dim)
        # Need to ensure seq_len matches num_patches here.
        tokens = self.position_embedding(embedded_x)  # (B, num_patches, embed_dim)

        # Call the transformer encoder on the tokens
        encoded_output = self.transformer_encoder(tokens)

        # 3. Classification Head (Mean Pooling)
        mean_output = encoded_output.mean(dim=1)  # (B, embed_dim)
        logits = self.classifier(mean_output)  # (B, num_classes)
        return logits


class ViTEncoderCLS(torch.nn.Module):
    """
    Class for the Vision Transformer Encoder model with a classification head.
    Uses the standard [CLS] token approach for classification.
    Input: Image tensor (B, C, H, W).
    Output: Logits (B, num_classes).
    """

    def __init__(
        self,
        num_patches=196,  # e.g., (224/16) * (224/16)
        img_size=(224, 224),
        patch_size=(16, 16),
        in_channels=3,
        embed_dim=768,
        num_heads=12,
        num_classes=2,  # Original had 2 classes
        dropout=0.1,
    ):
        super(ViTEncoderCLS, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Calculate number of patches
        self.num_patches = num_patches
        self.seq_length = self.num_patches + 1  # Add 1 for CLS token

        # Patch embedding (using Conv2D - PatchEmbeddingFull)
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels * patch_size[0] * patch_size[1],
            out_channels=embed_dim,
        )

        # CLS token (trainable parameter)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (for CLS token + patches)
        # Using sinusoidal embedding from embedding.py
        self.position_embedding = PositionEmbedding(self.seq_length, embed_dim)

        # Transformer Encoder Block components
        self.transformer_encoder = Encoder(embed_dim, num_heads, dropout)

        # Classification head (takes the CLS token output)
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers, layernorm, and cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
        # Note: Conv2d in PatchEmbeddingFull is initialized by default PyTorch, which is Kaiming uniform.
        # You could add specific initialization for it if needed.

    def forward(self, x):
        # x shape: (batch_size, in_channels, img_height, img_width)
        batch_size = x.shape[0]

        # 1. Patch Embedding
        patches = self.patch_embedding(x)  # (B, num_patches, embed_dim)

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat(
            (cls_tokens, patches), dim=1
        )  # (B, num_patches+1, embed_dim)

        # 3. Add Positional Embedding
        tokens = self.position_embedding(tokens)  # (B, seq_length, embed_dim)

        # Call the transformer encoder on the tokens
        encoded_output = self.transformer_encoder(
            tokens
        )  # Residual connection (B, seq_length, embed_dim)

        # 5. Classification
        cls_output = encoded_output[:, 0]  # Extract the CLS token output (B, embed_dim)
        logits = self.classifier(cls_output)  # (B, num_classes)

        return logits

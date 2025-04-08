#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from logger import print  # Import custom print


class FeedForward(torch.nn.Module):
    """
    Class for the feed forward layer (MLP block) in a Transformer.
    Input: (B, seq_length, embed_dim)
    Output: (B, seq_length, embed_dim)
    """

    def __init__(self, embed_dim=768, mlp_dim=None, dropout=0.1):
        super(FeedForward, self).__init__()
        if mlp_dim is None:
            mlp_dim = embed_dim * 4  # Common practice

        self.fc1 = torch.nn.Linear(embed_dim, mlp_dim)
        # Consider using GELU activation, which is common in Transformers
        # self.activation = torch.nn.GELU()
        self.activation = torch.nn.ReLU()  # Using ReLU as in the original code
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, seq_length, embed_dim)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        # Output shape: (B, seq_length, embed_dim)
        return x

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch


# TODO: Separate conv block into it's own class/component


class ProgressiveDownsamplingCNNEncoder(torch.nn.Module):
    """
    CNN architecture that progressively halves the size of the feature map and
    doubles the number of feature maps
    """

    def __init__(
        self,
        input_channels=3,
        base_out_channels=16,
        kernel_size=3,
        blocks=3,
        downsample_factor=2,
    ):
        self.in_channels = input_channels
        self.out_channels = base_out_channels
        self.downsample_block = (
            lambda in_ch, out_ch, k_size, p_size: torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=(k_size, k_size),
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(p_size),
            )
        )
        conv_block_list = torch.nn.ModuleList()
        for i in range(blocks):
            conv_block_list.append(
                self.downsample_block(
                    self.in_channels, self.out_channels, kernel_size, downsample_factor
                )
            )
            self.in_channels = self.out_channels
            self.out_channels = downsample_factor * self.out_channels

    def forward(self, x):
        for layer in self.conv_block_list:
            x = layer(x)
        return x

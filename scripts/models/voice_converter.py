# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class VCConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size,
                 upsample=1, norm=True, glu=True, **kwargs):
        leyers = [nn.Conv1d(in_ch, out_ch, kernel_size, **kwargs)]
        if upsample > 1:
            layers.append(nn.PixelShuffle(upsample))
        if norm:
            layers.append(nn.InstanceNorm1d(out_ch))
        if glu:
            layers.append(F.glu)
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class VCResidualBlock(nn.Module):
    def __init__(self):
        self.layers = nn.Sequential(
            VCConv1d(512, 1024, 3, padding=1),
            VCConv1d(1024, 512, 3, padding=1, glu=False)
        )

    def forward(self, x):
        return x + self.layers(x)


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_ch, out_ch),
            nn.Relu()
        )

    def forward(self, x):
        return self.layers(x)


class VCGenerator(nn.Module):
    def __init__(self, n_channel=24, n_blocks=6):
        self.downsample = nn.Sequential(
            VCConv1d(n_channel, 128, 15, padding=7, norm=False),
            VCConv1d(128, 256, 5, stride=2, padding=2),
            VCConv1d(256, 512, 5, stride=2, padding=2)
        )
        blocks = [VCResidualBlock() for _ in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.upsample = nn.Sequential(
            VCConv1d(512, 1024, 5, padding=2, upsample=2),
            VCConv1d(256, 512, 5, padding=2, upsample=2),
            VCConv1d(128, n_channel, 15, padding=7, norm=False, glu=False)
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        x = self.upsample(x)
        return x


class VCDiscriminator():
    def __init__(self, n_blocks=4):
        self.conv = VCConv1d(1, 128, 3, stride=(1, 2), norm=False)
        self.downsample = nn.Sequential(
            VCConv1d(128, 256, 3, stride=2),
            VCConv1d(256, 512, 3, stride=2),
            VCConv1d(512, 1024, (6, 3), stride=(1, 2))
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        fc = [FullyConnectedLayer(1024 * 6 * 6, 4096)]
        for _ in range(n_blocks - 1):
            fc.append(FullyConnectedLayer(4096, 4096))
        self.fc = nn.Sequential(fc)

    def forward(self, x):
        x = self.conv(x)
        x = self.downsample(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return torch.tanh(x)

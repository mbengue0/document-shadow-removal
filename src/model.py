"""Compact U-Net for RGB image restoration (Ronneberger et al., 2015, adapted).

Differences from the original segmentation U-Net:
  - 3 output channels (RGB) with continuous values instead of class scores.
  - "Same" padding, so output size = input size (input sides must be divisible by 2**depth).
  - GroupNorm instead of BatchNorm: behaves the same in training and evaluation and with
    small batches, which matters for full-page tiled inference.
  - Residual output: the network predicts a correction that is added to the input
    (output = input + correction). Shadow removal mostly brightens some regions and leaves
    the rest unchanged, so learning the change is easier than redrawing the whole page.
    The last layer starts at zero, so before training the model returns the input unchanged.
    The same setting is used for every experimental condition.
"""
import torch
from torch import nn


def block(c_in, c_out, groups=8):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(min(groups, c_out), c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(min(groups, c_out), c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base=32, depth=4, residual=True):
        super().__init__()
        self.depth, self.residual = depth, residual
        ch = [base * 2 ** i for i in range(depth + 1)]  # e.g. 32, 64, 128, 256, 512
        self.down = nn.ModuleList([block(3, ch[0])] + [block(ch[i - 1], ch[i]) for i in range(1, depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.upconv = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth, 0, -1)])
        self.up = nn.ModuleList([block(ch[i - 1] * 2, ch[i - 1]) for i in range(depth, 0, -1)])
        self.head = nn.Conv2d(ch[0], 3, 1)
        if residual:  # start as the identity: output = input until the network learns a correction
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips, h = [], x
        for i, d in enumerate(self.down):
            h = d(h)
            if i < self.depth:
                skips.append(h)
                h = self.pool(h)
        for upc, up, skip in zip(self.upconv, self.up, reversed(skips)):
            h = up(torch.cat([upc(h), skip], dim=1))
        out = self.head(h)
        return x + out if self.residual else out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Training losses.

L_total = L1 + lambda_edge * L_edge

L1      mean absolute difference between prediction and clean target, over all pixels and
        RGB channels. Images are in [0, 1].
L_edge  mean absolute difference between the Sobel gradients of prediction and target,
        computed on luminance Y = 0.299 R + 0.587 G + 0.114 B.
        - Kernels are the standard 3x3 Sobel kernels divided by 8, so a step of height 1
          between flat regions gives a gradient of at most 1 (same scale as L1).
        - Replicate padding: image borders do not create artificial edges.
        - Horizontal (gx) and vertical (gy) gradients are compared separately and averaged:
          L_edge = mean(|gx_p - gx_t| + |gy_p - gy_t|) / 2. No square root, so it is smooth.
        All operations are differentiable; no thresholding is used.
"""
import torch
import torch.nn.functional as F
from torch import nn

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
LUMA = torch.tensor([0.299, 0.587, 0.114])


def luminance(x):
    return (x * LUMA.to(x).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def sobel(y):
    kx = SOBEL_X.to(y).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    y = F.pad(y, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(y, kx), F.conv2d(y, ky)


def edge_loss(pred, target):
    gxp, gyp = sobel(luminance(pred))
    gxt, gyt = sobel(luminance(target))
    return 0.5 * ((gxp - gxt).abs().mean() + (gyp - gyt).abs().mean())


class RestorationLoss(nn.Module):
    def __init__(self, lambda_edge=0.0):
        super().__init__()
        self.lambda_edge = float(lambda_edge)

    def forward(self, pred, target):
        l1 = (pred - target).abs().mean()
        le = edge_loss(pred, target)  # always computed and logged, even when its weight is 0
        total = l1 + self.lambda_edge * le
        return total, {"l1": l1.detach(), "edge": le.detach(), "total": total.detach()}

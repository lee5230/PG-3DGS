import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class SL1Loss(nn.Module):
    def __init__(self, ohem=False, topk=0.6):
        super(SL1Loss, self).__init__()
        self.ohem = ohem
        self.topk = topk
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, inputs, targets, mask):
        loss = self.loss(inputs[mask], targets[mask])

        if self.ohem:
            num_hard_samples = int(self.topk * loss.numel())
            loss, _ = torch.topk(loss.flatten(), 
                                 num_hard_samples)

        return torch.mean(loss)

def l1_loss(prediction, gt):
    return torch.abs((prediction - gt)).mean()

def l2_loss(prediction, gt):
    return ((prediction - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def scaling_length_penalty(scalings, threshold, margin=0.1, p=2, reduction='mean'):
    """
    Per-dimension penalty: penalize each component of `scalings` that exceeds `threshold`.
    - No penalty for |value| <= threshold.
    - For threshold < |value| < threshold + margin: smooth quadratic ramp (C1-continuous).
    - For |value| >= threshold + margin: linear growth (Huber-like).
    Args:
        scalings: Tensor of shape (N, D) (or (..., D)) containing scaling vectors.
        threshold: scalar (or broadcastable tensor) threshold (no penalty below).
        margin: width of smooth transition after threshold. If margin <= 0 falls back to ReLU.
        p: kept for API compatibility (not used for per-dimension penalty).
        reduction: 'mean', 'sum', or 'none' (per-element).
    Returns:
        Scalar loss (or per-element tensor if reduction == 'none').
    """
    # per-dimension exceedance on absolute values
    x = scalings.abs() - threshold  # positive values indicate exceedance

    if margin <= 0:
        per_element = F.relu(x)
    else:
        per_element = torch.zeros_like(x)
        in_between = (x > 0) & (x < margin)
        out = x >= margin

        if in_between.any():
            xi = x[in_between]
            per_element[in_between] = (xi * xi) / (2.0 * margin)
        if out.any():
            xo = x[out]
            per_element[out] = xo - (margin / 2.0)

    if reduction == 'mean':
        return per_element.mean()
    elif reduction == 'sum':
        return per_element.sum()
    elif reduction == 'none':
        return per_element
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")

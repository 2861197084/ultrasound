from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


def pixel_grid(height, width, *, stride=1.0, offset=0.0, device=None):
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                          torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    return torch.stack((x * stride + offset, y * stride + offset), dim=-1)


def sample_pixels(image, xy):
    """B,C,H,W and B,...,2 pixel-center xy -> B,...,C, align_corners=True."""
    b, c, h, w = image.shape
    shape = xy.shape[1:-1]
    scale = xy.new_tensor([max(w - 1, 1), max(h - 1, 1)])
    normalized = 2 * xy / scale - 1
    values = F.grid_sample(image.float(), normalized.reshape(b, 1, -1, 2).float(),
                           mode="bilinear", padding_mode="border", align_corners=True)
    return values[:, :, 0].transpose(1, 2).reshape(b, *shape, c)


def boundary_distances(mask):
    mask = np.asarray(mask, dtype=bool)
    boundary = mask & ~ndimage.binary_erosion(mask, border_value=0)
    if not boundary.any():
        raise ValueError("Empty foreground boundary")
    return ndimage.distance_transform_edt(~boundary).astype(np.float32)


def grouped_queries(mask, rng, groups=12, points=12, max_pixels=4096, iterations=10):
    """Spatial K-means followed by farthest-point sampling in each group."""
    yy, xx = np.where(mask)
    if len(xx) < groups * points:
        raise ValueError("Foreground too small for grouped queries")
    coords = np.column_stack((xx, yy)).astype(np.float32)
    if len(coords) > max_pixels:
        coords = coords[rng.choice(len(coords), max_pixels, replace=False)]
    centers = [coords[int(rng.integers(len(coords)))]]
    for _ in range(1, groups):
        d = np.min(np.sum((coords[:, None] - np.asarray(centers)[None]) ** 2, -1), 1)
        centers.append(coords[int(rng.choice(len(coords), p=d / d.sum()))])
    centers = np.asarray(centers)
    for _ in range(iterations):
        distances = np.sum((coords[:, None] - centers[None]) ** 2, -1)
        labels = distances.argmin(1)
        for g in range(groups):
            cluster = coords[labels == g]
            centers[g] = cluster.mean(0) if len(cluster) else coords[distances.min(1).argmax()]
    labels = np.sum((coords[:, None] - centers[None]) ** 2, -1).argmin(1)
    result = []
    for g in range(groups):
        cluster = coords[labels == g]
        if not len(cluster):
            cluster = coords[np.argsort(np.sum((coords - centers[g]) ** 2, -1))[:points]]
        chosen = [int(rng.integers(len(cluster)))]
        distance = np.sum((cluster - cluster[chosen[0]]) ** 2, -1)
        for _ in range(1, points):
            j = int(distance.argmax())
            chosen.append(j)
            distance = np.minimum(distance, np.sum((cluster - cluster[j]) ** 2, -1))
        result.append(cluster[chosen])
    return np.stack(result)


@torch.no_grad()
def similarity_transform(source, target, eps=1e-6):
    """Least-squares orientation-preserving 2D similarity, with degeneracy mask."""
    source, target = source.float(), target.float()
    mx, my = source.mean(-2, keepdim=True), target.mean(-2, keepdim=True)
    x, y = source - mx, target - my
    covariance = x.transpose(-1, -2) @ y
    u, singular, vh = torch.linalg.svd(covariance)
    sign = torch.linalg.det(u @ vh).sign()
    diagonal = torch.stack((torch.ones_like(sign), sign), -1)
    rotation = (u * diagonal.unsqueeze(-2)) @ vh
    variance = x.square().sum((-2, -1))
    scale = (singular * diagonal).sum(-1) / variance.clamp_min(eps)
    valid = (variance > eps) & (y.square().sum((-2, -1)) > eps) & (scale > eps)
    for points in (source, target):
        distances = torch.cdist(points, points)
        distances = distances + torch.eye(points.shape[-2], device=points.device) * 1e6
        valid &= distances.amin((-2, -1)) > 1e-3
    return rotation, scale, mx, my, valid


@torch.no_grad()
def teacher_pseudopoints(queries, matched, confidence, top_k=3):
    ids = confidence.topk(top_k, dim=-1).indices
    gather = ids.unsqueeze(-1).expand(*ids.shape, 2)
    source = queries.gather(-2, gather)
    target = matched.gather(-2, gather)
    rotation, scale, mx, my, valid = similarity_transform(source, target)
    pseudo = ((queries - mx) @ rotation) * scale[..., None, None] + my
    pseudo.scatter_(-2, gather, target)
    return pseudo, valid

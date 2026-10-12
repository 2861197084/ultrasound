from __future__ import annotations

import copy
import math
import sys

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import pixel_grid, sample_pixels, teacher_pseudopoints
from .runtime import ROOT


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.a), self.b) * self.scale


class NormalizedConv(nn.Conv2d):
    """Spatial kernel normalization used by the cited DINO-Tracker head."""
    def forward(self, x):
        denominator = self.weight.sum((-2, -1), keepdim=True)
        sign = torch.where(denominator < 0, -1.0, 1.0)
        denominator = sign * denominator.abs().clamp_min(1e-6)
        return F.conv2d(x, self.weight / denominator, self.bias, padding=self.padding)


class PointHead(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.refiner = nn.Sequential(NormalizedConv(1, hidden, 3, padding=1), nn.ReLU(),
                                     NormalizedConv(hidden, 1, 3, padding=1))

    def forward(self, correlations):
        b, n, h, w = correlations.shape
        return self.refiner(correlations.reshape(b * n, 1, h, w)).reshape(b, n, h, w)


def local_soft_argmax(correlations, logits, patch_size=14, radius=35):
    b, n, h, w = correlations.shape
    grid = pixel_grid(h, w, stride=patch_size, offset=patch_size / 2,
                      device=correlations.device).reshape(-1, 2)
    maximum = correlations.flatten(-2).argmax(-1)
    center = grid[maximum]
    neighborhood = (grid[None, None] - center[:, :, None]).square().sum(-1) <= radius**2
    # Local normalization is equivalent to globally softmaxing then renormalizing
    # inside the circular neighborhood, but avoids numerical underflow.
    local = logits.float().flatten(-2).masked_fill(~neighborhood, -torch.inf).softmax(-1)
    # Keep subpixel coordinates in FP32 even inside the backbone AMP context.
    # An autocast matmul here would quantize coordinates to bfloat16 pixels.
    xy = (local[..., None] * grid).sum(-2)
    global_prob = logits.float().flatten(-2).softmax(-1).reshape(b, n, h, w)
    return xy, global_prob


def cosine_maps(source, target, queries, patch_size=14):
    feature_xy = (queries - patch_size / 2) / patch_size
    query_features = sample_pixels(source, feature_xy)
    query_features = F.normalize(query_features, dim=-1, eps=1e-6)
    target_features = F.normalize(target.float(), dim=1, eps=1e-6)
    return torch.einsum("bnc,bchw->bnhw", query_features, target_features)


class M2P(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vendor = ROOT / "vendor" / ("dinov2-" + config["dinov2_commit"])
        sys.path.insert(0, str(vendor))
        from dinov2.hub.backbones import dinov2_vitb14
        self.backbone = dinov2_vitb14(pretrained=False)
        state = torch.load(ROOT / "assets/dinov2_vitb14_pretrain.pth", map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(False)
        for block in self.backbone.blocks:
            for parent, name in ((block.attn, "qkv"), (block.attn, "proj"),
                                 (block.mlp, "fc1"), (block.mlp, "fc2")):
                setattr(parent, name, LoRALinear(getattr(parent, name), config["lora_rank"],
                                                config["lora_alpha"], config["lora_dropout"]))
        self.head = PointHead(config["head_hidden"])
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None], persistent=False)
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None], persistent=False)

    def features(self, images):
        size = self.config["patch_size"]
        images = (images - self.rgb_mean) / self.rgb_std
        tokens = self.backbone.forward_features(images)["x_norm_patchtokens"]
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, images.shape[-2] // size, images.shape[-1] // size)

    def predict_pair(self, frame0, frame1, queries, raw=False):
        features = self.features(torch.cat((frame0, frame1)))
        source, target = features.chunk(2)
        correlations = cosine_maps(source, target, queries, self.config["patch_size"])
        logits = correlations if raw else self.head(correlations)
        xy, probabilities = local_soft_argmax(correlations, logits, self.config["patch_size"], self.config["argmax_radius"])
        return xy, probabilities, correlations

    def adapter_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters() if value.requires_grad}

    def load_adapter(self, state):
        params = dict(self.named_parameters())
        expected = {name for name, value in params.items() if value.requires_grad}
        if set(state) != expected:
            raise ValueError("Adapter keys do not match this model")
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != params[name].shape:
                    raise ValueError(f"Wrong adapter shape: {name}")
                params[name].copy_(value)

    def frozen_teacher(self):
        return copy.deepcopy(self).eval().requires_grad_(False)

    @torch.inference_mode()
    def track(self, video, queries, raw=False, chunk_size=8):
        """video T,C,H,W; queries N,(t,x,y). Return T,N,xy and uncalibrated similarity."""
        device = next(self.parameters()).device
        self.eval()
        t, _, h, w = video.shape
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("Queries must have shape N,3 in (time,x,y) order")
        q = queries.to(device=device, dtype=torch.float32)
        query_times = q[:, 0].long()
        if not torch.equal(q[:, 0], query_times.float()) or (query_times < 0).any() or (query_times >= t).any():
            raise ValueError("Invalid query frame index")
        query_features = torch.empty((len(q), 768), device=device)
        for frame in query_times.unique():
            ids = query_times == frame
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source = self.features(video[int(frame):int(frame) + 1].to(device))
            query_features[ids] = sample_pixels(source, ((q[ids, 1:] - 7) / 14)[None])[0]
        query_features = F.normalize(query_features, dim=-1, eps=1e-6)
        tracks, scores = [], []
        for start in range(0, t, chunk_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                target = self.features(video[start:start + chunk_size].to(device))
            target = F.normalize(target.float(), dim=1, eps=1e-6)
            correlations = torch.einsum("nc,bchw->bnhw", query_features, target)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = correlations if raw else self.head(correlations)
            xy, _ = local_soft_argmax(correlations, logits, radius=self.config["argmax_radius"])
            tracks.append(xy.cpu())
            scores.append(correlations.flatten(-2).amax(-1).cpu())
        tracks, scores = torch.cat(tracks), torch.cat(scores)
        tracks[query_times.cpu(), torch.arange(len(q))] = q[:, 1:].cpu()
        return tracks, scores


@torch.no_grad()
def teacher_targets(teacher, batch, cfg):
    """Frozen correspondences, reusable across the accumulation loss passes."""
    b, g, k, _ = batch["queries"].shape
    queries = batch["queries"].reshape(b, g * k, 2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tf = teacher.features(torch.cat((batch["image0"], batch["image1"])))
    ts, tt = tf.chunk(2)
    correlation = cosine_maps(ts, tt, queries)
    best = correlation.flatten(-2).argmax(-1)
    grid = pixel_grid(*correlation.shape[-2:], stride=14, offset=7, device=queries.device).reshape(-1, 2)
    matched = grid[best].reshape(b, g, k, 2)
    confidence = correlation.flatten(-2).amax(-1).reshape(b, g, k)
    return teacher_pseudopoints(batch["queries"], matched, confidence, cfg["top_matches"])


def objective(model, teacher, batch, *, targets=None, lsc_denominator=None):
    cfg = model.config
    b, g, k, _ = batch["queries"].shape
    queries = batch["queries"].reshape(b, g * k, 2)
    pseudo, valid_groups = teacher_targets(teacher, batch, cfg) if targets is None else targets
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xy, probabilities, _ = model.predict_pair(batch["image0"], batch["image1"], queries)
    predicted = xy.float().reshape(b, g, k, 2)
    point_loss = F.huber_loss(predicted, pseudo, reduction="none", delta=cfg["huber_delta"]).sum(-1).mean(-1)
    denominator = valid_groups.sum().clamp_min(1) if lsc_denominator is None else lsc_denominator
    lsc = (point_loss * valid_groups).sum() / denominator
    fh, fw = probabilities.shape[-2:]
    centers = pixel_grid(fh, fw, stride=14, offset=7, device=xy.device).reshape(1, -1, 2).expand(b, -1, -1)
    target_mask = sample_pixels(batch["mask1"][:, None], centers)[..., 0].reshape(b, 1, fh, fw)
    mass = (probabilities * target_mask).sum((-2, -1))
    mlc = torch.where(mass <= cfg["mlc_threshold"], -torch.log(mass + cfg["epsilon"]), 0.0).mean()
    d0 = sample_pixels(batch["distance0"][:, None], batch["queries"])[..., 0]
    d1 = sample_pixels(batch["distance1"][:, None], predicted)[..., 0]
    n0 = d0 / (d0.sum(-1, keepdim=True) + cfg["epsilon"])
    n1 = d1 / (d1.sum(-1, keepdim=True) + cfg["epsilon"])
    weights = torch.exp(-d0)
    mbc = ((weights * (n0 - n1).abs()).sum(-1) / (weights.sum(-1) + cfg["epsilon"])).mean()
    total = cfg["loss_weights"][0] * lsc + cfg["loss_weights"][1] * mlc + cfg["loss_weights"][2] * mbc
    metrics = dict(total=total.detach(), lsc=lsc.detach(), mlc=mlc.detach(), mbc=mbc.detach(),
                   foreground_mass=mass.mean().detach(), valid_lsc_groups=valid_groups.float().mean())
    return total, metrics

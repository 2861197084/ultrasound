from __future__ import annotations

import ast
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .geometry import sample_pixels
from .runtime import ROOT, RUN, atomic_json, event, guard_disk, file_sha


def evaluation_fingerprint(model, raw):
    state_hash = hashlib.sha256()
    for name, value in sorted(model.adapter_state().items()):
        state_hash.update(name.encode())
        state_hash.update(value.numpy().tobytes())
    return dict(adapter_sha256=state_hash.hexdigest(), base_sha256=file_sha(ROOT / "assets/dinov2_vitb14_pretrain.pth"),
                source={name: file_sha(ROOT / "m2p_lab" / name) for name in ("model.py", "geometry.py", "evaluation.py")},
                eval_size=model.config["eval_size"], argmax_radius=model.config["argmax_radius"], raw=raw,
                torch=str(torch.__version__), tapnet_commit=model.config["tapnet_commit"])


def official_metrics():
    """Use unchanged upstream NumPy functions without importing TF/JAX stacks."""
    source = (ROOT / "assets/tapvid_evaluation_datasets.py").read_text()
    tree = ast.parse(source)
    wanted = {"sample_queries_first", "compute_tapvid_metrics"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(functions) != len(wanted):
        raise RuntimeError("Upstream TAP-Vid functions changed")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + functions, type_ignores=[])
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(module), "official_tapvid_functions", "exec"), scope)
    return scope["sample_queries_first"], scope["compute_tapvid_metrics"]


def decode_video(value):
    if isinstance(value, np.ndarray) and value.ndim == 4:
        return value
    return np.stack([np.asarray(Image.open(BytesIO(frame)).convert("RGB")) for frame in value])


def benchmark_path(name):
    found = list((ROOT.parent / "data/benchmarks").rglob("*.pkl"))
    candidates = [p for p in found if ("davis" in p.name.lower()) == (name == "davis")
                  and (name == "davis" or "stack" in p.name.lower())]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot locate {name} benchmark: {candidates}")
    return candidates[0]


def comparison_panel(video, tracks, queries, path, masks=None):
    times = np.linspace(0, len(video) - 1, 4).astype(int)
    width, height = 256, 256
    panel = Image.new("RGB", (width * len(times), height + 24), "white")
    colors = [(250, 90, 80), (30, 210, 160), (80, 150, 255), (240, 200, 60), (210, 100, 255)]
    for col, time in enumerate(times):
        image = Image.fromarray(video[time]).convert("RGB").resize((width, height))
        draw = ImageDraw.Draw(image)
        points = tracks[time]
        for n in range(min(len(points), 24)):
            color = colors[n % len(colors)]
            x, y = points[n]
            if np.isfinite([x, y]).all():
                draw.ellipse((float(x - 2), float(y - 2), float(x + 2), float(y + 2)), fill=color)
        panel.paste(image, (col * width, 24))
        ImageDraw.Draw(panel).text((col * width + 5, 5), f"Frame {time}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, quality=88)


@torch.inference_mode()
def evaluate_benchmarks(model, label, raw=False, limit=None):
    model.eval()
    sample_first, metric_fn = official_metrics()
    destination = RUN / "evaluation" / label
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = evaluation_fingerprint(model, raw)
    existing = destination / "summary.json"
    metadata = destination / "fingerprint.json"
    if limit is None and existing.exists() and metadata.exists() and json.loads(metadata.read_text()) == fingerprint:
        previous = json.loads(existing.read_text())
        if previous.get("davis", {}).get("videos") == 30 and previous.get("rgb_stacking", {}).get("videos") == 50:
            event("benchmark_reused", model=label)
            return previous
    summaries = {}
    for name in ("davis", "rgb_stacking"):
        # These are the official, checksum-recorded DeepMind dataset pickles.
        with benchmark_path(name).open("rb") as handle:
            raw_data = pickle.load(handle)
        entries = sorted(raw_data.items()) if isinstance(raw_data, dict) else [(str(i), r) for i, r in enumerate(raw_data)]
        expected_count = 30 if name == "davis" else 50
        if len(entries) != expected_count:
            raise RuntimeError(f"Unexpected {name} video count: {len(entries)}")
        if limit is not None:
            entries = entries[:limit]
        rows = []
        for index, (video_id, example) in enumerate(entries):
            frames = decode_video(example["video"])
            normalized_points = np.asarray(example["points"], dtype=np.float32)
            batch = sample_first(np.asarray(example["occluded"], dtype=bool), normalized_points * 256, frames)
            q = batch["query_points"][0]
            query = torch.from_numpy(q[:, [0, 2, 1]].copy()).float()
            query[:, 1:] *= model.config["eval_size"] / 256
            images = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255
            images = F.interpolate(images, (model.config["eval_size"],) * 2, mode="bilinear", align_corners=False)
            tracks, confidence = model.track(images, query, raw=raw)
            prediction = tracks.numpy().transpose(1, 0, 2)[None] * (256 / model.config["eval_size"])
            # delta_avg does not depend on predicted visibility. Do not publish
            # AJ/OA for this feature-matching baseline without an occlusion model.
            measured = metric_fn(batch["query_points"], batch["occluded"], batch["target_points"],
                                  np.zeros_like(batch["occluded"]), prediction, query_mode="first")
            row = dict(video=str(video_id), **{k: float(v[0]) * 100 for k, v in measured.items()
                                              if k.startswith("pts_within_") or k == "average_pts_within_thresh"})
            rows.append(row)
            np.savez_compressed(destination / f"{name}_{video_id}_tracks.npz", tracks_xy=prediction[0],
                                query_tyx=q, confidence=confidence.numpy().T)
            if index < 3:
                comparison_panel(frames, prediction[0].transpose(1, 0, 2), q, destination / f"{name}_{video_id}.jpg")
            atomic_json(destination / (name + "_progress.json"), dict(completed=len(rows), total=len(entries), last=row))
            event("benchmark_video", model=label, dataset=name, completed=len(rows), total=len(entries), **row)
        summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "video"}
        summary.update(videos=len(rows), smoke_subset=limit is not None, query_mode="first", feature_map=[32, 32], evaluation_coordinate_size=256,
                       reported_metrics="Visible-point localization only; no calibrated occlusion prediction")
        atomic_json(destination / (name + ".json"), dict(summary=summary, videos=rows))
        with (destination / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summaries[name] = summary
        guard_disk(model.config)
        del raw_data
    atomic_json(destination / "summary.json", summaries)
    atomic_json(destination / "fingerprint.json", fingerprint)
    return summaries


@torch.inference_mode()
def evaluate_echo(model, records, label, raw=False):
    destination = RUN / "echo_evaluation" / label
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    from .geometry import grouped_queries
    from .data import stable_seed
    size = model.config["train_size"]
    for index, record in enumerate(records):
        with np.load(record["path"], allow_pickle=False) as handle:
            images, masks = handle["images"], handle["masks"]
            timestamps = handle["timestamps"].copy()
        rng = np.random.default_rng(stable_seed(record["sample_id"]))
        xy = grouped_queries(masks[0], rng).reshape(-1, 2)
        query = torch.from_numpy(np.column_stack((np.zeros(len(xy)), xy))).float()
        video = torch.from_numpy(images.copy()).float()[:, None].expand(-1, 3, -1, -1) / 255
        tracks, confidence = model.track(video, query, raw=raw)
        inside = sample_pixels(torch.from_numpy(masks.astype(np.float32))[:, None], tracks)[..., 0] >= 0.5
        reverse_query = torch.cat((torch.zeros((len(xy), 1)), tracks[-1]), dim=-1)
        reverse_tracks, _ = model.track(video.flip(0), reverse_query, raw=raw)
        cycle = torch.linalg.vector_norm(reverse_tracks[-1] - query[:, 1:], dim=-1)
        # Known synthetic translation is an independent geometric test, not
        # ground truth for the real temporal trajectories.
        synthetic = torch.zeros_like(video[:2])
        synthetic[0] = video[0]
        synthetic[1, :, 5:, 7:] = video[0, :, :-5, :-7]
        synthetic_tracks, _ = model.track(synthetic, query, raw=raw)
        visible = (query[:, 1] + 7 < size) & (query[:, 2] + 5 < size)
        synthetic_error = torch.linalg.vector_norm(synthetic_tracks[1] - query[:, 1:] - torch.tensor([7, 5]), dim=-1)
        row = dict(sample_id=record["sample_id"], exam_id=record["exam_id"],
                   mask_retention=float(inside[1:].float().mean()), cycle_error_pixels=float(cycle.mean()),
                   synthetic_translation_epe=float(synthetic_error[visible].mean()),
                   cycle_error_mm=float(cycle.mean()) * record["pixel_spacing_m"] * 1000,
                   frames=len(images), points=len(xy))
        rows.append(row)
        np.savez_compressed(destination / (record["sample_id"] + ".npz"), tracks_xy=tracks.numpy(),
                            confidence=confidence.numpy(), timestamps=timestamps, query_txy=query.numpy())
        if index < 4:
            comparison_panel(images, tracks.numpy() * (256 / size), query.numpy(), destination / (record["sample_id"] + ".jpg"))
        event("echo_evaluation_video", model=label, **row)
        atomic_json(destination / "progress.json", dict(completed=len(rows), total=len(records), last=row))
    keys = ("mask_retention", "cycle_error_pixels", "cycle_error_mm", "synthetic_translation_epe")
    summary = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    summary.update(videos=len(rows), note="These are proxy and synthetic-warp metrics, not real myocardial point-tracking accuracy.")
    atomic_json(destination / "summary.json", dict(summary=summary, videos=rows))
    return summary

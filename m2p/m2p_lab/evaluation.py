from __future__ import annotations

import ast
import csv
import hashlib
from io import BytesIO
import json
import pickle

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

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
        for video_id, example in entries:
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

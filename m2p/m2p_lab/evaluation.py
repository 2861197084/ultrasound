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

from . import runtime
from .runtime import ROOT, atomic_json, event, guard_disk, file_sha


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
    destination = runtime.RUN / "evaluation" / label
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


def reference_queries(points_m, reference_valid, grid, size):
    """First valid position for each middle-wall point with a future target."""
    from .echoxflow import physical_to_pixels
    valid = np.asarray(reference_valid, bool) & np.isfinite(points_m).all(-1)
    ids = np.flatnonzero(valid.sum(0) >= 2)
    if not len(ids):
        raise ValueError("No reference point has a query and a future valid target")
    first = valid[:, ids].argmax(0)
    xy = physical_to_pixels(points_m[first, ids], grid, size)
    return np.column_stack((first, xy)).astype(np.float32), ids


def echo_point_metrics(predicted_xy, reference_m, reference_valid, query_times, grid, size):
    from .echoxflow import pixels_to_physical
    prediction = np.asarray(predicted_xy)
    reference = np.asarray(reference_m)
    if prediction.shape != reference.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid predicted trajectories")
    after_query = np.arange(len(prediction))[:, None] > np.asarray(query_times)[None]
    eligible = after_query & np.asarray(reference_valid, bool) & np.isfinite(reference).all(-1)
    if not eligible.any():
        raise ValueError("No evaluable target positions")
    predicted_m = pixels_to_physical(prediction, grid, size)
    errors_m = np.linalg.norm(predicted_m - reference, axis=-1)[eligible]
    errors_256 = errors_m * (256 / grid["side_m"])
    scores = {f"pck_{threshold}": float((errors_256 < threshold).mean() * 100) for threshold in (1, 2, 4, 8, 16)}
    return dict(epe_mm=float(errors_m.mean() * 1000), pck_average=float(np.mean(list(scores.values()))), **scores,
                evaluated_positions=int(eligible.sum()), possible_positions=int(after_query.sum()),
                query_points=prediction.shape[1])


def aggregate_echo_records(rows):
    keys = ("epe_mm", "pck_average", "pck_1", "pck_2", "pck_4", "pck_8", "pck_16")
    exams = []
    for exam in sorted({row["exam_id"] for row in rows}):
        selected = [row for row in rows if row["exam_id"] == exam]
        exams.append(dict(exam_id=exam, sequences=len(selected),
                          **{key: float(np.mean([row[key] for row in selected])) for key in keys}))
    return dict(exams=exams, metrics={key: float(np.mean([row[key] for row in exams])) for key in keys},
                sequences=len(rows), examinations=len(exams),
                evaluated_positions=sum(row["evaluated_positions"] for row in rows),
                possible_positions=sum(row["possible_positions"] for row in rows),
                query_points=sum(row["query_points"] for row in rows))


@torch.inference_mode()
def evaluate_echoxflow(model, manifest, label, raw=False):
    import zarr
    records = [record for record in manifest["records"] if record["split"] == "test"]
    destination = runtime.RUN / "evaluation" / label
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = evaluation_fingerprint(model, raw)
    fingerprint.update(data_fingerprint=manifest["fingerprint"], coordinate_size=256,
                       echo_source=file_sha(ROOT / "m2p_lab/echoxflow.py"))
    summary_path, identity_path = destination / "summary.json", destination / "fingerprint.json"
    if summary_path.exists():
        if not identity_path.exists() or json.loads(identity_path.read_text()) != fingerprint:
            raise RuntimeError("Saved EchoXFlow evaluation differs from the current model/data/protocol")
        previous = json.loads(summary_path.read_text())
        if [row["id"] for row in previous["sequences"]] != [record["id"] for record in records]:
            raise RuntimeError("Saved evaluation does not cover the fixed test set")
        event("echo_evaluation_reused", model=label, sequences=len(records))
        return previous
    atomic_json(identity_path, fingerprint)
    rows = []
    size = model.config["eval_size"]
    for number, record in enumerate(records, 1):
        group = zarr.open_group(str(runtime.RUN / record["cache"]), mode="r")
        references, valid = group["reference_points_m"][:], group["reference_valid"][:]
        queries, ids = reference_queries(references, valid, record["grid"], size)
        # Targets remain outside the model; only the query positions enter track().
        video = torch.from_numpy(group["frames"][:]).unsqueeze(1).expand(-1, 3, -1, -1).float() / 255
        tracks, confidence = model.track(video, torch.from_numpy(queries), raw=raw,
                                         chunk_size=model.config.get("eval_chunk_size", 1))
        measured = echo_point_metrics(tracks.numpy(), references[:, ids], valid[:, ids],
                                      queries[:, 0], record["grid"], size)
        row = dict(id=record["id"], exam_id=record["exam_id"], view=record["view"], **measured)
        rows.append(row)
        np.savez_compressed(destination / (record["id"] + "_tracks.npz"), tracks_xy=tracks.numpy(),
                            query_txy=queries, reference_point_indices=ids, confidence=confidence.numpy())
        atomic_json(destination / "progress.json", dict(completed=number, total=len(records), last=row))
        event("echo_evaluation_video", model=label, completed=number, total=len(records), **row)
        del video, tracks, confidence
    aggregate = aggregate_echo_records(rows)
    result = dict(summary={k: v for k, v in aggregate.items() if k != "exams"}, exams=aggregate["exams"], sequences=rows,
                  data_fingerprint=manifest["fingerprint"], query_mode="first_valid",
                  normalized_coordinate_size=256, metric_aggregation="sequence mean then examination mean",
                  reference_kind=manifest["reference_kind"], masks_used_at_inference=False)
    atomic_json(summary_path, result)
    guard_disk(model.config)
    event("echo_evaluation_finished", model=label, **result["summary"])
    return result


def compare_echoxflow(baseline, trained, cfg):
    if baseline["data_fingerprint"] != trained["data_fingerprint"]:
        raise ValueError("Cannot compare evaluations on different data")
    if [row["id"] for row in baseline["sequences"]] != [row["id"] for row in trained["sequences"]]:
        raise ValueError("Cannot compare different test sequences")
    before, after = baseline["exams"], trained["exams"]
    if [row["exam_id"] for row in before] != [row["exam_id"] for row in after]:
        raise ValueError("Cannot pair different examinations")
    rng = np.random.default_rng(cfg["seed"])
    indices = rng.integers(len(before), size=(cfg["bootstrap_samples"], len(before)))
    changes = {}
    for key in baseline["summary"]["metrics"]:
        differences = np.array([b[key] - a[key] for a, b in zip(before, after)])
        interval = np.quantile(differences[indices].mean(1), [0.025, 0.975])
        changes[key] = dict(trained_minus_baseline=float(differences.mean()), ci95=interval.tolist())
    return dict(baseline=baseline["summary"], trained=trained["summary"], paired_changes=changes,
                bootstrap_unit="examination", bootstrap_samples=cfg["bootstrap_samples"], seed=cfg["seed"],
                data_fingerprint=baseline["data_fingerprint"], reference_kind=baseline["reference_kind"])

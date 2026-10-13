"""EchoXFlow A4C 原始数据准备、后训练和软件参考轨迹评测。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PREPARATION_VERSION = 1
METRIC_KEYS = [
    "l1_delta_1", "l1_delta_2", "l1_delta_4", "l1_davg_1_2_4", "l1_mte", "l1_mean_error",
    "l2_delta_1", "l2_delta_2", "l2_delta_4", "l2_delta_8", "l2_delta_16",
    "l2_davg_1_2_4", "l2_davg_1_2_4_8_16", "l2_mte", "l2_mean_error",
]


def sha_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def config_from_file(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("raw_root", "scan_source", "output", "checkpoint"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if cfg["image_size"] != 256:
        raise ValueError("This protocol requires a 256 x 256 image grid")
    folds = [set(cfg[k + "_folds"]) for k in ("train", "validation", "test")]
    if set.union(*folds) != set(range(5)) or any(folds[i] & folds[j] for i in range(3) for j in range(i)):
        raise ValueError("Official folds must form three disjoint partitions")
    return cfg


def preparation_identity(cfg):
    raw = Path(cfg["raw_root"])
    return {
        "version": PREPARATION_VERSION, "view": "4ch", "image_size": cfg["image_size"],
        "data_revision": cfg["data_revision"], "scan_revision": cfg["scan_revision"],
        "catalog_sha256": sha_file(raw / "croissant.json"),
        "splits_sha256": sha_file(raw / "splits.csv"),
        "folds": {k: cfg[k + "_folds"] for k in ("train", "validation", "test")},
        "coordinates": "square physical field, pixel centers 0..255, midwall indices 2::5",
        "alignment": "contour-to-nearest-video; half median video interval + 1e-6 s; closest collision wins",
    }


def align_timestamps(contour_times, video_times):
    """每条轮廓仅匹配一帧；冲突取时间误差最小者，平局取较早轮廓。"""
    ct, bt = np.asarray(contour_times), np.asarray(video_times)
    if min(len(ct), len(bt)) < 2 or not np.isfinite(ct).all() or not np.isfinite(bt).all():
        raise ValueError("Invalid or insufficient timestamps")
    if not (np.diff(ct) > 0).all() or not (np.diff(bt) > 0).all():
        raise ValueError("Timestamps must increase strictly")
    tolerance = float(np.median(np.diff(bt)) * .5 + 1e-6)
    distance = abs(ct[:, None] - bt[None, :])
    nearest = distance.argmin(axis=1)
    errors = distance[np.arange(len(ct)), nearest]
    if (errors > tolerance).any():
        raise ValueError("A contour timestamp has no video frame within tolerance")
    frame_indices = np.arange(nearest.min(), nearest.max() + 1, dtype=np.int64)
    contour_indices = np.full(len(frame_indices), -1, dtype=np.int64)
    for ci in np.argsort(errors, kind="stable"):
        dest = nearest[ci] - frame_indices[0]
        if contour_indices[dest] == -1:
            contour_indices[dest] = ci
    return frame_indices, contour_indices, tolerance, errors


def checked_array(group, spec):
    value = np.asarray(group[spec["arrays/array_path"]][:])
    declaration = next(a for a in group.attrs["arrays"] if a["name"] == spec["arrays/array_path"])
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    actual = digest.hexdigest()
    if (list(value.shape) != spec["arrays/shape"] or str(value.dtype) != spec["arrays/dtype"]
            or actual != declaration["data_sha256"]
            or (spec["arrays/data_sha256"] and actual != spec["arrays/data_sha256"])):
        raise ValueError(f"Official array verification failed: {spec['arrays/array_id']}")
    return value, actual


def prepare(cfg):
    import zarr
    sys.path.insert(0, cfg["scan_source"])
    from echoxflow.scan.geometry import CartesianGrid, physical_points_to_pixels, sector_geometry_from_mapping
    from echoxflow.scan.conversion import sector_to_cartesian

    destination = Path(cfg["output"]) / "prepared"
    identity = preparation_identity(cfg)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] == identity:
            for item in previous["records"]:
                if sha_file(destination / item["path"]) != item["sha256"]:
                    raise ValueError(f"Prepared cache changed: {item['sample_id']}")
            print("Prepared cache matches configuration and all saved sample hashes.", flush=True)
            return previous

    raw = Path(cfg["raw_root"])
    sets = {s["@id"]: s["data"] for s in json.loads((raw / "croissant.json").read_text())["recordSet"]}
    arrays = {a["arrays/array_id"]: a for a in sets["arrays"]}
    recordings = {r["recordings/recording_id"]: r for r in sets["recordings"]}
    with (raw / "splits.csv").open() as f:
        folds = {r["exam_id"]: int(r["five_fold_split"]) for r in csv.DictReader(f)}
    groups, exclusions = defaultdict(list), []
    for link in sets["recording_links"]:
        if link["recording_links/panel_role_id"] != "4ch":
            continue
        frame = arrays.get(link["recording_links/linked_frame_array_id"])
        if frame is None:
            exclusions.append({"link_id": link["recording_links/link_id"], "reason": "unresolved_frame_array"})
            continue
        groups[frame["arrays/data_sha256"]].append(link)

    results = []
    split_exams, split_hashes = defaultdict(set), defaultdict(set)
    for number, (video_hash, candidates) in enumerate(sorted(groups.items()), 1):
        candidates.sort(key=lambda x: (x["recording_links/source_recording_id"], x["recording_links/linked_recording_id"]))
        candidate_folds = {
            folds[recordings[x[key]]["recordings/exam_id"]]
            for x in candidates
            for key in ("recording_links/source_recording_id", "recording_links/linked_recording_id")
        }
        if len(candidate_folds) != 1:
            raise ValueError("A linked or duplicate recording crosses official folds")
        fold = candidate_folds.pop()
        split = next(s for s in ("train", "validation", "test") if fold in cfg[s + "_folds"])
        link = candidates[0]
        annotation_id, video_id = (link[k] for k in ("recording_links/source_recording_id", "recording_links/linked_recording_id"))
        ag = zarr.open_group(str(raw / link["recording_links/source_recording_zarr_path"]), mode="r")
        vg = zarr.open_group(str(raw / link["recording_links/linked_recording_zarr_path"]), mode="r")
        panel = next(p for p in ag.attrs["recording_manifest"]["linked_panels"] if p["semantic_id"] == "4ch")
        if panel["linked_recording"]["recording_id"] != video_id:
            raise ValueError("Catalog and recording manifest disagree about the video")
        source_arrays = {}
        values = []
        for group, array_id in (
            (ag, annotation_id + "/data/4ch_contour"),
            (ag, annotation_id + "/timestamps/4ch_contour"),
            (vg, link["recording_links/linked_frame_array_id"]),
            (vg, link["recording_links/linked_timestamp_array_id"]),
        ):
            value, digest = checked_array(group, arrays[array_id])
            values.append(value)
            source_arrays[array_id] = digest
        contours, ct, beamspace, bt = values
        if contours.ndim != 3 or contours.shape[-1] != 2 or contours.shape[1] % 5:
            raise ValueError("Invalid LV five-layer contour layout")
        if len(contours) != len(ct) or len(beamspace) != len(bt) or beamspace.ndim != 3:
            raise ValueError("Frame/contour timestamp dimensions disagree")
        frame_indices, contour_indices, tolerance, errors = align_timestamps(ct, bt)
        geometry = sector_geometry_from_mapping(panel["geometry"], grid_shape=beamspace.shape[1:])
        natural = CartesianGrid.from_sector_height(geometry, cfg["image_size"])
        xr, yr = natural.x_range_m, natural.y_range_m
        side = max(xr[1] - xr[0], yr[1] - yr[0])
        xc, yc = sum(xr) / 2, sum(yr) / 2
        grid = CartesianGrid(shape=(256, 256), x_range_m=(xc - side/2, xc + side/2), y_range_m=(yc - side/2, yc + side/2))
        images = np.stack([sector_to_cartesian(beamspace[i], geometry, grid=grid).data for i in frame_indices])
        point_indices = np.arange(2, contours.shape[1], 5, dtype=np.int64)
        xy = np.zeros((len(images), len(point_indices), 2), dtype=np.float32)
        valid = np.zeros(xy.shape[:-1], dtype=bool)
        for frame, ci in enumerate(contour_indices):
            if ci == -1:
                continue
            physical = contours[ci, point_indices]
            finite = np.isfinite(physical).all(-1)
            safe = np.where(finite[:, None], physical, 0)
            xy[frame] = physical_points_to_pixels(safe, grid)
            radius, angle = np.linalg.norm(safe, axis=-1), np.arctan2(safe[:, 0], safe[:, 1])
            valid[frame] = (finite & (radius >= geometry.depth_start_m) & (radius <= geometry.depth_end_m)
                            & (angle >= geometry.angle_start_rad) & (angle <= geometry.angle_end_rad)
                            & (xy[frame] >= 0).all(-1) & (xy[frame] <= 255).all(-1))
        if not valid[0].any() or not valid[1:, valid[0]].any():
            raise ValueError("No evaluable first-frame query tracks")
        sid = "4ch_" + hashlib.sha256((video_hash + ":4ch").encode()).hexdigest()[:20]
        path = destination / "sequences" / (sid + ".npz")
        save_npz(path, frames=images, trajs=xy, valid=valid, point_indices=point_indices,
                 frame_indices=frame_indices, contour_indices=contour_indices, timestamps=bt[frame_indices])
        exam_id = recordings[video_id]["recordings/exam_id"]
        annotation_exam = recordings[annotation_id]["recordings/exam_id"]
        all_exams = {recordings[c[k]]["recordings/exam_id"] for c in candidates for k in
                     ("recording_links/source_recording_id", "recording_links/linked_recording_id")}
        split_exams[split].update(all_exams)
        split_hashes[split].add(video_hash)
        results.append({
            "sample_id": sid, "exam_id": exam_id, "annotation_exam_id": annotation_exam,
            "fold": fold, "split": split, "annotation_id": annotation_id, "video_id": video_id,
            "video_sha256": video_hash, "source_arrays": source_arrays,
            "source_links": [x["recording_links/link_id"] for x in candidates],
            "annotation_path": link["recording_links/source_recording_zarr_path"],
            "video_path": link["recording_links/linked_recording_zarr_path"],
            "path": str(path.relative_to(destination)), "sha256": sha_file(path),
            "frames": len(images), "points": len(point_indices), "query_points": int(valid[0].sum()),
            "valid_positions": int(valid.sum()), "unmatched_frames": int((contour_indices == -1).sum()),
            "discarded_timestamp_collisions": int(len(ct) - (contour_indices >= 0).sum()),
            "timestamp_tolerance_seconds": tolerance, "max_timestamp_error_seconds": float(errors.max()),
            "geometry": panel["geometry"],
            "grid": {"shape": [256, 256], "x_range_m": grid.x_range_m, "y_range_m": grid.y_range_m},
        })
        if number % 10 == 0 or number == len(groups):
            print(f"Prepared {number}/{len(groups)} A4C videos; verified four source arrays per video.", flush=True)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_exams[left] & split_exams[right] or split_hashes[left] & split_hashes[right]:
            raise ValueError("Examination or video-content leakage across splits")
    results.sort(key=lambda r: r["sample_id"])
    manifest = {
        "complete": True, "identity": identity, "records": results, "excluded_links": exclusions,
        "counts": dict(Counter(r["split"] for r in results)),
        "exam_counts": {s: len({r["exam_id"] for r in results if r["split"] == s})
                        for s in ("train", "validation", "test")},
        "reference": "Software reference coordinates, not independently verified tissue-point ground truth.",
    }
    save_json(manifest_path, manifest)
    print(json.dumps({"prepared": manifest["counts"], "excluded_links": len(exclusions)}, ensure_ascii=False), flush=True)
    return manifest


def load_manifest(cfg):
    prepared = Path(cfg["output"]) / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    if not manifest["complete"] or manifest["identity"] != preparation_identity(cfg):
        raise ValueError("Prepared data do not match this configuration; run prepare")
    for item in manifest["records"]:
        if sha_file(prepared / item["path"]) != item["sha256"]:
            raise ValueError(f"Prepared sample hash mismatch: {item['sample_id']}")
    return manifest


class EchoXFlowDataset:
    """返回 uint8 RGB 帧、归一化 xy 参考点和有效掩码；无 Torch 依赖。"""
    def __init__(self, records, cfg, training=False):
        self.records, self.cfg, self.training = records, cfg, training
        self.prepared = Path(cfg["output"]) / "prepared"
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        with np.load(self.prepared / self.records[index]["path"], allow_pickle=False) as a:
            frames, xy, valid = a["frames"], a["trajs"], a["valid"]
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], self.epoch, index]))
            length = min(self.cfg["clip_frames"], len(frames))
            start = int(rng.integers(0, len(frames) - length + 1))
            frames, xy, valid = (a[start:start + length].copy() for a in (frames, xy, valid))
            if rng.random() < self.cfg["time_reverse_probability"]:
                frames, xy, valid = (a[::-1].copy() for a in (frames, xy, valid))
            if rng.random() < self.cfg["horizontal_flip_probability"]:
                frames = frames[:, :, ::-1].copy()
                xy[..., 0] = 255 - xy[..., 0]
        return np.repeat(frames[..., None], 3, axis=-1), xy / 255.0, valid


def initialize_torch(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu_uuid"]
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The configured physical GPU must be the only visible CUDA device")
    torch.set_num_threads(cfg["torch_threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    return torch


def make_model(cfg, checkpoint, training=False):
    from models.net import ECHOTRACKER2
    model = ECHOTRACKER2(device_ids=[0], ft_model=True, sp_attn=True,
                        resnet_temp_module=True, patch_size=9)
    model.load(path=checkpoint, eval=not training)
    return model


def evaluate_model(model, records, cfg, destination=None, warmup=True):
    import torch
    from utils.evaluate import compute_reference_metrics
    prepared = Path(cfg["output"]) / "prepared"
    rows = []
    if model is not None:
        model.model.eval()
    for index, item in enumerate(records):
        with np.load(prepared / item["path"], allow_pickle=False) as a:
            frames, reference, valid = a["frames"], a["trajs"], a["valid"]
            point_indices, frame_indices = a["point_indices"], a["frame_indices"]
        keep = valid[0] & valid[1:].any(axis=0)
        reference, valid, point_indices = reference[:, keep], valid[:, keep], point_indices[keep]
        if model is None:
            start = time.perf_counter()
            prediction = np.broadcast_to(reference[0], reference.shape).copy()
            elapsed, peak = time.perf_counter() - start, 0.0
        else:
            video = torch.from_numpy(np.repeat(frames[..., None], 3, axis=-1)).unsqueeze(0).to(model.device)
            query = torch.zeros((1, len(point_indices), 3), device=model.device)
            query[0, :, 1:] = torch.from_numpy(reference[0] / 255.0).to(model.device)
            with torch.inference_mode():
                if warmup and index == 0:
                    model.infer(video, query)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                prediction = model.infer(video, query)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated() / 2**30
                prediction = prediction[0].permute(1, 0, 2).cpu().numpy() * 255.0
            del video, query
        row = {k: item[k] for k in ("sample_id", "exam_id", "fold")}
        row.update(compute_reference_metrics(reference, prediction, valid))
        row.update(inference_seconds=elapsed, peak_allocated_gib=peak)
        rows.append(row)
        if destination is not None:
            save_npz(Path(destination) / "predictions" / (item["sample_id"] + ".npz"),
                     tracks=prediction, point_indices=point_indices,
                     frame_indices=frame_indices, query_frame=np.int64(0))
        if (index + 1) % 16 == 0:
            print(f"Evaluated {index + 1}/{len(records)} videos", flush=True)
    metrics = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
    metrics.update(videos=len(rows), examinations=len({r["exam_id"] for r in rows}),
                   mean_inference_seconds=float(np.mean([r["inference_seconds"] for r in rows])),
                   max_allocated_gib=float(max(r["peak_allocated_gib"] for r in rows)))
    if destination is not None:
        destination = Path(destination)
        save_json(destination / "metrics.json", metrics)
        with (destination / "per_video.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def record_runtime(cfg, manifest, stage):
    import torch
    output = Path(cfg["output"])
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("Run configuration changed; use a separate output directory")
    save_json(config_path, cfg)
    files = [Path(__file__), ROOT / "configs.py", *sorted((ROOT / "models").rglob("*.py")),
             *sorted((ROOT / "utils").rglob("*.py"))]
    sources = {str(p.relative_to(ROOT)): sha_file(p) for p in files}
    metadata = {
        "config_sha256": json_hash(cfg),
        "prepared_manifest_sha256": sha_file(output / "prepared/manifest.json"),
        "initial_checkpoint_sha256": sha_file(cfg["checkpoint"]),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources,
        "environment": {"python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "numpy": np.__version__, "gpu_uuid": cfg["gpu_uuid"],
                        "gpu_name": torch.cuda.get_device_name(0)},
        "split_counts": manifest["counts"],
    }
    for path in files:
        target = output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha_file(target) != sha_file(path):
            raise ValueError("Implementation changed inside an existing run; use a separate output directory")
        shutil.copyfile(path, target)
    save_json(output / ("runtime_" + stage + ".json"), metadata)
    with (output / "commands.log").open("a", encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat() + " " +
                shlex.join([sys.executable, "-B", str(Path(__file__)), *sys.argv[1:]]) + "\n")
    return metadata


def train(cfg, manifest, resume):
    torch = initialize_torch(cfg)
    metadata = record_runtime(cfg, manifest, "train")
    from torch.utils.data import DataLoader
    train_records = [r for r in manifest["records"] if r["split"] == "train"]
    validation = [r for r in manifest["records"] if r["split"] == "validation"]
    dataset = EchoXFlowDataset(train_records, cfg, training=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=cfg["loader_workers"],
                        generator=torch.Generator().manual_seed(cfg["seed"]), pin_memory=True)
    model = make_model(cfg, cfg["checkpoint"], training=True)
    model.train(
        dataloaders={"train": loader}, dataset_size={"train": len(dataset)},
        log_dir=str(Path(cfg["output"]) / "training"), ckpt_path=str(Path(cfg["output"]) / "checkpoints"),
        epochs=cfg["epochs"], q_frame=-2, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        accumulation_steps=cfg["accumulation_steps"], max_points=cfg["max_points"],
        gamma=cfg["gamma"], grad_clip=cfg["grad_clip"],
        validation_fn=lambda: evaluate_model(model, validation, cfg, warmup=False),
        metadata=metadata, resume=resume,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--config", default=str(ROOT / "echoxflow_a4c.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--model", nargs="+", choices=("pretrained", "finetuned", "static"),
                        default=["pretrained", "finetuned", "static"])
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    cfg = config_from_file(args.config)
    if args.command == "prepare":
        prepare(cfg)
        return
    manifest = load_manifest(cfg)
    if args.command == "train":
        train(cfg, manifest, args.resume)
        return
    torch = initialize_torch(cfg)
    runtime = record_runtime(cfg, manifest, "evaluate")
    records = [r for r in manifest["records"] if r["split"] == args.split]
    for name in args.model:
        path = (Path(cfg["checkpoint"]) if name == "pretrained" else
                args.checkpoint or Path(cfg["output"]) / "checkpoints/best.pth")
        model = None if name == "static" else make_model(cfg, path)
        destination = Path(cfg["output"]) / "evaluation" / args.split / name
        metrics = evaluate_model(model, records, cfg, destination=destination)
        save_json(destination / "provenance.json", {
            "model": name, "checkpoint": str(path) if model else None,
            "checkpoint_sha256": sha_file(path) if model else None,
            "config_sha256": runtime["config_sha256"],
            "prepared_manifest_sha256": runtime["prepared_manifest_sha256"],
            "query_frame": 0, "coordinate_scale": "256x256 pixels, 0..255",
        })
        print(json.dumps({"model": name, **metrics}, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

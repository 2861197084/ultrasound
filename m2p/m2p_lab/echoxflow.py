"""EchoXFlow masked pairs and physical Cartesian caches, independent of other projects."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numcodecs import Blosc
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
import torch
from torch.utils.data import Dataset
import zarr

from .data import prepare_pair, stable_seed
from . import runtime
from .runtime import ROOT, atomic_json, event, file_sha, guard_disk


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def align_timestamps(contour_times, video_times):
    source = np.asarray(contour_times, dtype=np.float64)
    target = np.asarray(video_times, dtype=np.float64)
    for values in (source, target):
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or not (np.diff(values) > 0).all():
            raise ValueError("Expected at least two finite, strictly increasing timestamps")
    tolerance = float(np.median(np.diff(target))) * 0.5 + 1e-6
    distances = np.abs(source[:, None] - target[None, :])
    nearest = distances.argmin(1)
    errors = distances[np.arange(len(source)), nearest]
    best = {}
    for index, (frame, error) in enumerate(zip(nearest, errors)):
        if error <= tolerance and (int(frame) not in best or error < best[int(frame)][1]):
            best[int(frame)] = (index, float(error))
    frames = sorted(best)
    contours = [best[frame][0] for frame in frames]
    return contours, frames, dict(tolerance_seconds=tolerance,
        outside_tolerance=int((errors > tolerance).sum()),
        duplicate_matches=int((errors <= tolerance).sum()) - len(frames),
        max_retained_error_seconds=max((best[frame][1] for frame in frames), default=0.0))


def square_geometry(geometry):
    depth0, depth1 = float(geometry["depth_start_m"]), float(geometry["depth_end_m"])
    angle0 = float(geometry["tilt_rad"]) - float(geometry["width_rad"]) / 2
    angle1 = float(geometry["tilt_rad"]) + float(geometry["width_rad"]) / 2
    if not np.isfinite([depth0, depth1, angle0, angle1]).all() or not (0 <= depth0 < depth1 and angle0 < angle1):
        raise ValueError("Invalid EchoXFlow sector geometry")
    angles = [angle0, angle1] + [a for a in np.arange(-4, 5) * np.pi / 2 if angle0 <= a <= angle1]
    points = np.array([(r * np.sin(a), r * np.cos(a)) for r in (depth0, depth1) for a in angles])
    low, high = points.min(0), points.max(0)
    side = float((high - low).max())
    origin = ((low + high - side) / 2).tolist()
    return dict(origin_m=origin, side_m=side, depth_start_m=depth0, depth_end_m=depth1,
                angle_start_rad=angle0, angle_end_rad=angle1)


def physical_to_pixels(points, grid, size):
    return (np.asarray(points) - grid["origin_m"]) * ((size - 1) / grid["side_m"])


def pixels_to_physical(points, grid, size):
    return np.asarray(points) * (grid["side_m"] / (size - 1)) + grid["origin_m"]


def inside_sector(points, grid):
    values = np.asarray(points)
    radius = np.linalg.norm(values, axis=-1)
    angle = np.arctan2(values[..., 0], values[..., 1])
    return (np.isfinite(values).all(-1) & (radius >= grid["depth_start_m"])
            & (radius <= grid["depth_end_m"]) & (angle >= grid["angle_start_rad"])
            & (angle <= grid["angle_end_rad"]))


def scan_coordinates(grid, size, source_shape):
    xx, yy = np.meshgrid(np.linspace(grid["origin_m"][0], grid["origin_m"][0] + grid["side_m"], size),
                         np.linspace(grid["origin_m"][1], grid["origin_m"][1] + grid["side_m"], size))
    points = np.stack((xx, yy), -1)
    rows = ((np.linalg.norm(points, axis=-1) - grid["depth_start_m"])
            / (grid["depth_end_m"] - grid["depth_start_m"]) * (source_shape[0] - 1))
    cols = ((np.arctan2(xx, yy) - grid["angle_start_rad"])
            / (grid["angle_end_rad"] - grid["angle_start_rad"]) * (source_shape[1] - 1))
    return np.stack((rows, cols)), inside_sector(points, grid)


def myocardium_mask(points_px, size, sector):
    """Rasterize the four strips between the five wall layers; remove the cavity."""
    points = np.asarray(points_px)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) % 5 or not np.isfinite(points).all():
        raise ValueError("Expected finite five-layer LV contour points")
    grid = points.reshape(-1, 5, 2)
    if len(grid) < 2:
        raise ValueError("Too few wall locations")
    endo, myo = Image.new("1", (size, size)), Image.new("1", (size, size))
    ImageDraw.Draw(endo).polygon([tuple(point) for point in grid[:, 0]], fill=1)
    draw = ImageDraw.Draw(myo)
    for row in range(len(grid) - 1):
        for layer in range(4):
            quad = grid[[row, row + 1, row + 1, row], [layer, layer, layer + 1, layer + 1]]
            draw.polygon([tuple(point) for point in quad], fill=1)
    return np.asarray(myo) & ~np.asarray(endo) & sector


def verify_array(values, metadata):
    if list(values.shape) != metadata["arrays/shape"] or str(values.dtype) != metadata["arrays/dtype"]:
        raise ValueError("Array shape or dtype differs from Croissant metadata")
    expected = metadata["arrays/data_sha256"]
    # EchoXFlow hashes dtype + JSON shape + contiguous array bytes.
    hasher = hashlib.sha256(str(values.dtype).encode())
    hasher.update(json.dumps(list(values.shape)).encode())
    hasher.update(np.ascontiguousarray(values).tobytes())
    if expected and hasher.hexdigest() != expected:
        raise ValueError(f"Source array checksum mismatch: {metadata['arrays/array_id']}")


def index_candidates(cfg):
    root = (ROOT / cfg["data_root"]).resolve()
    metadata_path, splits_path = root / "croissant.json", root / "splits.csv"
    metadata = json.loads(metadata_path.read_text())
    sets = {entry["name"]: entry["data"] for entry in metadata["recordSet"]}
    arrays = {(row["arrays/recording_zarr_path"], row["arrays/array_path"]): row for row in sets["arrays"]}
    with splits_path.open() as handle:
        rows = list(csv.DictReader(handle))
    folds = {row["exam_id"]: int(row["five_fold_split"]) for row in rows}
    if len(folds) != len(rows):
        raise ValueError("Duplicate exam IDs in official split file")
    partitions = dict(train=list(cfg["train_folds"]), validation=[cfg["validation_fold"]], test=[cfg["test_fold"]])
    chosen_folds = [fold for values in partitions.values() for fold in values]
    if len(chosen_folds) != len(set(chosen_folds)) or set(chosen_folds) != set(range(5)):
        raise ValueError("Train, validation and test must partition the five official folds")
    if not cfg["views"] or not set(cfg["views"]) <= {"4ch", "2ch", "3ch"}:
        raise ValueError("Unsupported EchoXFlow views")
    candidates, exclusions, by_hash = {}, [], {}
    for link in sets["recording_links"]:
        view = link["recording_links/panel_role_id"]
        if view not in cfg["views"]:
            continue
        source = link["recording_links/source_recording_zarr_path"]
        video = link["recording_links/linked_recording_zarr_path"]
        link_id = link["recording_links/link_id"]
        frame_path = link["recording_links/linked_frames_array_path"]
        contour_path = f"data/{view}_contour"
        frame_meta = arrays.get((video, frame_path))
        contour_meta = arrays.get((source, contour_path))
        if frame_meta is None:
            exclusions.append(dict(link_id=link_id, reason="unresolved_frame_array"))
            continue
        if contour_meta is None or contour_meta["arrays/shape"][0] == 0:
            exclusions.append(dict(link_id=link_id, reason="empty_contour"))
            continue
        annotation_exam = source.split("/")[1]
        exam = video.split("/")[1]
        fold = folds[exam]
        if folds[annotation_exam] != fold:
            raise ValueError(f"Linked video and annotation cross official folds: {link_id}")
        sha = frame_meta["arrays/data_sha256"]
        if not sha:
            raise ValueError(f"Missing video content hash: {video}")
        if sha in by_hash and by_hash[sha] != fold:
            raise ValueError(f"Video content leaks across folds: {sha}")
        by_hash[sha] = fold
        sg, vg = zarr.open_group(str(root / source), mode="r"), zarr.open_group(str(root / video), mode="r")
        points = sg[contour_path][:]
        verify_array(points, contour_meta)
        if points.ndim != 3 or points.shape[-1] != 2 or points.shape[1] % 5 or not np.isfinite(points).all():
            raise ValueError(f"Invalid contour geometry: {link_id}")
        contour_times = sg[f"timestamps/{view}_contour"][:]
        video_times = vg[link["recording_links/linked_timestamps_array_path"]][:]
        if len(contour_times) != len(points) or len(video_times) != frame_meta["arrays/shape"][0]:
            raise ValueError(f"Timestamp/array length mismatch: {link_id}")
        ci, fi, alignment = align_timestamps(contour_times, video_times)
        if len(fi) < 2:
            exclusions.append(dict(link_id=link_id, reason="fewer_than_two_aligned_frames", alignment=alignment))
            continue
        geometry = dict(vg.attrs)["recording_manifest"]["metadata"]["bmode_geometry"]
        grid = square_geometry(geometry)
        entry = dict(id=f"{exam}_{view}_{sha[:16]}", exam_id=exam, annotation_exam_id=annotation_exam, view=view, fold=fold,
                     split=next(name for name, values in partitions.items() if fold in values),
                     link_id=link_id, source=source, video=video, frame_path=frame_path,
                     contour_path=contour_path, video_timestamps_path=link["recording_links/linked_timestamps_array_path"],
                     contour_indices=ci, frame_indices=fi, alignment=alignment, grid=grid,
                     frame_metadata=frame_meta, contour_metadata=contour_meta,
                     source_metadata_sha256=file_sha(root / source / ".zattrs"),
                     video_metadata_sha256=file_sha(root / video / ".zattrs"),
                     timestamps_sha256=hashlib.sha256(contour_times.tobytes() + video_times.tobytes()).hexdigest())
        key = (exam, view, sha)
        candidates.setdefault(key, []).append(entry)
    records = []
    for entries in candidates.values():
        ordered = sorted(entries, key=lambda item: (-len(item["frame_indices"]), item["source"]))
        records.append(ordered[0])
        exclusions.extend(dict(link_id=item["link_id"], reason="duplicate_video_annotation",
                               selected=ordered[0]["link_id"]) for item in ordered[1:])
    records.sort(key=lambda item: item["id"])
    if not records:
        raise ValueError("No eligible EchoXFlow sequences")
    source_identity = dict(data_revision=cfg["data_revision"], metadata_sha256=file_sha(metadata_path),
                           splits_sha256=file_sha(splits_path), records=records, views=cfg["views"], partitions=partitions)
    return root, records, exclusions, source_identity


def prepare_echoxflow(cfg):
    root, candidates, exclusions, source_identity = index_candidates(cfg)
    preprocessing = {key: cfg[key] for key in ("eval_size", "train_size", "groups", "points_per_group")}
    identity = dict(source=source_identity, preprocessing=preprocessing, source_code_sha256=file_sha(Path(__file__)))
    fingerprint = digest(identity)
    path = runtime.RUN / "echoxflow_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["fingerprint"] != fingerprint:
            raise RuntimeError("EchoXFlow cache/configuration mismatch; use a new run directory or explicitly rebuild the cache")
        for record in existing["records"]:
            group = zarr.open_group(str(runtime.RUN / record["cache"]), mode="r")
            if not group.attrs.get("complete") or group.attrs["fingerprint"] != fingerprint:
                raise RuntimeError(f"Incomplete or incompatible sequence cache: {record['id']}")
        event("echoxflow_cache_reused", **existing["counts"])
        return existing
    size, minimum = cfg["eval_size"], cfg["groups"] * cfg["points_per_group"]
    records = []
    cache = runtime.RUN / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    for number, candidate in enumerate(candidates, 1):
        destination = cache / (candidate["id"] + ".zarr")
        if destination.exists():
            group = zarr.open_group(str(destination), mode="r")
            if not group.attrs.get("complete") or group.attrs["fingerprint"] != fingerprint:
                raise RuntimeError(f"Incomplete or incompatible cache: {destination}")
            record = dict(group.attrs["record"])
            if record.get("excluded_reason"):
                exclusions.append(dict(link_id=candidate["link_id"], reason=record["excluded_reason"]))
            else:
                records.append(record)
            continue
        vg = zarr.open_group(str(root / candidate["video"]), mode="r")
        sg = zarr.open_group(str(root / candidate["source"]), mode="r")
        # Raw Zarr videos use whole-video chunks; decode each video exactly once.
        frames = vg[candidate["frame_path"]][:]
        verify_array(frames, candidate["frame_metadata"])
        if frames.ndim != 3 or frames.dtype != np.uint8:
            raise ValueError("Expected uint8 B-mode video in beamspace")
        points = sg[candidate["contour_path"]][:]
        first, last = candidate["frame_indices"][0], candidate["frame_indices"][-1]
        length, point_count = last - first + 1, points.shape[1] // 5
        guard_disk(cfg, length * size * size * 2)
        maps, sector = scan_coordinates(candidate["grid"], size, frames.shape[1:])
        group = zarr.open_group(str(destination), mode="w")
        group.attrs.update(fingerprint=fingerprint, complete=False)
        images = group.create_dataset("frames", shape=(length, size, size), chunks=(1, size, size), dtype="u1", compressor=compressor)
        masks = group.create_dataset("masks", shape=(length, size, size), chunks=(1, size, size), dtype="u1", compressor=compressor)
        references = np.full((length, point_count, 2), np.nan, np.float32)
        reference_valid = np.zeros((length, point_count), bool)
        annotated = dict(zip(candidate["frame_indices"], candidate["contour_indices"]))
        train_indices, small_masks = [], []
        for local, original in enumerate(range(first, last + 1)):
            image = map_coordinates(frames[original].astype(np.float32), maps, order=1, mode="constant", cval=0)
            image[~sector] = 0
            images[local] = np.clip(np.rint(image), 0, 255).astype(np.uint8)
            if original not in annotated:
                continue
            contour = points[annotated[original]]
            mask = myocardium_mask(physical_to_pixels(contour, candidate["grid"], size), size, sector)
            masks[local] = mask.astype(np.uint8)
            train_mask = np.asarray(Image.fromarray(mask).resize((cfg["train_size"],) * 2, Image.Resampling.NEAREST))
            if train_mask.sum() >= minimum and train_mask[7::14, 7::14].any():
                train_indices.append(local)
            else:
                small_masks.append(original)
            references[local] = contour[2::5]
            reference_valid[local] = inside_sector(contour[2::5], candidate["grid"])
        times = vg[candidate["video_timestamps_path"]][first:last + 1]
        group.create_dataset("timestamps", data=times, compressor=compressor)
        group.create_dataset("reference_points_m", data=references, compressor=compressor)
        group.create_dataset("reference_valid", data=reference_valid, compressor=compressor)
        group.create_dataset("sector", data=sector, compressor=compressor)
        record = {key: candidate[key] for key in ("id", "exam_id", "view", "fold", "split", "link_id", "grid", "alignment")}
        record.update(cache=str(destination.relative_to(runtime.RUN)), frames=length, first_video_frame=first,
                      train_indices=train_indices, small_mask_video_frames=small_masks, reference_points=point_count,
                      valid_reference_positions=int(reference_valid.sum()), video_sha256=candidate["frame_metadata"]["arrays/data_sha256"])
        if len(train_indices) < 2 or not (reference_valid.sum(0) >= 2).any():
            record["excluded_reason"] = "insufficient_valid_masks_or_reference_positions"
            exclusions.append(dict(link_id=candidate["link_id"], reason=record["excluded_reason"]))
        else:
            records.append(record)
        group.attrs.update(record=record, complete=True)
        if number % 10 == 0 or number == len(candidates):
            event("echoxflow_prepared", completed=number, total=len(candidates), accepted=len(records))
        del frames, group
    counts = {split: sum(record["split"] == split for record in records) for split in ("train", "validation", "test")}
    if not all(counts.values()):
        raise ValueError(f"Empty EchoXFlow partition: {counts}")
    manifest = dict(fingerprint=fingerprint, identity=identity, records=records, exclusions=exclusions,
                    counts=counts, reference_kind="Software-tracked middle-wall contour trajectories; no manual point correspondence certification")
    atomic_json(path, manifest)
    event("echoxflow_data_indexed", **counts, excluded=len(exclusions))
    return manifest


class EchoXFlowPairs(Dataset):
    def __init__(self, records, cfg, validation=False, run_dir=None):
        if not records:
            raise ValueError("No EchoXFlow pair records")
        self.records, self.cfg, self.validation = records, cfg, validation
        self.run_dir = Path(run_dir or runtime.RUN)

    def __len__(self):
        return (len(self.records) * self.cfg["validation_pairs_per_video"] if self.validation
                else self.cfg["train_steps"] * self.cfg["effective_batch"])

    def __getitem__(self, index):
        rng = np.random.default_rng(stable_seed((self.cfg["seed"], "val" if self.validation else "train", index)))
        record_index = index // self.cfg["validation_pairs_per_video"] if self.validation else int(rng.integers(len(self.records)))
        record = self.records[record_index]
        indices = rng.choice(record["train_indices"], 2, replace=False)
        group = zarr.open_group(str(self.run_dir / record["cache"]), mode="r")
        # Deliberately never read reference_points_m/reference_valid here.
        images = [Image.fromarray(group["frames"][int(i)]).convert("RGB") for i in indices]
        masks = [group["masks"][int(i)].astype(bool) for i in indices]
        pair = prepare_pair(*images, *masks, self.cfg, rng, augment=not self.validation)
        pair["record_id"] = record["id"]
        return pair

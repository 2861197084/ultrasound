"""CPU-only, training-mask preparation for the locked continuous EchoXFlow split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b782e0a4e3dbaadf3c88fa572710bf4254d86143"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def prepare(cfg):
    sys.path.insert(0, str(ROOT / "vendor" / ("echoxflow-" + REVISION) / "src"))
    from echoxflow.scan.contours import build_contour_masks, LV_GROUP_LAYOUT
    from echoxflow.scan.geometry import CartesianGrid, physical_points_to_pixels
    out = Path(cfg["output"]) / "preparation"
    prepared = Path(cfg["prepared"])
    split = json.loads(Path(cfg["split_manifest"]).read_text())["groups"]
    sets = [{r["exam_id"] for r in split[k]} for k in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Examination leakage across splits")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if not manifest["complete"] or len(manifest["samples"]) != 291:
        raise ValueError("Incomplete continuous cohort")
    # Verify every existing image cache; masks are generated only for training.
    checks = []
    for item in manifest["samples"]:
        path = prepared / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"Changed image cache: {path}")
        with np.load(path, allow_pickle=False) as z:
            images = z["images"]
            meta = json.loads(str(z["metadata_json"]))
        if not np.all(np.diff(meta["original_frame_indices"]) == 1):
            raise ValueError("Nonconsecutive frames")
        with np.load(Path(cfg["pairs"]) / "pairs" / (item["sample_id"] + ".npz")) as pair:
            if not np.array_equal(images[[meta["ed_frame"], meta["es_frame"]]], pair["images"]):
                raise ValueError("ED/ES image mismatch")
        checks.append(dict(sample_id=item["sample_id"], sha256=item["sha256"], frames=len(images), ed_es_exact=True))
    records = []
    for n, item in enumerate(split["train"], 1):
        if shutil.disk_usage(ROOT).free < cfg["reserve_gib"] * 1024**3:
            raise RuntimeError("Disk reserve reached")
        sid = item["sample_id"]
        meta = json.loads((prepared / "records" / (sid + ".json")).read_text())["metadata"]
        grid = CartesianGrid(shape=tuple(meta["cartesian_grid"]["shape"]),
            x_range_m=tuple(meta["cartesian_grid"]["x_range_m"]),
            y_range_m=tuple(meta["cartesian_grid"]["y_range_m"]))
        with np.load(prepared / item["path"], allow_pickle=False) as z:
            sector = z["valid_sector"].astype(bool)
        annotation = zarr.open_group(meta["source_zarr_paths"]["annotation_zarr_path"], mode="r")
        contours = np.asarray(annotation["data/4ch_contour"][:])
        digest = hashlib.sha256()
        digest.update(str(contours.dtype).encode())
        digest.update(json.dumps(list(contours.shape)).encode())
        digest.update(np.ascontiguousarray(contours).tobytes())
        contour_hash = digest.hexdigest()
        if contour_hash != meta["verified_source_arrays"]["contours"]["sha256"]:
            raise ValueError(f"Source contour array changed: {sid}")
        masks = np.zeros((item["frames"], 336, 336), dtype=bool)
        invalid = []
        for frame, ci in enumerate(meta["contour_frame_indices"]):
            reason = None
            if ci < 0:
                reason = "no_matched_contour"
            elif not np.isfinite(contours[ci]).all():
                reason = "nonfinite_contour"
            else:
                masks[frame] = build_contour_masks(physical_points_to_pixels(contours[ci], grid),
                    image_shape=(336, 336), group_layout=LV_GROUP_LAYOUT).myo_mask & sector
                if masks[frame].sum() < cfg["groups"] * cfg["points_per_group"]:
                    reason = "fewer_than_144_foreground_pixels"
            if reason:
                invalid.append(dict(frame=frame, reason=reason))
        eligible = np.flatnonzero(masks.sum((1, 2)) >= cfg["groups"] * cfg["points_per_group"])
        if len(eligible) < 2:
            write_json(out / "failure.json", dict(sample_id=sid, invalid=invalid))
            raise ValueError(f"Fewer than two eligible frames: {sid}")
        with np.load(Path(cfg["pairs"]) / "pairs" / (sid + ".npz")) as pair:
            if not np.array_equal(masks[[meta["ed_frame"], meta["es_frame"]]], pair["masks"]):
                raise ValueError(f"ED/ES masks differ: {sid}")
        path = out / "masks" / (sid + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as f:
            np.savez_compressed(f, masks=masks, eligible_frames=eligible)
        temp.replace(path)
        record = dict(sample_id=sid, exam_id=item["exam_id"], frames=item["frames"],
            image_path=str(prepared / item["path"]), image_sha256=item["sha256"],
            mask_path=str(path), mask_sha256=sha(path), eligible_frames=eligible.tolist(),
            invalid_frames=invalid, ed_frame=meta["ed_frame"], es_frame=meta["es_frame"],
            source_contour_array_sha256=contour_hash,
            source_zarr_path=meta["source_zarr_paths"]["annotation_zarr_path"],
            original_frame_indices=meta["original_frame_indices"],
            contour_frame_indices=meta["contour_frame_indices"])
        records.append(record)
        print(f"masks {n}/{len(split['train'])}: {sid} {len(eligible)}/{len(masks)} eligible", flush=True)
    result = dict(complete=True, records=records, all_image_checks=checks,
        split_sha256=sha(cfg["split_manifest"]), scan_revision=REVISION,
        training_videos=len(records), eligible_frames=sum(len(r["eligible_frames"]) for r in records),
        policy="Only image and mask arrays enter training. Spatial contour order rasterizes masks; no cross-frame point correspondence is exposed.")
    write_json(out / "manifest.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    prepare(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()

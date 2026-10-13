"""从固定版本的原始 Zarr 构建 SemiEchoTracker 的 A4C 三点数据。"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.ndimage import median_filter
import zarr
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DATA_REVISION = "d907c536cae16750b9e46f269a27befd4979ebdd"
CODE_REVISION = "b782e0a4e3dbaadf3c88fa572710bf4254d86143"
VENDOR = PROJECT / "vendor" / f"echoxflow-{CODE_REVISION}"
sys.path.insert(0, str(VENDOR / "src"))
from echoxflow.scan.conversion import sector_to_cartesian
from echoxflow.scan.geometry import (
    CartesianGrid, physical_points_to_pixels, sector_geometry_from_mapping,
)


def select_phases(contours, times, qrs):
    """Estimate ED BEFORE ES from one software-annotated cardiac cycle.

    ES = minimum of the 3-frame median-filtered endocardial polygon area;
    ED = maximum preceding ES. QRS proximity is an audit, not an ES label.
    All thresholds only flag review cases and never create clinical truth.
    """
    if contours.ndim != 3 or contours.shape[2] != 2 or contours.shape[1] % 5 or len(contours) < 8:
        raise ValueError("invalid_lv_contour_layout")
    if not np.isfinite(contours).all() or not np.isfinite(times).all() or not np.all(np.diff(times)>0):
        raise ValueError("invalid_contour_coordinates_or_timestamps")
    endo = contours[:, ::5].astype(np.float64)
    x, y = endo[..., 0], endo[..., 1]
    area = abs(np.sum(x*np.roll(y, -1, axis=1)-y*np.roll(x, -1, axis=1), axis=1)*.5)
    if (area <= 0).any():
        raise ValueError("nonpositive_lv_cavity_area")
    smooth = median_filter(area, size=3, mode="nearest")
    es = int(np.argmin(smooth))
    if es < 2 or es >= len(area)-1:
        raise ValueError("no_valid_preceding_ED_and_interior_ES")
    ed = int(np.argmax(smooth[:es]))
    flags = []
    if smooth[ed] <= smooth[es]:
        raise ValueError("no_cavity_area_contraction")
    duration = float(times[-1]-times[0])
    if smooth[ed] < .9 * smooth.max():
        flags.append("preceding_ED_area_below_90pct_cycle_max")
    if smooth[es]/smooth[ed] > .95:
        flags.append("weak_area_change")
    if not .1 <= (times[es]-times[0])/duration <= .8:
        flags.append("ES_outside_central_cycle_window")
    if times[es]-times[ed] < .08:
        flags.append("ED_ES_interval_below_80ms")
    qrs = np.asarray(qrs, dtype=float)
    qrs = np.unique(qrs[np.isfinite(qrs)])
    offset = float(np.min(abs(qrs-times[ed]))) if len(qrs) else None
    rr = float(np.median(np.diff(qrs))) if len(qrs)>1 else None
    if rr is None:
        flags.append("QRS_cycle_unavailable")
    elif offset > .2 * rr:
        flags.append("ED_far_from_QRS")
    return np.array([ed,es]), dict(method="preceding_area_max_then_area_min_median3", label_status="estimated_not_official",
        area_m2=area.tolist(), smoothed_area_m2=smooth.tolist(), contour_times_seconds=times.tolist(),
        ED_contour_index=ed, ES_contour_index=es, ED_ES_interval_seconds=float(times[es]-times[ed]),
        ES_ED_area_ratio=float(smooth[es]/smooth[ed]), ED_nearest_QRS_seconds=offset,
        median_RR_seconds=rr, qrs_timestamps_seconds=qrs.tolist(), flags=flags)

def align_phases(ct, bt, selected):
    if not np.isfinite(bt).all() or not np.all(np.diff(bt)>0):
        raise ValueError("nonmonotonic_image_timestamps")
    tolerance = float(np.median(np.diff(bt))*.5+1e-6)
    distances = abs(ct[selected, None]-bt[None,:])
    bi = distances.argmin(axis=1)
    errors = distances[np.arange(2),bi]
    if (errors>tolerance).any() or bi[0]>=bi[1]:
        raise ValueError("ED_ES_image_timestamp_alignment_failed")
    return bi, errors, tolerance


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    digest = hashlib.sha256(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def read_verified(group, array_path, declaration):
    array = np.asarray(group[array_path][:])
    attrs = next(a for a in group.attrs["arrays"] if a["name"] == array_path)
    expected = declaration.get("arrays/data_sha256") or attrs["data_sha256"]
    if attrs.get("data_sha256") and attrs["data_sha256"] != expected:
        raise ValueError(f"元数据校验和不一致: {array_path}")
    if (list(array.shape) != declaration["arrays/shape"]
            or str(array.dtype) != declaration["arrays/dtype"]
            or array_sha(array) != expected):
        raise ValueError(f"原始数组校验失败: {array_path}")
    return array, expected


def sample_indices(contours, contour_times, video_times):
    """相位决定视频区间；参考标签取每个实际视频帧最近的轮廓。"""
    phases, phase_info = select_phases(contours, contour_times, [])
    endpoints, _, _ = align_phases(contour_times, video_times, phases)
    if endpoints[1] - endpoints[0] + 1 < 10:
        return None
    frames = np.rint(np.linspace(endpoints[0], endpoints[1], 10)).astype(np.int64)
    distances = abs(video_times[frames, None] - contour_times[None, :])
    refs = distances.argmin(axis=1)
    errors = distances[np.arange(10), refs]
    tolerance = float(np.median(np.diff(contour_times)) * .5 + 1e-6)
    if (errors > tolerance).any() or len(np.unique(frames)) != 10:
        raise ValueError("中间帧时间戳关联失败")
    return phases, phase_info, frames, refs, errors, tolerance


def landmark_indices(contours, ed):
    endo = contours[:, ::5]
    bases = [0, endo.shape[1] - 1]
    midpoint = endo[ed, bases].mean(axis=0)
    apex = int(np.argmax(np.linalg.norm(endo[ed] - midpoint, axis=-1)))
    if apex in bases:
        raise ValueError("心尖代理点与基底端点重合")
    bases.sort(key=lambda index: float(endo[ed, index, 0]))
    return np.array([apex, *bases], dtype=np.int64)


def prepare_sample(job, raw, destination, records, arrays):
    link, fold = job
    aid = link["recording_links/source_recording_id"]
    bid = link["recording_links/linked_recording_id"]
    sid = f"{aid}_4ch"
    annotation = records[aid]
    video_record = records[bid]
    if annotation["recordings/exam_id"] != video_record["recordings/exam_id"]:
        return None, dict(sample_id=sid, annotation_recording=aid, video_recording=bid,
                          annotation_exam=annotation["recordings/exam_id"],
                          video_exam=video_record["recordings/exam_id"], fold=fold,
                          reason="annotation_video_exam_mismatch")
    ag = zarr.open_group(str(raw / annotation["recordings/zarr_path"]), mode="r")
    vg = zarr.open_group(str(raw / video_record["recordings/zarr_path"]), mode="r")
    source_hashes = {}

    def read(group, rid, path):
        value, digest = read_verified(group, path, arrays[rid + "/" + path])
        source_hashes[rid + "/" + path] = digest
        return value

    contours = read(ag, aid, "data/4ch_contour")
    ct = read(ag, aid, "timestamps/4ch_contour")
    time_path = link["recording_links/linked_timestamps_array_path"]
    frame_path = link["recording_links/linked_frames_array_path"]
    bt = read(vg, bid, time_path)
    identity = dict(sample_id=sid, exam_id=annotation["recordings/exam_id"],
                    annotation_recording=aid, video_recording=bid, fold=fold)
    selection = sample_indices(contours, ct, bt)
    if selection is None:
        return None, dict(identity, reason="fewer_than_10_real_frames")
    phases, phase_info, frame_ids, ref_ids, time_errors, tolerance = selection
    points = landmark_indices(contours, int(phases[0]))
    xy_m = contours[ref_ids][:, ::5][:, points].astype(np.float64)
    panel = next(p for p in ag.attrs["recording_manifest"]["linked_panels"]
                 if p["semantic_id"] == "4ch")
    if panel["linked_recording"]["recording_id"] != bid:
        raise ValueError(f"面板视频关联不一致: {sid}")
    shape = arrays[bid + "/" + frame_path]["arrays/shape"]
    geometry = sector_geometry_from_mapping(panel["geometry"], grid_shape=tuple(shape[1:]))
    natural = CartesianGrid.from_sector_height(geometry, 224)
    xr, yr = natural.x_range_m, natural.y_range_m
    side = max(xr[1] - xr[0], yr[1] - yr[0])
    xc, yc = sum(xr) / 2, sum(yr) / 2
    grid = CartesianGrid((224, 224), (xc-side/2, xc+side/2), (yc-side/2, yc+side/2))
    xy = physical_points_to_pixels(xy_m, grid).reshape(10, 3, 2)
    radius, angle = np.linalg.norm(xy_m, axis=-1), np.arctan2(xy_m[..., 0], xy_m[..., 1])
    if (not np.isfinite(xy).all() or xy.min() < 0 or xy.max() > 223
            or np.any(radius < geometry.depth_start_m) or np.any(radius > geometry.depth_end_m)
            or np.any(angle < geometry.angle_start_rad) or np.any(angle > geometry.angle_end_rad)):
        raise ValueError(f"关键点超出成像区域: {sid}")
    native_to_mm = np.array([[grid.spacing_m[0]*1000, 0, grid.x_range_m[0]*1000],
                            [0, grid.spacing_m[1]*1000, grid.y_range_m[0]*1000],
                            [0, 0, 1]], dtype=np.float64)
    restored_m = (xy @ native_to_mm[:2, :2].T + native_to_mm[:2, 2]) / 1000
    if not np.allclose(restored_m, xy_m, atol=1e-7, rtol=0):
        raise ValueError(f"物理坐标往返转换失败: {sid}")
    raw_frames = read(vg, bid, frame_path)
    if raw_frames.dtype != np.uint8 or raw_frames.ndim != 3 or len(raw_frames) != len(bt):
        raise ValueError(f"不支持的视频数组: {sid}, {raw_frames.shape}, {raw_frames.dtype}")
    images = np.stack([sector_to_cartesian(raw_frames[i], geometry, grid=grid).data
                       for i in frame_ids])
    if np.any(images.max(axis=(1, 2)) == 0):
        raise ValueError(f"空视频帧: {sid}")
    split = "train" if fold in (0, 1, 2) else "val" if fold == 3 else "test"
    label_ids = np.array([0, 9], dtype=np.int64) if split == "train" else np.arange(10)
    path = destination / "selected_data" / sid / "data.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path, video=np.repeat(images[..., None], 3, axis=-1),
        landmarks=xy[label_ids], valid_mask=np.ones((len(label_ids), 3), dtype=bool),
        label_frame_indices=label_ids, frame_indices=frame_ids,
        timestamps_seconds=bt[frame_ids], native_to_mm=native_to_mm,
    )
    record = dict(identity, split=split, data_path=str(path.relative_to(destination)),
                  data_sha256=file_sha(path), source_hashes=source_hashes,
                  image_sha256=source_hashes[bid + "/" + frame_path],
                  phase_selection=phase_info, anchor_endo_indices=points.tolist(),
                  bmode_frame_indices=frame_ids.tolist(), contour_frame_indices=ref_ids.tolist(),
                  timestamp_errors_seconds=time_errors.tolist(), timestamp_tolerance_seconds=tolerance,
                  grid=dict(shape=[224, 224], x_range_m=grid.x_range_m, y_range_m=grid.y_range_m),
                  landmark_frames=len(label_ids))
    return record, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/echoxflow/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/semiechotracker/echoxflow")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖已有派生数据: {args.output}")
    catalog = json.loads((args.raw / "croissant.json").read_text())
    if DATA_REVISION not in catalog["description"]:
        raise ValueError("原始数据目录不是预期的固定版本")
    sets = {r["@id"]: r["data"] for r in catalog["recordSet"]}
    records = {r["recordings/recording_id"]: r for r in sets["recordings"]}
    arrays = {a["arrays/array_id"]: a for a in sets["arrays"]}
    with (args.raw / "splits.csv").open() as stream:
        folds = {r["exam_id"]: int(r["five_fold_split"]) for r in csv.DictReader(stream)}
    grouped, excluded = defaultdict(list), []
    for link in sets["recording_links"]:
        if link["recording_links/panel_role_id"] != "4ch":
            continue
        frame_id = link["recording_links/linked_frame_array_id"]
        if frame_id not in arrays:
            excluded.append(dict(link_id=link["recording_links/link_id"], reason="unresolved_multiplane_link"))
        else:
            digest = arrays[frame_id]["arrays/data_sha256"]
            if not digest:
                raise ValueError(f"视频缺少官方内容哈希: {frame_id}")
            grouped[digest].append(link)
    jobs = []
    for links in grouped.values():
        links.sort(key=lambda l: l["recording_links/source_recording_id"])
        group_folds = {folds[records[l["recording_links/source_recording_id"]]["recordings/exam_id"]]
                       for l in links}
        if len(group_folds) != 1:
            raise ValueError("相同视频内容跨官方折号")
        jobs.append((links[0], group_folds.pop()))
        excluded.extend(dict(link_id=l["recording_links/link_id"], reason="duplicate_video_content",
                             selected_link=links[0]["recording_links/link_id"]) for l in links[1:])
    jobs.sort(key=lambda job: job[0]["recording_links/source_recording_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".echoxflow-", dir=args.output.parent) as temporary:
        destination = Path(temporary)
        prepared = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(lambda job: prepare_sample(job, args.raw, destination, records, arrays), jobs)
            for record, rejection in tqdm(results, total=len(jobs), desc="EchoXFlow 校验与转换", mininterval=10):
                if rejection is not None:
                    excluded.append(rejection)
                else:
                    prepared.append(record)
        exam_sets = {s: {r["exam_id"] for r in prepared if r["split"] == s}
                     for s in ("train", "val", "test")}
        if any(exam_sets[a] & exam_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("检查身份跨训练、验证或测试集合")
        split_hashes = {}
        for split in exam_sets:
            path = destination / f"{split}.txt"
            path.write_text("".join(r["sample_id"] + "\n" for r in prepared if r["split"] == split))
            split_hashes[split] = file_sha(path)
        manifest = dict(
            schema=1, complete=True, dataset="EchoXFlow A4C", data_revision=DATA_REVISION,
            geometry_code_revision=CODE_REVISION, raw_root=str(args.raw.resolve()),
            source_metadata_sha256={name: file_sha(args.raw / name) for name in ("croissant.json", "splits.csv", "LICENSE")},
            preparation_code_sha256=file_sha(__file__),
            phase_code_sha256=file_sha(__file__),
            counts=dict(Counter(r["split"] for r in prepared)),
            exam_counts={s: len(ids) for s, ids in exam_sets.items()}, split_sha256=split_hashes,
            protocol="10 real ED-to-ES frames; train endpoint labels only; fixed ED endocardial indices",
            point_order=["apex_proxy", "base_image_left", "base_image_right"],
            reference="EchoXFlow software contours; estimated ED/ES; not expert anatomical landmark annotations",
            samples=prepared, exclusions=excluded,
        )
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        destination.rename(args.output)
    print(json.dumps(dict(output=str(args.output), counts=manifest["counts"], exam_counts=manifest["exam_counts"],
                          exclusions=dict(Counter(r["reason"] for r in excluded))), ensure_ascii=False))


if __name__ == "__main__":
    main()

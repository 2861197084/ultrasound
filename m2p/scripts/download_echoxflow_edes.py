"""Build a compact, provenance-preserving EchoXFlow LV ED/ES-estimate dataset.

Only two image frames per unique linked video/view are retained. Full video
compression blocks are decoded in memory, never stored as new full videos.
ED/ES are estimates from software contours, NOT official phase annotations.
Run with .venv-data/bin/python scripts/download_echoxflow_edes.py --workers 6.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
import threading
import time

import numcodecs
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

from download_echoxflow_raw import RemoteTar, safe_name

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/echoxflow_edes"
LOCAL = ROOT / "data/echoxflow_a4c100"
REVISION = "d907c536cae16750b9e46f269a27befd4979ebdd"
CODE_REVISION = "b782e0a4e3dbaadf3c88fa572710bf4254d86143"
sys.path.insert(0, str(ROOT / "vendor" / ("echoxflow-" + CODE_REVISION) / "src"))
from echoxflow.scan.contours import LV_GROUP_LAYOUT, build_contour_masks
from echoxflow.scan.conversion import sector_to_cartesian
from echoxflow.scan.geometry import CartesianGrid, physical_points_to_pixels, sector_geometry_from_mapping

MiB, GiB = 1024**2, 1024**3
PRINT_LOCK = threading.Lock()
SCHEMA = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def event(kind, **values):
    row = dict(time=datetime.now(timezone.utc).isoformat(), event=kind, **values)
    with PRINT_LOCK:
        with (OUT / "download.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if kind not in {"array_verified", "range_transfer"}:
            print(json.dumps(row, ensure_ascii=False), flush=True)


def array_digest(value):
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def build_plan():
    source = OUT / "source"
    catalog = json.loads((source / "croissant.json").read_text(encoding="utf-8"))
    sets = {r["@id"]: r["data"] for r in catalog["recordSet"]}
    records = {r["recordings/recording_id"]: r for r in sets["recordings"]}
    arrays = {a["arrays/array_id"]: a for a in sets["arrays"]}
    folds = {r["exam_id"]: int(r["five_fold_split"]) for r in csv.DictReader((source / "splits.csv").open())}
    links = defaultdict(list)
    for link in sets["recording_links"]:
        links[(link["recording_links/source_recording_id"], link["recording_links/panel_role_id"])].append(link)
    eligible, excluded = [], []
    raw_count = Counter()
    for ann in sets["arrays"]:
        ap = ann["arrays/array_path"]
        if ap not in {f"data/{v}_contour" for v in ("2ch", "3ch", "4ch")}:
            continue
        rid = ann["arrays/recording_id"]
        rec = records[rid]
        if "2d_left_ventricular_strain" not in rec["recordings/content_types"]:
            continue
        view = ap.split("/")[1].split("_")[0]
        raw_count[view] += 1
        identity = dict(annotation_recording=rid, annotation_exam=rec["recordings/exam_id"], view=view)
        matches = links[(rid, view)]
        if len(matches) != 1:
            excluded.append(dict(identity, reason="missing_or_ambiguous_video_link", count=len(matches)))
            continue
        link = matches[0]
        bid = link["recording_links/linked_recording_id"]
        bpath = link["recording_links/linked_frames_array_path"]
        btpath = link["recording_links/linked_timestamps_array_path"]
        br = records.get(bid)
        frame = arrays.get(bid + "/" + bpath)
        if not br or not frame:
            available = br.get("recordings/array_paths", []) if br else []
            planes = [p for p in available if p.startswith("data/2d_brightness_mode_")]
            reason = "unresolved_multiplane_view_mapping" if planes else "linked_image_array_unavailable"
            excluded.append(dict(identity, reason=reason, linked_recording=bid, available_plane_arrays=planes))
            continue
        ctpath = f"timestamps/{view}_contour"
        required = [bid + "/" + btpath, rid + "/" + ctpath]
        if len(frame["arrays/shape"]) != 3 or any(p not in arrays for p in required):
            excluded.append(dict(identity, reason="unsupported_image_layout_or_missing_timestamps", shape=frame["arrays/shape"]))
            continue
        be, ae = br["recordings/exam_id"], rec["recordings/exam_id"]
        if folds[be] != folds[ae]:
            excluded.append(dict(identity, reason="cross_fold_link", bmode_exam=be))
            continue
        if min(ann["arrays/shape"][0], frame["arrays/shape"][0]) < 8:
            excluded.append(dict(identity, reason="fewer_than_eight_frames"))
            continue
        info = {"frames": frame, "frame_times": arrays[required[0]], "contours": ann, "contour_times": arrays[required[1]]}
        qp = rid + f"/timestamps/{view}_ecg_qrs"
        if qp in arrays:
            info["qrs_times"] = arrays[qp]
        key = frame["arrays/data_sha256"] + ":" + view
        sid = view + "_" + hashlib.sha256(key.encode()).hexdigest()[:20]
        eligible.append(dict(identity, sample_id=sid, bmode_recording=bid, bmode_exam=be,
                             fold=folds[be], bmode_hash=frame["arrays/data_sha256"], arrays=info,
                             annotation_zarr_path=rec["recordings/zarr_path"], bmode_zarr_path=br["recordings/zarr_path"]))
    groups = defaultdict(list)
    hash_folds = defaultdict(set)
    for row in eligible:
        groups[row["sample_id"]].append(row)
        hash_folds[row["bmode_hash"]].add(row["fold"])
    jobs, duplicates = [], []
    for sid, rows in sorted(groups.items()):
        if len(hash_folds[rows[0]["bmode_hash"]]) > 1:
            excluded.extend(dict(annotation_recording=r["annotation_recording"], view=r["view"], reason="duplicate_image_content_across_folds") for r in rows)
            continue
        rows.sort(key=lambda r: (not (LOCAL / r["annotation_zarr_path"] / ".zattrs").exists(), r["annotation_recording"], r["bmode_recording"]))
        jobs.append(dict(sample_id=sid, view=rows[0]["view"], alternatives=rows))
        duplicates.extend(dict(sample_id=sid, annotation_recording=r["annotation_recording"], bmode_recording=r["bmode_recording"]) for r in rows[1:])
    plan = dict(schema=SCHEMA, dataset="Ahus-AIM/EchoXFlow", revision=REVISION, official_code_revision=CODE_REVISION,
                raw_lv_contour_counts=dict(raw_count), linked_candidates=len(eligible), unique_video_view_pairs=len(jobs),
                counts_by_view=dict(Counter(j["view"] for j in jobs)), duplicates=duplicates, metadata_exclusions=excluded, jobs=jobs,
                selection="All resolvable LV 2ch/3ch/4ch contour-image links; deduplicate image hash plus view, preserve official folds.",
                phase_label_source="estimated_from_software_contours_with_QRS_audit; not official ED/ES annotations")
    atomic_json(OUT / "plan.json", plan)
    event("plan_ready", **{k: plan[k] for k in ("raw_lv_contour_counts", "linked_candidates", "unique_video_view_pairs", "counts_by_view")},
          metadata_exclusions=len(excluded), duplicate_links=len(duplicates))
    return plan


class Provider:
    def __init__(self, reserve_gib=35):
        self.sizes = {Path(r["path"]).stem: r["size"] for r in json.loads((OUT / "source/hf_exam_files.json").read_text()) if r["path"].endswith(".tar")}
        self.reserve_gib = reserve_gib
        self.lock = threading.Lock()
        self.exam_locks = defaultdict(threading.Lock)
        self.indices = {}
        self.attrs_cache = {}
        self.tls = threading.local()
        self.network_bytes, self.local_bytes, self.requests = 0, 0, 0

    def reserve(self):
        if shutil.disk_usage(OUT).free < self.reserve_gib * GiB:
            raise RuntimeError("disk_reserve_reached")

    def remote(self, exam):
        if not hasattr(self.tls, "remotes"):
            self.tls.remotes = OrderedDict()
        if exam not in self.tls.remotes:
            url = f"https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/resolve/{REVISION}/exams/{exam}.tar"
            provider = self

            class CountedRemote(RemoteTar):
                def fetch(self, start, end):
                    old_bytes, old_requests = self.network_bytes, self.requests_count
                    try:
                        return super().fetch(start, end)
                    finally:
                        with provider.lock:
                            provider.network_bytes += self.network_bytes-old_bytes
                            provider.requests += self.requests_count-old_requests

            self.tls.remotes[exam] = CountedRemote(url, self.sizes[exam], 64)
            if len(self.tls.remotes) > 8:
                _, old = self.tls.remotes.popitem(last=False)
                old.close()
        self.tls.remotes.move_to_end(exam)
        return self.tls.remotes[exam]

    def index(self, exam):
        with self.lock:
            lock = self.exam_locks[exam]
        with lock:
            if exam in self.indices:
                return self.indices[exam]
            path = OUT / "source/tar_indices" / (exam + ".json")
            if path.exists():
                cached = json.loads(path.read_text())
                if cached["revision"] != REVISION or cached["archive_bytes"] != self.sizes[exam]:
                    raise ValueError("stale_tar_index")
                entries = cached["members"]
            else:
                self.reserve()
                remote = self.remote(exam)
                start_bytes, start_requests = remote.network_bytes, remote.requests_count
                event("index_started", exam=exam)
                entries = []
                started = time.monotonic()
                last_progress = started
                with tarfile.open(fileobj=remote, mode="r:") as tf:
                    for member in tf:
                        safe_name(member.name, exam)
                        if member.isfile():
                            if member.offset_data + member.size > remote.size:
                                raise ValueError("tar_member_outside_archive")
                            entries.append(dict(name=member.name, size=member.size, offset=member.offset_data))
                        elif not member.isdir():
                            raise ValueError("unexpected_tar_link")
                        if time.monotonic()-last_progress > 30:
                            event("index_progress", exam=exam, archive_fraction=round(remote.tell()/remote.size,4), files=len(entries), bytes=remote.network_bytes-start_bytes)
                            last_progress=time.monotonic()
                atomic_json(path, dict(revision=REVISION, archive_bytes=self.sizes[exam], members=entries))
                event("index_complete", exam=exam, files=len(entries), seconds=round(time.monotonic()-started, 2), bytes=remote.network_bytes-start_bytes)
            self.indices[exam] = {e["name"]: e for e in entries}
            return self.indices[exam]

    def read(self, relative):
        path = LOCAL / relative
        if path.exists():
            data = path.read_bytes()
            with self.lock:
                self.local_bytes += len(data)
            return data
        exam = Path(relative).parts[1]
        entry = self.index(exam).get(relative)
        if entry is None:
            raise ValueError("missing_tar_member:" + relative)
        remote = self.remote(exam)
        start_bytes, start_requests = remote.network_bytes, remote.requests_count
        data = bytearray()
        a, stop = entry["offset"], entry["offset"] + entry["size"]
        for begin in range(a, stop, 4 * MiB):
            self.reserve()
            data.extend(remote.fetch(begin, min(stop, begin + 4 * MiB) - 1))
        return bytes(data)

    def attrs(self, zarr_path):
        with self.lock:
            cached = self.attrs_cache.get(zarr_path)
        if cached is not None:
            return cached
        value = json.loads(self.read(zarr_path + "/.zattrs"))
        with self.lock:
            self.attrs_cache[zarr_path] = value
        return value

    def array(self, info):
        prefix = info["arrays/recording_zarr_path"] + "/" + info["arrays/array_path"]
        meta = json.loads(self.read(prefix + "/.zarray"))
        shape, chunks, dtype = tuple(meta["shape"]), tuple(meta["chunks"]), np.dtype(meta["dtype"])
        if list(shape) != info["arrays/shape"] or str(dtype) != info["arrays/dtype"]:
            raise ValueError("array_metadata_mismatch:" + prefix)
        if math.prod(shape) * dtype.itemsize > 512 * MiB:
            raise ValueError("array_exceeds_512MiB_memory_limit:" + prefix)
        if meta.get("filters"):
            raise ValueError("unsupported_array_filters:" + prefix)
        out = np.empty(shape, dtype=dtype)
        codec = numcodecs.get_codec(meta["compressor"]) if meta.get("compressor") else None
        separator = meta.get("dimension_separator", ".")
        for coords in itertools.product(*(range(math.ceil(s / c)) for s, c in zip(shape, chunks))):
            chunk_bytes = self.read(prefix + "/" + separator.join(map(str, coords)))
            decoded = codec.decode(chunk_bytes) if codec else chunk_bytes
            chunk = np.frombuffer(decoded, dtype=dtype).reshape(chunks, order=meta.get("order", "C"))
            slices = tuple(slice(i*c, min((i+1)*c, s)) for i,c,s in zip(coords,chunks,shape))
            out[slices] = chunk[tuple(slice(0, x.stop-x.start) for x in slices)]
        expected = info.get("arrays/data_sha256")
        if not expected:
            attrs = self.attrs(info["arrays/recording_zarr_path"])
            expected = next((a.get("data_sha256") for a in attrs.get("arrays", [])
                             if a["name"] == info["arrays/array_path"]), None)
        if not expected or array_digest(out) != expected:
            raise ValueError("array_checksum_mismatch:" + prefix)
        info["arrays/data_sha256"] = expected
        event("array_verified", array=info["arrays/array_id"], bytes=out.nbytes)
        return out


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


def convert(row, provider):
    provider.reserve()
    info = row["arrays"]
    attrs = provider.attrs(row["annotation_zarr_path"])
    panel = next(p for p in attrs["recording_manifest"]["linked_panels"] if p["semantic_id"]==row["view"])
    if panel["linked_recording"]["recording_id"] != row["bmode_recording"]:
        raise ValueError("manifest_video_link_mismatch")
    contour = provider.array(info["contours"])
    ct = provider.array(info["contour_times"]).reshape(-1)
    bt = provider.array(info["frame_times"]).reshape(-1)
    qrs = provider.array(info["qrs_times"]).reshape(-1) if "qrs_times" in info else np.array([])
    if len(ct)!=len(contour) or len(bt)!=info["frames"]["arrays/shape"][0]:
        raise ValueError("timestamp_length_mismatch")
    selected, phase = select_phases(contour, ct, qrs)
    bi, errors, tolerance = align_phases(ct, bt, selected)
    geometry = sector_geometry_from_mapping(panel["geometry"], grid_shape=tuple(info["frames"]["arrays/shape"][1:]))
    natural = CartesianGrid.from_sector_height(geometry, 336)
    xr, yr = natural.x_range_m, natural.y_range_m
    side = max(xr[1]-xr[0], yr[1]-yr[0]); xc, yc = sum(xr)/2, sum(yr)/2
    grid = CartesianGrid(shape=(336,336), x_range_m=(xc-side/2,xc+side/2), y_range_m=(yc-side/2,yc+side/2))
    xy = contour[selected]
    radius, angle = np.linalg.norm(xy,axis=-1), np.arctan2(xy[...,0],xy[...,1])
    outside = (radius<geometry.depth_start_m)|(radius>geometry.depth_end_m)|(angle<geometry.angle_start_rad)|(angle>geometry.angle_end_rad)
    fractions = outside.mean(axis=1)
    flags = list(phase["flags"])
    if fractions.max()>.01:
        flags.append("more_than_1pct_selected_contour_points_outside_sector")
    raw = provider.array(info["frames"])
    beamspace = np.ascontiguousarray(raw[bi])
    del raw
    converted = [sector_to_cartesian(frame,geometry,grid=grid) for frame in beamspace]
    masks = [build_contour_masks(physical_points_to_pixels(points,grid), image_shape=(336,336),group_layout=LV_GROUP_LAYOUT) for points in xy]
    sector = converted[0].mask
    images = np.stack([f.data for f in converted])
    myo = np.stack([m.myo_mask & sector for m in masks])
    cavity = np.stack([m.endo_mask & sector for m in masks])
    if min(m.sum() for m in myo)<144 or min(m.sum() for m in cavity)<144:
        raise ValueError("selected_mask_too_small")
    if np.array_equal(images[0],images[1]):
        raise ValueError("selected_image_frames_identical")
    arrays = dict(images=images, masks=myo, cavity_masks=cavity, valid_sector=sector,
                  beamspace_images=beamspace, contour_xy_m=xy,
                  bmode_frame_indices=bi.astype(np.int64), contour_frame_indices=selected.astype(np.int64),
                  image_timestamps_seconds=bt[bi], contour_timestamps_seconds=ct[selected])
    target = OUT / "pairs" / (row["sample_id"]+".npz")
    target.parent.mkdir(exist_ok=True)
    tmp = target.with_suffix(".npz.part")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    with np.load(tmp, allow_pickle=False) as loaded:
        if set(loaded.files)!=set(arrays) or any(not np.array_equal(loaded[k],v) for k,v in arrays.items()):
            raise ValueError("saved_pair_roundtrip_failed")
    tmp.replace(target)
    result = {k:v for k,v in row.items() if k!="arrays"}
    result.update(schema=SCHEMA, revision=REVISION, official_code_revision=CODE_REVISION,
        status="review" if flags else "accepted", flags=flags, path=str(target.relative_to(OUT)),
        bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        phase_order=["ED_estimated","ES_estimated"], phase_selection=phase,
        bmode_frame_indices=bi.tolist(), contour_frame_indices=selected.tolist(),
        image_timestamps_seconds=bt[bi].tolist(), contour_timestamps_seconds=ct[selected].tolist(),
        timestamp_errors_seconds=errors.tolist(), timestamp_tolerance_seconds=tolerance,
        selected_contour_outside_sector_fraction=fractions.tolist(), myocardium_pixels=myo.sum(axis=(1,2)).tolist(),
        geometry=panel["geometry"], cartesian_grid=dict(shape=[336,336],x_range_m=grid.x_range_m,y_range_m=grid.y_range_m,pixel_spacing_m=side/335),
        verified_source_arrays={k:dict(array_id=v["arrays/array_id"],sha256=v["arrays/data_sha256"]) for k,v in info.items()},
        label_note="Software-derived myocardium/cavity masks; estimated phases; no independent material-point trajectories.")
    atomic_json(OUT / "records" / (row["sample_id"]+".json"), result)
    return result


def process_job(job, provider, retry_failed=False):
    sid = job["sample_id"]
    record = OUT / "records" / (sid+".json")
    if record.exists():
        old = json.loads(record.read_text())
        if old.get("schema")==SCHEMA and old.get("revision")==REVISION:
            if old["status"] in {"accepted","review"}:
                p=OUT/old["path"]
                if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==old["sha256"]:
                    return old
            elif old["status"]=="excluded" and not retry_failed:
                return old
    errors=[]
    for row in job["alternatives"]:
        try:
            result=convert(row,provider)
            result["alternative_links"]=len(job["alternatives"])
            return result
        except Exception as error:
            detail=dict(annotation_recording=row["annotation_recording"],error_type=type(error).__name__,reason=str(error))
            errors.append(detail)
            event("candidate_error",sample_id=sid,**detail)
    status="failed" if any(e["error_type"] not in {"ValueError","StopIteration"} for e in errors) else "excluded"
    result=dict(sample_id=sid,view=job["view"],schema=SCHEMA,revision=REVISION,status=status,attempts=errors)
    atomic_json(record,result)
    return result


def summarize(plan, results, provider, complete=False):
    counts=Counter(r["status"] for r in results)
    saved=[r for r in results if r["status"] in {"accepted","review"}]
    summary=dict(schema=SCHEMA,revision=REVISION,phase_labels="estimated_not_official",complete=complete,
        total_candidates=plan["unique_video_view_pairs"], processed=len(results),status_counts=dict(counts),
        saved_pairs=len(saved),saved_frames=2*len(saved),saved_pair_bytes=sum(r["bytes"] for r in saved),
        saved_by_view=dict(Counter(r["view"] for r in saved)),accepted_by_fold=dict(Counter(str(r["fold"]) for r in saved if r["status"]=="accepted")),
        network_bytes_this_run=provider.network_bytes,local_bytes_read_this_run=provider.local_bytes,http_requests_this_run=provider.requests,
        free_disk_GiB=round(shutil.disk_usage(OUT).free/GiB,2),updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(OUT/"status.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--local-only",action="store_true")
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--reserve-gib",type=float,default=35)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    plan_path=OUT/"plan.json"
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else build_plan()
    if plan["revision"]!=REVISION or plan["schema"]!=SCHEMA:
        raise ValueError("stale_plan")
    if args.plan_only:return
    jobs=plan["jobs"]
    if args.local_only:
        jobs=[j for j in jobs if any((LOCAL/r["bmode_zarr_path"]/r["arrays"]["frames"]["arrays/array_path"]/".zarray").exists()
                                    and (LOCAL/r["annotation_zarr_path"]/".zattrs").exists() for r in j["alternatives"])]
    if args.limit:jobs=jobs[:args.limit]
    jobs=sorted(jobs,key=lambda j: not (OUT/"records"/(j["sample_id"]+".json")).exists())
    provider=Provider(args.reserve_gib)
    results=[]
    event("run_started",jobs=len(jobs),workers=args.workers,local_only=args.local_only)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(process_job,j,provider,args.retry_failed):j["sample_id"] for j in jobs}
        for future in tqdm(as_completed(futures),total=len(futures),desc="EchoXFlow ED/ES",mininterval=10):
            result=future.result();results.append(result)
            event("pair_complete",sample_id=result["sample_id"],view=result["view"],status=result["status"],done=len(results),total=len(jobs))
            summarize(plan,results,provider)
    # Include previously completed shards, e.g. an earlier local smoke run.
    all_results=[json.loads(p.read_text()) for p in sorted((OUT/"records").glob("*.json"))]
    done=len(all_results)==len(plan["jobs"]) and not any(r["status"]=="failed" for r in all_results)
    summary=summarize(plan,all_results,provider,complete=done)
    atomic_json(OUT/"manifest.json",dict(**summary,records=all_results,metadata_exclusions=plan["metadata_exclusions"],duplicate_links=plan["duplicates"]))
    event("run_complete",**summary)
    if any(r["status"]=="failed" for r in results):raise SystemExit(1)


if __name__=="__main__":
    main()

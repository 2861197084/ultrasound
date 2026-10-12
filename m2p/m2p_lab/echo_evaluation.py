"""M2P inference with the locked EchoTracker2 reference and scoring contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from . import runtime as rt

ECHO = rt.ROOT.parent / "EchoTracker2"
sys.path.insert(0, str(ECHO))
from evaluation_data import load_sample
from evaluate import score, score_es, write_csv

WINDOWS = ("full_cycle", "ed_es_sequence", "two_frames")


def sample(cfg, item, window="full_cycle"):
    images, reference, valid, meta = load_sample("echoxflow-sequence", Path(cfg["prepared"]), item)
    q, es = meta["ed_frame"], meta["es_frame"]
    ids = np.flatnonzero(valid[:, q])
    reference, valid = reference[ids], valid[ids]
    if window == "full_cycle":
        frames = np.arange(len(images))
    elif window == "ed_es_sequence":
        frames = np.arange(q, es + 1)
    elif window == "two_frames":
        frames = np.array([q, es])
    else:
        raise ValueError(window)
    meta = dict(meta, ed_frame=int(np.flatnonzero(frames == q)[0]),
                es_frame=int(np.flatnonzero(frames == es)[0]), selected_cycle_frames=frames.tolist(),
                window=window, source_sha256=item["sha256"])
    return images[frames], reference[:, frames], valid[:, frames], meta, ids


@torch.inference_mode()
def infer(model, images, query_xy_normalized, query_frame, raw=False):
    # Only images and ED coordinates enter model.track. No target references/masks.
    video = torch.from_numpy(np.ascontiguousarray(images)).float().unsqueeze(1).repeat(1, 3, 1, 1) / 255
    query_normalized = np.column_stack((np.full(len(query_xy_normalized), query_frame), query_xy_normalized)).astype(np.float32)
    query_pixels = query_normalized.copy()
    query_pixels[:, 1:] *= images.shape[-1] - 1
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prediction, similarity = model.track(video, torch.from_numpy(query_pixels), raw=raw,
                                         chunk_size=model.config["inference_chunk_size"])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    prediction = prediction.numpy().transpose(1, 0, 2).astype(np.float64) / (images.shape[-1] - 1)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite model predictions")
    timing = dict(inference_seconds=seconds, peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    return prediction, similarity.numpy().T, query_normalized, timing


def metrics(prediction, reference, valid, meta, label, timing=None):
    q, es = meta["ed_frame"], meta["es_frame"]
    active = valid.copy()
    active[:, q] = False
    row = {k: meta[k] for k in ("sample_id", "exam_id", "status", "view", "fold")}
    row.update(model=label, window=meta["window"], frames=reference.shape[1], query_frame=q, es_frame=es,
               queried_points=len(reference), valid_target_observations=int(active.sum()),
               valid_es_observations=int(valid[:, es].sum()),
               **(timing or {}), **score(prediction, reference, valid, q, meta["span_m"],
                                        endpoint=meta["window"] == "two_frames"),
               **score_es(prediction, reference, valid, q, es, meta["span_m"]))
    if meta["window"] == "full_cycle":
        other = valid.copy()
        other[:, [q, es]] = False
        if other.any():
            row.update({"other_frames_" + k: v for k, v in score(prediction, reference, other, q, meta["span_m"]).items()})
    return row


def validate_points(model, records, cfg):
    model.eval()
    rows = []
    for item in records:
        images, ref, valid, meta, _ = sample(cfg, item)
        pred, _, _, timing = infer(model, images, ref[:, meta["ed_frame"]], meta["ed_frame"])
        rows.append(metrics(pred, ref, valid, meta, "m2p", timing))
    model.train()
    return dict(es_l2_epe_px=float(np.mean([r["es_l2_epe_px"] for r in rows])), videos=len(rows), per_video=rows)


def evaluate_model(model, label, cfg, records, raw=False):
    destination = rt.RUN / "evaluation" / label
    all_rows, failures = [], []
    model.eval()
    for window in WINDOWS:
        folder = destination / window / "predictions"
        folder.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(records):
            sid = item["sample_id"]
            path = folder / (sid + ".npz")
            images, ref, valid, meta, ids = sample(cfg, item, window)
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    pred = z["prediction_xy_normalized"]
                    if not np.array_equal(z["reference_xy_normalized"], ref, equal_nan=True) or not np.array_equal(z["reference_valid"], valid):
                        raise ValueError("Saved prediction reference changed")
                    timing = json.loads(str(z["timing_json"]))
            else:
                try:
                    if index == 0:
                        infer(model, images, ref[:, meta["ed_frame"]], meta["ed_frame"], raw)
                    pred, similarity, query, timing = infer(model, images, ref[:, meta["ed_frame"]], meta["ed_frame"], raw)
                except Exception as error:
                    failures.append(dict(model=label, window=window, sample_id=sid, reason=repr(error)))
                    rt.atomic_json(destination / "failures.json", failures)
                    raise
                rt.guard_disk(cfg, 4 * 1024**2)
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as f:
                    np.savez_compressed(f, prediction_xy_normalized=pred, reference_xy_normalized=ref,
                        reference_valid=valid, source_point_indices=ids, query_txy=query,
                        uncalibrated_similarity=similarity, metadata_json=np.array(json.dumps(meta)),
                        timing_json=np.array(json.dumps(timing)))
                temp.replace(path)
            all_rows.append(metrics(pred, ref, valid, meta, label, timing))
            if window != "full_cycle":
                with np.load(destination / "full_cycle" / "predictions" / (sid + ".npz")) as z:
                    full = z["prediction_xy_normalized"][:, meta["selected_cycle_frames"]]
                if not np.array_equal(pred, full):
                    raise ValueError(f"Pairwise prediction changes across windows: {label} {sid}")
            if index % 8 == 0 or index == len(records) - 1:
                rt.event("evaluation", model=label, window=window, completed=index+1, total=len(records))
                rt.status("evaluation", model=label, window=window, completed=index+1, total=len(records))
        rt.atomic_json(destination / "per_video.json", all_rows)
    write_csv(destination / "per_video.csv", all_rows)
    rt.atomic_json(destination / "failures.json", failures)
    return all_rows

import importlib
import json
import os

import numpy as np
import pytest
import torch
import zarr

import m2p_lab
from m2p_lab import echoxflow, runtime
from m2p_lab.echoxflow import (EchoXFlowPairs, align_timestamps, physical_to_pixels,
                                pixels_to_physical, square_geometry, inside_sector)
from m2p_lab.evaluation import echo_point_metrics, reference_queries, compare_echoxflow, aggregate_echo_records


def test_alignment_deduplicates_nearest_and_never_extrapolates():
    ci, fi, details = align_timestamps([.02, .08, .10, .32], [0., .1, .2])
    assert ci == [0, 2] and fi == [0, 1]
    assert details["outside_tolerance"] == details["duplicate_matches"] == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        align_timestamps([0., 0.], [0., .1])


def test_physical_square_roundtrip_and_sector_visibility():
    grid = square_geometry(dict(depth_start_m=0., depth_end_m=.1, tilt_rad=0., width_rad=np.pi / 3))
    assert grid["side_m"] == pytest.approx(.1)
    assert grid["origin_m"] == pytest.approx([-.05, 0.])
    points = np.array([[0., .05], [.02, .07], [-.02, .07], [.09, .02]])
    xy = physical_to_pixels(points, grid, 448)
    assert xy[0] == pytest.approx([223.5, 223.5])
    assert np.allclose(pixels_to_physical(xy, grid, 448), points, atol=1e-9)
    assert inside_sector(points, grid).tolist() == [True, True, True, False]


def test_mask_pair_dataset_requires_no_reference_trajectories(tmp_path):
    cfg = runtime.load_config(runtime.ROOT / "configs/m2p_echoxflow_a4c.json")
    group = zarr.open_group(str(tmp_path / "video.zarr"), mode="w")
    images = np.zeros((2, 448, 448), np.uint8)
    images[:, 50:380, 60:370] = 150
    masks = images > 0
    group.create_dataset("frames", data=images, chunks=(1, 448, 448))
    group.create_dataset("masks", data=masks, chunks=(1, 448, 448))
    records = [dict(id="a4c", cache="video.zarr", train_indices=[0, 1])]
    dataset = EchoXFlowPairs(records, cfg, validation=True, run_dir=tmp_path)
    a, b = dataset[0], dataset[0]
    assert set(a) == {"image0", "image1", "mask0", "mask1", "distance0", "distance1", "queries", "record_id"}
    assert a["queries"].shape == (12, 12, 2)
    assert all(torch.equal(a[key], b[key]) for key in a if torch.is_tensor(a[key]))
    xy = a["queries"].long()
    assert a["mask0"][xy[..., 1], xy[..., 0]].all()
    assert torch.equal(a["image0"][0], a["image0"][1])


def test_echo_metrics_known_mm_displacement_ignore_query_and_invalid_reference():
    grid = dict(origin_m=[0., 0.], side_m=.256)
    reference = np.full((3, 2, 2), .02, dtype=np.float64)
    prediction = physical_to_pixels(reference, grid, 257)
    prediction[..., 0] += 1.5  # 1 pixel = 1 mm on this physical square.
    prediction[0] += 1000  # Query frames never contribute.
    valid = np.ones((3, 2), bool)
    valid[2, 1] = False
    prediction[2, 1] += 1000
    result = echo_point_metrics(prediction, reference, valid, [0, 0], grid, 257)
    assert result["epe_mm"] == pytest.approx(1.5)
    assert result["pck_1"] == 0 and result["pck_2"] == 100
    assert result["evaluated_positions"] == 3
    prediction[1, 0] = [-100, -100]
    assert echo_point_metrics(prediction, reference, valid, [0, 0], grid, 257)["epe_mm"] > 50
    valid = np.array([[1, 0], [1, 1], [1, 1]], bool)
    queries, ids = reference_queries(reference, valid, grid, 257)
    assert queries[:, 0].tolist() == [0, 1] and ids.tolist() == [0, 1]


@pytest.mark.parametrize("index", [0, 1])
def test_gpu_is_selected_from_configuration_without_import_side_effects(monkeypatch, index):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "prior-device")
    importlib.reload(m2p_lab)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "prior-device"
    calls = []
    def query(args, **kwargs):
        calls.append(args)
        return f"GPU-selected-{index}, 00000000:0{index + 1}:00.0\n"
    monkeypatch.setattr(runtime.subprocess, "check_output", query)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    result = runtime.select_gpu({"gpu": index})
    assert f"--id={index}" in calls[0]
    assert result["physical_device"] == index
    assert os.environ["CUDA_VISIBLE_DEVICES"] == f"GPU-selected-{index}"
    assert runtime.scientific_config({"gpu": 0, "seed": 1}) == runtime.scientific_config({"gpu": 1, "seed": 1})


def test_cache_identity_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUN", tmp_path)
    monkeypatch.setattr(echoxflow, "index_candidates", lambda cfg: (tmp_path, [], [], {"version": 1}))
    (tmp_path / "echoxflow_manifest.json").write_text(json.dumps({"fingerprint": "different"}))
    with pytest.raises(RuntimeError, match="cache/configuration mismatch"):
        echoxflow.prepare_echoxflow(dict(eval_size=448, train_size=336, groups=12, points_per_group=12))


def test_comparison_weights_exams_equally_and_bootstraps_paired_changes():
    keys = ("epe_mm", "pck_average", "pck_1", "pck_2", "pck_4", "pck_8", "pck_16")
    rows = [dict(id=str(i), exam_id=exam, evaluated_positions=10, possible_positions=12, query_points=2,
                 **dict.fromkeys(keys, value)) for i, (exam, value) in enumerate([("a", 2.), ("a", 4.), ("b", 9.)])]
    before = aggregate_echo_records(rows)
    after_rows = [dict(row, **{key: row[key] + 1 for key in keys}) for row in rows]
    after = aggregate_echo_records(after_rows)
    assert before["metrics"]["epe_mm"] == 6.0
    def package(aggregate, sequences):
        return dict(summary={k: v for k, v in aggregate.items() if k != "exams"}, exams=aggregate["exams"],
                    sequences=sequences, data_fingerprint="same", reference_kind="test")
    result = compare_echoxflow(package(before, rows), package(after, after_rows), dict(seed=1, bootstrap_samples=200))
    assert result["paired_changes"]["epe_mm"]["ci95"] == [1., 1.]

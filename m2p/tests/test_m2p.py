import json
import math
from pathlib import Path
import tempfile

import m2p_lab  # Enforce physical GPU 1 before importing torch.
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import default_collate

from m2p_lab.data import prepare_pair
from m2p_lab.data import build_vos_records
from m2p_lab.evaluation import official_metrics
from m2p_lab.geometry import (boundary_distances, grouped_queries, pixel_grid,
                              sample_pixels, similarity_transform, teacher_pseudopoints)
from m2p_lab.model import LoRALinear, M2P, local_soft_argmax, objective
from m2p_lab.runtime import RUN, check_gpu, load_config, save_best, seed_everything


def test_pixel_sampling_and_coordinate_gradients():
    grid = pixel_grid(20, 30)
    image = (grid[..., 0] + 2 * grid[..., 1])[None, None]
    points = torch.tensor([[[4.25, 5.75], [18.5, 11.125]]], requires_grad=True)
    values = sample_pixels(image, points)[..., 0]
    assert torch.allclose(values, points[..., 0] + 2 * points[..., 1], atol=1e-5)
    values.sum().backward()
    assert torch.allclose(points.grad, torch.tensor([[[1., 2.], [1., 2.]]]), atol=1e-5)


def test_known_similarity_and_degeneracy():
    source = torch.tensor([[[2., 3.], [30., 7.], [9., 35.]]])
    theta = 0.3
    rotation = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    target = 1.25 * source @ rotation.T + torch.tensor([7., -5.])
    r, scale, mx, my, valid = similarity_transform(source, target)
    recovered = (source - mx) @ r * scale[:, None, None] + my
    assert valid.all()
    assert torch.allclose(recovered, target, atol=1e-4)
    *_, valid = similarity_transform(source, torch.ones_like(source))
    assert not valid.any()


def test_teacher_pseudopoints_recover_motion():
    source = torch.tensor([[[[0., 0.], [20., 0.], [0., 20.], [10., 8.], [5., 6.]]]])
    target = source * 1.2 + torch.tensor([3., 4.])
    confidence = torch.tensor([[[1., .9, .8, .2, .1]]])
    noisy = target.clone(); noisy[..., 3:, :] += 30
    predicted, valid = teacher_pseudopoints(source, noisy, confidence)
    assert valid.all()
    assert torch.allclose(predicted, target, atol=1e-4)


def test_queries_stay_inside_foreground_and_are_reproducible():
    mask = np.zeros((160, 160), bool)
    mask[20:140, 30:130] = True
    a = grouped_queries(mask, np.random.default_rng(4))
    b = grouped_queries(mask, np.random.default_rng(4))
    assert a.shape == (12, 12, 2)
    assert np.array_equal(a, b)
    assert mask[a[..., 1].astype(int), a[..., 0].astype(int)].all()
    assert len(np.unique(a.reshape(-1, 2), axis=0)) >= 130
    with pytest.raises(ValueError):
        grouped_queries(np.zeros((10, 10), bool), np.random.default_rng(4))


def test_boundary_distance_is_nearest_contour_inside_and_outside():
    mask = np.zeros((21, 21), bool); mask[5:16, 5:16] = True
    d = boundary_distances(mask)
    assert d[5, 10] == 0
    assert d[10, 10] == 5
    assert d[2, 10] == 3


def test_local_argmax_uses_patch_centers_and_has_gradients():
    corr = torch.zeros(1, 1, 24, 24); corr[0, 0, 10, 9] = 1
    logits = torch.zeros_like(corr, requires_grad=True)
    xy, probability = local_soft_argmax(corr, logits)
    assert torch.allclose(xy, torch.tensor([[[133., 147.]]]), atol=1e-4)
    assert torch.allclose(probability.sum((-1, -2)), torch.ones(1, 1))
    xy.sum().backward()
    assert logits.grad.abs().sum() > 0


def test_lora_initially_preserves_backbone():
    layer = torch.nn.Linear(8, 12)
    adapter = LoRALinear(layer, 4, 4, 0.1)
    x = torch.randn(2, 10, 8)
    assert torch.equal(layer(x), adapter(x))
    adapter(x).square().sum().backward()
    assert adapter.b.grad.abs().sum() > 0
    assert layer.weight.grad is None


def test_official_metric_oracle_and_query_exclusion():
    first, metrics = official_metrics()
    tracks = np.array([[[10., 10.], [12., 10.], [14., 10.], [16., 10.]]], np.float32)
    occluded = np.zeros((1, 4), bool)
    sample = first(occluded, tracks, np.zeros((4, 256, 256, 3), np.uint8))
    prediction = tracks[None].copy()
    prediction[:, :, 0] += 100  # Query frame must not count toward localization.
    result = metrics(sample['query_points'], sample['occluded'], sample['target_points'],
                     occluded[None], prediction, 'first')
    assert result['average_pts_within_thresh'][0] == 1
    prediction[:, :, 1:, 0] += 1.1
    result = metrics(sample['query_points'], sample['occluded'], sample['target_points'],
                     occluded[None], prediction, 'first')
    assert result['pts_within_1'][0] == 0
    assert result['pts_within_2'][0] == 1


def test_validation_init_masks_are_not_mistaken_for_dense_annotations(tmp_path):
    root = tmp_path / "valid"
    for kind in ("JPEGImages", "Annotations"):
        (root / kind / "video").mkdir(parents=True)
    for index, oid in (("00000", 1), ("00005", 2)):
        Image.fromarray(np.zeros((20, 20, 3), np.uint8)).save(root / "JPEGImages/video" / (index + ".jpg"))
        mask = np.zeros((20, 20), np.uint8); mask[2:18, 2:18] = oid
        Image.fromarray(mask).save(root / "Annotations/video" / (index + ".png"))
    metadata = {"videos": {"video": {"objects": {str(i): {"frames": ["00000", "00005"]} for i in (1, 2)}}}}
    (root / "meta.json").write_text(json.dumps(metadata))
    records = build_vos_records(tmp_path, "valid", "youtubevos")
    assert all(len(obj['frames']) == 1 for obj in records[0]['objects'])


@pytest.fixture(scope="module")
def cuda_model():
    cfg = load_config()
    check_gpu(cfg)
    seed_everything(cfg['seed'])
    model = M2P(cfg).cuda()
    return model, model.frozen_teacher(), cfg


def example_batch(cfg):
    rng = np.random.default_rng(10)
    pixels = rng.integers(0, 255, (336, 336, 3), dtype=np.uint8)
    mask = np.zeros((336, 336), bool); mask[30:290, 40:280] = True
    image = Image.fromarray(pixels)
    pair = prepare_pair(image, image, mask, mask, cfg, rng, synthetic=True)
    return {k: v.cuda() for k, v in default_collate([pair]).items()}


def test_full_objective_updates_only_adapters(cuda_model):
    model, teacher, cfg = cuda_model
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 9437489
    batch = example_batch(cfg)
    model.train()
    loss, parts = objective(model, teacher, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in parts.values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith('.b'))
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
    model.zero_grad(set_to_none=True)


def test_best_checkpoint_replaces_and_reloads_exact_predictions(cuda_model):
    model, teacher, cfg = cuda_model
    model.eval()
    batch = example_batch(cfg)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model.predict_pair(batch['image0'], batch['image1'], batch['queries'].reshape(1, -1, 2))[0]
    (RUN / 'preflight').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=RUN / 'preflight') as folder:
        path = Path(folder) / 'best.pt'
        save_best(model, path, cfg, 1, {'total': 2.0}, {'test': True})
        save_best(model, path, cfg, 2, {'total': 1.0}, {'test': True})
        assert [p.name for p in Path(folder).iterdir()] == ['best.pt']
        saved = torch.load(path, weights_only=True, map_location='cpu')
        assert saved['step'] == 2
        assert path.stat().st_size < 45 * 1024**2
        assert not any('optimizer' in key for key in saved)
        with torch.no_grad():
            next(p for p in model.parameters() if p.requires_grad).add_(1)
        model.load_adapter(saved['adapter'])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            actual = model.predict_pair(batch['image0'], batch['image1'], batch['queries'].reshape(1, -1, 2))[0]
        assert torch.equal(expected, actual)

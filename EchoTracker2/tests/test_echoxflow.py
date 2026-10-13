"""只覆盖本次实验的坐标、监督、累积和恢复风险。"""
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from echoxflow import align_timestamps, EchoXFlowDataset, save_npz
from models.net import ECHOTRACKER2
from utils.evaluate import compute_reference_metrics
from utils.saverloader import load_echo_checkpoint
from utils.utils_ import fetch_optimizer


class Refiner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offsets = torch.nn.Parameter(torch.arange(1., 5.))

    def forward(self, video, query, return_iterations=False):
        self.last_query = query.detach()
        base = query[..., [2, 1]][:, :, None].expand(-1, -1, video.shape[1], -1)
        iterations = base[None] + self.offsets[:, None, None, None, None]
        return {"tracks": iterations[-1], "iterations": iterations}


def wrapper(model):
    result = ECHOTRACKER2.__new__(ECHOTRACKER2)
    result.model, result.device, result.device_ids = model, "cpu", [0]
    return result


class ProtocolTests(unittest.TestCase):
    def test_timestamp_collision_gap_and_failure(self):
        fi, ci, tolerance, errors = align_timestamps(
            np.array([.01, .04, .2, .4]), np.array([0., .1, .2, .3, .4]))
        np.testing.assert_array_equal(fi, np.arange(5))
        np.testing.assert_array_equal(ci, [0, -1, 2, -1, 3])
        self.assertAlmostEqual(tolerance, .050001)
        with self.assertRaises(ValueError):
            align_timestamps(np.array([0., .7]), np.array([0., .1, .2]))
        with self.assertRaises(ValueError):
            align_timestamps(np.array([0., 0.]), np.array([0., .1]))

    def test_metrics_norms_boundaries_query_and_masks(self):
        reference = np.zeros((3, 2, 2))
        prediction = reference.copy()
        prediction[0] = 999  # 查询帧不能进入误差统计。
        prediction[1:, 0] = (1, 1)
        prediction[1:, 1] = 999
        valid = np.ones((3, 2), dtype=bool)
        valid[1:, 1] = False
        metrics = compute_reference_metrics(reference, prediction, valid)
        self.assertEqual(metrics["l1_delta_2"], 0)
        self.assertEqual(metrics["l2_delta_2"], 100)
        self.assertEqual(metrics["l1_delta_4"], 100)
        self.assertEqual(metrics["l1_mte"], 2)
        self.assertAlmostEqual(metrics["l2_mte"], math.sqrt(2))
        exact = compute_reference_metrics(reference, reference, valid)
        self.assertEqual(exact["l1_davg_1_2_4"], 100)
        self.assertEqual(exact["l1_mte"], 0)

    def test_augmentation_keeps_frames_coordinates_and_masks_aligned(self):
        with tempfile.TemporaryDirectory() as folder:
            frames = np.zeros((3, 256, 256), dtype=np.uint8)
            frames[:, 1, 2] = [1, 2, 3]
            xy = np.tile(np.array([[[2., 1.]]], dtype=np.float32), (3, 1, 1))
            valid = np.array([[True], [False], [True]])
            save_npz(Path(folder) / "prepared/a.npz", frames=frames, trajs=xy, valid=valid)
            cfg = dict(output=folder, seed=12, clip_frames=36,
                       time_reverse_probability=1., horizontal_flip_probability=1.)
            ds = EchoXFlowDataset([dict(path="a.npz")], cfg, training=True)
            images, coords, mask = ds[0]
            np.testing.assert_array_equal(images[:, 1, 253, 0], [3, 2, 1])
            np.testing.assert_allclose(coords[..., 0] * 255, 253)
            np.testing.assert_allclose(coords[..., 1] * 255, 1)
            np.testing.assert_array_equal(mask, valid[::-1])

    def test_query_axes_four_iteration_gradients_and_invalid_targets(self):
        model = Refiner()
        w = wrapper(model)
        video = torch.zeros(1, 5, 16, 32, 3)
        reference = torch.tensor([.1, .2]).expand(1, 5, 3, 2).clone()
        valid = torch.ones(1, 5, 3, dtype=torch.bool)
        valid[0, 1, 1] = False
        valid[0, 3, 0] = False
        reference[0, 3, 0] = float('nan')
        loss = w.supervised_loss(video, reference, valid, q_frame=1, max_points=40)
        loss.backward()
        self.assertEqual(tuple(model.last_query.shape), (1, 2, 3))
        torch.testing.assert_close(model.last_query[0, 0], torch.tensor([1., 3., 3.1]))
        expected_gradient = torch.tensor([.8**3, .8**2, .8, 1.]) / 4
        torch.testing.assert_close(model.offsets.grad, expected_gradient)
        self.assertTrue(torch.isfinite(loss))
        points = torch.tensor([[[1., .1, .2]]])
        original = points.clone()
        w.infer(video, points)
        torch.testing.assert_close(points, original)

    def test_accumulation_tail_scheduler_and_epoch_resume(self):
        def build():
            model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.ones_(model.weight)
            w = wrapper(model)
            w.supervised_loss = types.MethodType(
                lambda self, frames, *unused: self.model.weight.sum() * frames.float().mean(), w)
            samples = torch.arange(1., 7.).reshape(6, 1)
            loader = DataLoader(TensorDataset(samples, samples, samples), batch_size=1,
                                generator=torch.Generator().manual_seed(17))
            return w, loader

        metrics = {"l1_davg_1_2_4": 50., "l1_mte": 2.}
        with tempfile.TemporaryDirectory() as folder:
            args = dict(dataset_size={"train": 6}, epochs=2, lr=.01, weight_decay=0.,
                        accumulation_steps=4, grad_clip=100.)
            full, loader = build()
            full.train({"train": loader}, log_dir=folder + "/full/log",
                       ckpt_path=folder + "/full/ckpt", validation_fn=lambda: metrics, **args)
            # 独立验证每组平均梯度，尾组有两个样本，不能仍除以四。
            manual = torch.nn.Parameter(torch.ones(1))
            opt, sch = fetch_optimizer(.01, 0., 1e-8, 2, 2, [manual])
            for gradient in (2.5, 5.5, 2.5, 5.5):
                manual.grad = torch.tensor([gradient])
                opt.step()
                sch.step()
            torch.testing.assert_close(full.model.weight.flatten(), manual)
            interrupted, loader = build()
            calls = []
            def validation():
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                return metrics
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                interrupted.train({"train": loader}, log_dir=folder + "/resume/log",
                                  ckpt_path=folder + "/resume/ckpt", validation_fn=validation, **args)
            resumed, loader = build()
            resumed.train({"train": loader}, log_dir=folder + "/resume/log",
                          ckpt_path=folder + "/resume/ckpt", validation_fn=lambda: metrics,
                          resume=folder + "/resume/ckpt/latest.pth", **args)
            checkpoint = load_echo_checkpoint(folder + "/resume/ckpt/latest.pth", resumed.model)
            self.assertEqual(checkpoint["global_step"], 4)
            self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], 4)
            torch.testing.assert_close(resumed.model.weight, full.model.weight)
            history = [json.loads(x) for x in Path(folder + "/resume/log/history.jsonl").read_text().splitlines()]
            self.assertEqual([x["epoch"] for x in history], [1, 2])


if __name__ == "__main__":
    unittest.main()

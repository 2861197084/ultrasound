"""Check accumulation against a single full-batch loss, including invalid groups."""
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch

from m2p_lab.model import objective
from m2p_lab.training import accumulated_step


class PointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(0.25))
        self.config = dict(huber_delta=1.0, mlc_threshold=0.5, epsilon=1e-6,
                           loss_weights=[0.02, 0.5, 10.0])

    def predict_pair(self, image0, image1, queries):
        scale = 1 + image0[:, :1, :1, :1].reshape(-1, 1, 1)
        xy = queries + self.shift * scale * queries.new_tensor([1., 0.3])
        logits = self.shift * queries.new_tensor([1., 0., -0.5, 0.3])
        probabilities = logits.softmax(-1).reshape(1, 1, 2, 2).expand(len(queries), queries.shape[1], 2, 2)
        return xy, probabilities, None


def example_batch(all_invalid=False):
    queries = torch.tensor([[[7., 7.], [8., 7.], [9., 7.]],
                            [[7., 9.], [8., 9.], [9., 9.]]]).repeat(5, 1, 1, 1)
    offsets = torch.tensor([0.1, 0.4, 1., -1.5, 3.2]).reshape(5, 1, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(28), torch.arange(28), indexing="ij")
    mask = torch.zeros(5, 28, 28)
    mask[:, :14, :14] = 1
    valid = torch.tensor([[True, True], [False, False], [True, False],
                          [True, True], [False, False]])
    if all_invalid:
        valid.zero_()
    return dict(image0=torch.arange(5.).reshape(5, 1, 1, 1), image1=torch.zeros(5, 1, 1, 1),
                queries=queries, pseudo=queries + offsets * torch.tensor([1., -0.5]), valid=valid,
                mask1=mask, distance0=(2 + 0.05 * xx + 0.08 * yy).repeat(5, 1, 1),
                distance1=(3 + 0.07 * xx + 0.01 * yy).repeat(5, 1, 1))


class AccumulationTests(unittest.TestCase):
    def compare_with_full_batch(self, batch):
        full = PointModel()
        optimizer = torch.optim.SGD(full.parameters(), lr=0.01)
        with patch("torch.autocast", lambda *args, **kwargs: nullcontext()):
            loss, expected = objective(full, None, batch, targets=(batch["pseudo"], batch["valid"]))
            loss.backward()
            expected_gradient = full.shift.grad.clone()
            optimizer.step()
            for micro in (1, 2, 3, 5):
                with self.subTest(micro=micro):
                    model = PointModel()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    with patch("m2p_lab.training.gpu_batch", lambda value: value), \
                         patch("m2p_lab.training.teacher_targets", lambda teacher, part, cfg: (part["pseudo"], part["valid"])):
                        metrics, used = accumulated_step(model, None, optimizer, batch, micro)
                    self.assertEqual(used, micro)
                    torch.testing.assert_close(model.shift.grad, expected_gradient, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(model.shift, full.shift, atol=1e-6, rtol=1e-5)
                    for name, value in expected.items():
                        self.assertAlmostEqual(metrics[name], float(value), places=5)

    def test_unequal_valid_groups_and_partial_micro_batch(self):
        self.compare_with_full_batch(example_batch())

    def test_no_valid_lsc_groups_keeps_other_losses_and_gradients(self):
        self.compare_with_full_batch(example_batch(all_invalid=True))


if __name__ == "__main__":
    unittest.main()

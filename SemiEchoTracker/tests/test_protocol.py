"""覆盖会改变实验结论的损失、时间索引、毫米换算与标签边界。"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset.echo_dataset import EchoXFlowDataset, make_dataset
from train import co_training_loss, coordinate_loss, labelled_predictions, prediction_errors_mm, velocity_consistency_loss


class ProtocolTests(unittest.TestCase):
    def test_opposite_trackers_do_not_cancel_co_training_loss(self):
        detector = torch.zeros(1, 1, 1, 2)
        forward = torch.tensor([[[[1., 0.]]]], requires_grad=True)
        backward = torch.tensor([[[[-1., 0.]]]], requires_grad=True)
        loss = co_training_loss(detector, forward, backward, None, None)
        self.assertEqual(loss.item(), 2.)
        loss.backward()
        self.assertEqual(forward.grad[..., 0].item(), 1.)
        self.assertEqual(backward.grad[..., 0].item(), -1.)

    def test_small_error_gradient_points_toward_target(self):
        pred = torch.tensor([[.3, .4], [0., 0.]], requires_grad=True)
        loss = coordinate_loss(pred, torch.zeros_like(pred))
        loss.backward()
        self.assertAlmostEqual(loss.item(), .25)
        torch.testing.assert_close(pred.grad, torch.tensor([[.3, .4], [0., 0.]]))
        self.assertLess(coordinate_loss(pred.detach()-.1*pred.grad, torch.zeros_like(pred)).item(), loss.item())

    def test_velocity_matches_arrivals_at_the_same_middle_frame(self):
        forward = torch.tensor([[[[3., 4.]], [[0., 2.]], [[99., 0.]]]])
        backward = torch.tensor([[[[81., 0.]], [[0., 5.]], [[2., 0.]]]])
        self.assertEqual(velocity_consistency_loss(forward, backward).item(), 0.)
        backward[:, 2] = 0.
        self.assertEqual(velocity_consistency_loss(forward, backward).item(), 1.)

    def test_physical_transform_uses_every_sample(self):
        pred = torch.tensor([[[[3., 4.]]], [[[3., 4.]]]])
        batch = {'input_to_mm': torch.tensor([[[2., 0., 97.], [0., 2., 23.], [0., 0., 1.]],
                                              [[0., -2., 0.], [1., 0., 0.], [0., 0., 1.]]])}
        error = prediction_errors_mm(pred, torch.zeros_like(pred), batch, 'echoxflow')
        torch.testing.assert_close(error.flatten(), torch.tensor([10., 73.**.5]))

    def test_sparse_labels_select_last_video_frame(self):
        pred = torch.arange(10.).reshape(1, 10, 1, 1).expand(2, 10, 3, 2)
        batch = {'label_frame_indices': torch.tensor([[0, 9], [0, 9]])}
        torch.testing.assert_close(labelled_predictions(pred, batch)[:, :, 0, 0], torch.tensor([[0., 9.], [0., 9.]]))

    def test_training_reader_rejects_dense_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'echoxflow'
            root.mkdir()
            split = root / 'train.txt'
            split.write_text('sample\n')
            path = root / 'sample.npz'
            np.savez_compressed(path, video=np.zeros((10, 224, 224, 3), dtype=np.uint8),
                                landmarks=np.zeros((10, 3, 2)), label_frame_indices=np.arange(10),
                                valid_mask=np.ones((10, 3), dtype=bool))
            manifest = dict(complete=True, dataset='EchoXFlow A4C',
                            split_sha256={'train': hashlib.sha256(split.read_bytes()).hexdigest()},
                            samples=[dict(split='train', sample_id='sample', data_path='sample.npz',
                                          data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'label scope'):
                make_dataset(root_dir=directory, data_type='echoxflow', splits='train')

    def test_rotation_moves_images_and_labels_together(self):
        dataset = EchoXFlowDataset.__new__(EchoXFlowDataset)
        video = np.zeros((10, 224, 224, 3), dtype=np.uint8)
        video[:, 120, 100] = 255
        dataset.data_list = [dict(patient='synthetic', video=video,
                                  landmarks=np.tile([100., 120.], (2, 3, 1)),
                                  label_frame_indices=np.array([0, 9]), frame_indices=np.arange(10),
                                  timestamps_seconds=np.arange(10), native_to_mm=np.diag([2., 2., 1.]))]
        dataset.target_size = (224, 224)
        dataset.rotation = True
        dataset.rotation_limit = 90
        dataset.rotation_prob = 1
        dataset.transform = None
        dataset.normalize = lambda image: torch.from_numpy(image.transpose(2, 0, 1).copy())
        with patch('dataset.echo_dataset.random.uniform', return_value=90.), patch('dataset.echo_dataset.random.random', return_value=0.):
            sample = dataset[0]
        y, x = np.unravel_index(sample['video'][0, 0].argmax(), (224, 224))
        np.testing.assert_allclose(sample['landmarks'][0, 0], [x, y], atol=1e-6)
        restored = sample['input_to_mm'] @ np.r_[sample['landmarks'][0, 0], 1]
        np.testing.assert_allclose(restored, [200., 240., 1.], atol=1e-6)


if __name__ == '__main__':
    unittest.main()

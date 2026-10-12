"""Protect mask-only sampling, deterministic resume and non-first ED scoring."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import torch

from m2p_lab.data import ContinuousMaskPairs
from m2p_lab.echo_evaluation import metrics, sample
from m2p_lab.echo_experiment import save_latest


class ContinuousProtocol(unittest.TestCase):
    def test_full_checkpoint_replays_adamw_scheduler_and_rng(self):
        class Model(torch.nn.Sequential):
            def adapter_state(self):
                return {k:v.clone() for k,v in self.state_dict().items()}
        def build():
            model = Model(torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
            opt = torch.optim.AdamW(model.parameters(), lr=.001)
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2], gamma=.1)
            return model, opt, sched
        torch.manual_seed(99)
        m, o, s = build()
        x = torch.ones(3, 4)
        def update(model, opt, scheduler):
            opt.zero_grad(); model(x).square().mean().backward(); opt.step(); scheduler.step()
        update(m,o,s)
        with tempfile.TemporaryDirectory() as folder, patch("m2p_lab.echo_experiment.rt.guard_disk"), patch("torch.cuda.get_rng_state_all", return_value=[]):
            path = Path(folder)/"latest.pt"
            save_latest(m,o,s,{"effective_batch":16},1,{"step":1},0.,path)
            update(m,o,s)
            expected = m.adapter_state()
            new, opt, scheduler = build()
            saved = torch.load(path, weights_only=False)
            new.load_state_dict(saved["adapter"]); opt.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"]); torch.set_rng_state(saved["torch_rng"])
            update(new,opt,scheduler)
            for key, value in expected.items():
                torch.testing.assert_close(new.state_dict()[key],value,rtol=0,atol=0)
            self.assertEqual(opt.param_groups[0]["lr"],o.param_groups[0]["lr"])
            self.assertEqual(saved["next_sampling_index"],16)

    def test_training_never_reads_trajectory_arrays_and_resume_replays_pairs(self):
        class ImagesOnly:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __getitem__(self, key):
                if key != "images":
                    raise AssertionError("Training tried to read trajectory reference")
                return np.zeros((8, 20, 20), np.uint8)
        class MasksOnly(ImagesOnly):
            def __getitem__(self, key):
                if key != "masks": raise AssertionError(key)
                return np.ones((8, 20, 20), bool)
        record = dict(sample_id="a", image_path="images", mask_path="masks", eligible_frames=[0,2,4,7])
        cfg = dict(seed=17, echo_steps=20, effective_batch=16)
        def fake_pair(*args, **kwargs):
            return {"queries": torch.from_numpy(args[5].random((2,2)).astype(np.float32))}
        with patch("m2p_lab.data.np.load", side_effect=lambda path, **kw: ImagesOnly() if path == "images" else MasksOnly()), patch("m2p_lab.data.prepare_pair", side_effect=fake_pair):
            a = ContinuousMaskPairs([record], cfg)
            b = ContinuousMaskPairs([record], cfg, start_index=80)
            for i in range(16):
                left, right = a[80+i], b[i]
                self.assertEqual(left["pair_frames"], right["pair_frames"])
                self.assertNotEqual(*left["pair_frames"])
                self.assertTrue(set(left["pair_frames"]) <= set(record["eligible_frames"]))
                torch.testing.assert_close(left["queries"], right["queries"], rtol=0, atol=0)
                self.assertEqual(left["sampling_index"], right["sampling_index"])

    def test_windows_keep_unreferenced_images_and_actual_ed(self):
        images = np.arange(7, dtype=np.uint8)[:,None,None] * np.ones((7,3,3), np.uint8)
        ref = np.full((2,7,2), .5)
        valid = np.ones((2,7), bool); valid[:,3] = False
        meta = dict(sample_id="a", exam_id="e", status="accepted", view="4ch", fold="4", ed_frame=2,
                    es_frame=5, source_shape=list(images.shape), original_frame_indices=list(range(7)), span_m=[.256,.256])
        with patch("m2p_lab.echo_evaluation.load_sample", return_value=(images,ref,valid,meta)):
            full = sample({"prepared":"."}, {"sha256":"hash"})
            segment = sample({"prepared":"."}, {"sha256":"hash"}, "ed_es_sequence")
            pair = sample({"prepared":"."}, {"sha256":"hash"}, "two_frames")
        self.assertEqual(full[3]["ed_frame"], 2)
        np.testing.assert_array_equal(segment[0], images[2:6])
        self.assertFalse(segment[2][:,1].any())
        np.testing.assert_array_equal(pair[0], images[[2,5]])
        prediction = ref.copy(); prediction[:,:,0] += np.array([8,6,1000,999,4,2,0])/256
        row = metrics(prediction, ref, valid, full[3], "test")
        self.assertEqual(row["es_l2_epe_px"], 2)
        self.assertEqual(row["l2_mte_px"], 4)
        self.assertEqual(row["other_frames_l2_mte_px"], 5)


if __name__ == "__main__":
    unittest.main()

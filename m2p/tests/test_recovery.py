import csv
import json
from pathlib import Path

import m2p_lab
import pytest
import torch
from torch.utils.data import Dataset

from m2p_lab import recovery, training, runtime


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.b = torch.nn.Parameter(torch.tensor([0.2]))


class IndexedPairs(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"image0": torch.tensor([index], dtype=torch.float32)}


def test_best_recovery_preserves_original_logs_and_checkpoint(tmp_path, monkeypatch):
    cfg = {"reserve_gib": 35, "artifact_budget_gib": 2}
    fields = ["step", "seconds", "valid_lsc_groups"]
    with (tmp_path / "training.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows(dict(step=i, seconds=i*.4, valid_lsc_groups=.5) for i in range(1, 7))
    original = (tmp_path / "training.csv").read_bytes()
    (tmp_path / "validation.jsonl").write_text(''.join(json.dumps(dict(step=i, total=1.0 if i == 3 else 2.0))+'\n' for i in [3, 6]))
    torch.save(dict(step=3, adapter={"layer.b": torch.tensor([.2])}, config=cfg,
                    provenance={"test": True}, validation={"total": 1.0}), tmp_path / "best.pt")
    checkpoint = (tmp_path / "best.pt").read_bytes()
    monkeypatch.setattr(recovery, "event", lambda *a, **k: None)
    _, result = recovery.prepare_best_resume(tmp_path, cfg, 8)
    assert (Path(result["archive"]) / "training.csv").read_bytes() == original
    assert (tmp_path / "best.pt").read_bytes() == checkpoint
    assert list(tmp_path.rglob("*.pt")) == [tmp_path / "best.pt"]
    with (tmp_path / "training.csv").open() as f:
        assert [int(r["step"]) for r in csv.DictReader(f)] == [1, 2, 3]
    assert [json.loads(line)["step"] for line in (tmp_path / "validation.jsonl").read_text().splitlines()] == [3]
    assert result["discarded_updates"] == 3
    assert result["discarded_compute_seconds"] == pytest.approx(1.2)
    assert not result["exact_resume"]


@pytest.mark.parametrize("start, expected_rates", [(3, [.01, .01, .001, .001, .001]), (6, [.001, .001])])
def test_recovery_uses_correct_data_cursor_lr_and_keeps_better_checkpoint(tmp_path, monkeypatch, start, expected_rates):
    cfg = dict(learning_rate=.01, weight_decay=0, decay_step=5, decay_factor=.1,
               effective_batch=2, micro_batch=2, loader_workers=0, log_every=1)
    run = tmp_path / "natural"; run.mkdir()
    fields = ['step','seconds','step_seconds','lr','micro_batch','total','lsc','mlc','mbc','foreground_mass','valid_lsc_groups']
    with (run / "training.csv").open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i in range(1, start+1):
            row = dict.fromkeys(fields, 1); row.update(step=i, seconds=i*.4); writer.writerow(row)
    checkpoint = run / "best.pt"; checkpoint.write_bytes(b'prior best must stay unchanged')
    info = dict(restored_step=start, lineage_seconds_before_resume=start*.4, previous_compute_seconds=4,
                checkpoint_validation={"total": 1.0}, discarded_updates=10-start)
    model, teacher = TinyModel(), TinyModel()
    seen, rates = [], []
    def step(model, teacher, optimizer, batch, micro):
        seen.extend(batch['image0'][:,0].tolist()); rates.append(optimizer.param_groups[0]['lr'])
        optimizer.zero_grad(); model.layer.b.square().sum().backward(); optimizer.step()
        return dict(total=2.,lsc=1.,mlc=1.,mbc=1.,foreground_mass=.5,valid_lsc_groups=.5), micro
    monkeypatch.setattr(runtime, 'RUN', tmp_path)
    monkeypatch.setattr(training, 'accumulated_step', step)
    monkeypatch.setattr(training, 'validate', lambda *a: {'total': 2.0})
    monkeypatch.setattr(training, 'guard_disk', lambda *a: None)
    monkeypatch.setattr(training, 'event', lambda *a, **k: None)
    monkeypatch.setattr(training, 'status', lambda *a, **k: None)
    monkeypatch.setattr(training, 'save_best', lambda *a, **k: pytest.fail('Worse validation must not overwrite best'))
    result = training.train(model, teacher, IndexedPairs(), IndexedPairs(), cfg, 'natural', 8, 2, {}, recovery=info)
    assert seen == list(range(start*2, 16))
    assert rates == pytest.approx(expected_rates)
    with (run / "training.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['step']) for r in rows] == list(range(1, 9))
    assert all(float(r['seconds']) >= start*.4 for r in rows[start:])
    assert result['steps'] == 8 and result['updates_this_segment'] == 8-start
    assert result['actual_compute_updates'] == 18-start
    assert not result['exact_uninterrupted_run']
    assert checkpoint.read_bytes() == b'prior best must stay unchanged'

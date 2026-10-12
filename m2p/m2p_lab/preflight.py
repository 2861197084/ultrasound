from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import EchoPairs
from .model import M2P
from .runtime import ROOT, RUN, atomic_json, check_gpu, event, load_config, seed_everything, status
from .training import train


class RepeatedPairs(Dataset):
    def __init__(self, base, length):
        self.base, self.length = base, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.base[index % len(self.base)]


def run(steps=200):
    cfg = load_config()
    check_gpu(cfg)
    seed_everything(cfg["seed"])
    manifest = json.loads((ROOT / "subset_manifests/a4c_full/pilot_training_manifest.json").read_text())
    records = [r for r in manifest["records"] if r["split"] == "train"][:4]
    if len(records) != 4:
        raise RuntimeError("Four training-only Echo clips are required for preflight")
    fixed = EchoPairs(records, cfg, validation=True)
    # This is optimization/throughput verification only. Its weights are discarded.
    dataset = RepeatedPairs(fixed, steps * cfg["effective_batch"])
    model = M2P(cfg).cuda()
    teacher = model.frozen_teacher()
    status("preflight", steps=steps, note="Training-only Echo pairs; formal natural-video training has not started")
    summary = train(model, teacher, dataset, None, cfg, "preflight/optimization", steps, steps,
                    dict(purpose="preflight_only"), preflight=True)
    summary["passed"] = summary["late_mean_loss"] < summary["early_mean_loss"]
    summary["weights_retained"] = False
    summary["data"] = [r["sample_id"] for r in records]
    atomic_json(RUN / "preflight/summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError("Small-sample optimization did not reduce the training objective")
    event("preflight_passed", **summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    run(args.steps)

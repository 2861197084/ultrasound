"""Recover an interrupted run from its single best adapter, without optimizer state."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from .runtime import atomic_json, event, file_sha, guard_disk, scientific_config


def prepare_best_resume(out, cfg, target_steps):
    out = Path(out)
    checkpoint = out / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    start = saved["step"]
    if not 0 < start < target_steps:
        raise RuntimeError(f"Checkpoint step cannot be continued: {start}/{target_steps}")
    if scientific_config(saved["config"]) != scientific_config(cfg):
        raise RuntimeError("Recovery configuration differs from the saved experiment")
    if not all(torch.isfinite(t).all() for t in saved["adapter"].values()):
        raise RuntimeError("Retained checkpoint has nonfinite parameters")
    training = out / "training.csv"
    with training.open() as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, [r for r in reader if r.get("valid_lsc_groups") is not None]
    if not rows or any(int(r["step"]) != i + 1 for i, r in enumerate(rows)):
        raise RuntimeError("Training history is not a continuous sequence")
    if int(rows[-1]["step"]) < start:
        raise RuntimeError("Checkpoint is ahead of the available training history")
    previous = json.loads((out / "recovery.json").read_text()) if (out / "recovery.json").exists() else {}
    actual_seconds = float(rows[-1]["seconds"]) + previous.get("discarded_compute_seconds", 0)
    actual_updates = int(rows[-1]["step"]) + previous.get("discarded_updates", 0)
    prefix_seconds = float(rows[start-1]["seconds"])
    archive = out / "segments" / datetime.now(timezone.utc).strftime("interrupted_%Y%m%dT%H%M%S_%fZ")
    guard_disk(cfg, training.stat().st_size * 2)
    archive.mkdir(parents=True)
    for name in ("training.csv", "validation.jsonl", "recovery.json"):
        original = out / name
        if original.exists():
            shutil.copyfile(original, archive / name)
            if file_sha(original) != file_sha(archive / name):
                raise RuntimeError("Interrupted history archive verification failed")
    info = dict(mode="best_adapter_fresh_optimizer", exact_resume=False, restored_step=start,
                previous_logged_step=int(rows[-1]["step"]), target_steps=target_steps,
                checkpoint_sha256=file_sha(checkpoint), checkpoint_validation=saved["validation"],
                checkpoint_provenance=saved["provenance"], archive=str(archive),
                lineage_seconds_before_resume=prefix_seconds, previous_compute_seconds=actual_seconds,
                discarded_compute_seconds=actual_seconds-prefix_seconds, discarded_updates=actual_updates-start,
                teacher="Original initialization teacher, unchanged by recovery",
                note="Optimizer moments and RNG state were not retained. The adapter is continued from its saved step; this is not an uninterrupted paper reproduction.")
    atomic_json(archive / "recovery_decision.json", info)
    # Preserve all original rows in the archive, and expose the retained model's
    # history plus its future updates in the normal curve/monitoring locations.
    temp = training.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows[:start])
    temp.replace(training)
    validation = out / "validation.jsonl"
    if validation.exists():
        values = [json.loads(line) for line in validation.read_text().splitlines()]
        temp = validation.with_suffix(".jsonl.tmp")
        temp.write_text("".join(json.dumps(row) + "\n" for row in values if row["step"] <= start))
        temp.replace(validation)
    atomic_json(out / "recovery.json", info)
    event("recovery_prepared", restored_step=start, previous_logged_step=info["previous_logged_step"],
          optimizer_restored=False, archive=str(archive), checkpoint_sha256=info["checkpoint_sha256"])
    return saved, info

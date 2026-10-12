from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .model import objective, teacher_targets
from .runtime import RUN, atomic_json, event, guard_disk, save_best, status


def fixed_validation_subset(dataset, name):
    valid, excluded = [], []
    for index in range(len(dataset)):
        try:
            sample = dataset[index]
            if all(torch.isfinite(v).all() for v in sample.values() if isinstance(v, torch.Tensor)):
                valid.append(index)
            else:
                excluded.append(dict(index=index, reason="nonfinite_pair"))
        except (ValueError, RuntimeError) as error:
            excluded.append(dict(index=index, reason=str(error)))
    if not valid:
        raise RuntimeError("No valid independent validation pairs")
    atomic_json(RUN / name / "validation_selection.json", dict(indices=valid, exclusions=excluded,
                policy="Fixed seed and fixed pairs; exclusions determined before optimization"))
    return Subset(dataset, valid)


def loader(dataset, cfg, batch_size, workers=None):
    count = cfg["loader_workers"] if workers is None else workers
    options = dict(batch_size=batch_size, num_workers=count, shuffle=False, pin_memory=True)
    if count:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=1)
    return DataLoader(dataset, **options)


def gpu_batch(batch, begin=0, end=None):
    return {key: value[begin:end].to("cuda:0", non_blocking=True) for key, value in batch.items()
            if isinstance(value, torch.Tensor)}


@torch.no_grad()
def validate(model, teacher, val_loader):
    model.eval()
    totals, count = {}, 0
    for batch in val_loader:
        n = batch["image0"].shape[0]
        _, values = objective(model, teacher, gpu_batch(batch))
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value) * n
        count += n
    result = {key: value / count for key, value in totals.items()}
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError("Nonfinite validation objective")
    model.train()
    return result


def accumulated_step(model, teacher, optimizer, batch, micro_batch):
    total_size = len(batch["image0"])
    if total_size < 1 or micro_batch < 1:
        raise ValueError("Batch sizes must be positive")
    # Transfer each tensor once; only the student activations are micro-batched.
    full_batch = gpu_batch(batch)
    while True:
        optimizer.zero_grad(set_to_none=True)
        sums = {}
        prepared_targets = []
        try:
            for begin in range(0, total_size, micro_batch):
                end = min(total_size, begin + micro_batch)
                part = {key: value[begin:end] for key, value in full_batch.items()}
                prepared_targets.append(teacher_targets(teacher, part, model.config))
            valid_count = torch.stack([target[1].sum() for target in prepared_targets]).sum().clamp_min(1)
            for begin in range(0, total_size, micro_batch):
                end = min(total_size, begin + micro_batch)
                weight = (end - begin) / total_size
                part = {key: value[begin:end] for key, value in full_batch.items()}
                # After multiplication by weight, every valid LSC group has the
                # same denominator across the whole effective batch. MLC/MBC
                # retain their ordinary per-sample mean weighting.
                loss, values = objective(model, teacher, part,
                                         targets=prepared_targets[begin // micro_batch],
                                         lsc_denominator=valid_count * weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training objective")
                (loss * weight).backward()
                for key, value in values.items():
                    sums[key] = sums.get(key, 0) + value * weight
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
                raise RuntimeError("Missing or nonfinite gradients")
            optimizer.step()
            return {key: float(value) for key, value in sums.items()}, micro_batch
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if micro_batch == 1 or model.config.get("allow_micro_batch_fallback") is False:
                raise
            micro_batch = 1
            loss = values = None
            sums.clear()
            prepared_targets.clear()
            torch.cuda.empty_cache()
            event("micro_batch_reduced", new_micro_batch=1, accumulation=total_size)


def train(model, teacher, train_data, validation_data, cfg, name, steps, validation_every, provenance,
          preflight=False, start_checks=True, recovery=None):
    out = RUN / name
    out.mkdir(parents=True, exist_ok=True)
    model.train()
    teacher.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    start_step = recovery["restored_step"] if recovery else 0
    if not 0 <= start_step < steps:
        raise ValueError("Invalid training start step")
    elapsed_before = recovery["lineage_seconds_before_resume"] if recovery else 0
    initial_lr = cfg["learning_rate"] * (cfg["decay_factor"] if start_step >= cfg["decay_step"] else 1)
    optimizer = torch.optim.AdamW(parameters, lr=initial_lr, weight_decay=cfg["weight_decay"])
    milestone = cfg["decay_step"] - start_step
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone] if milestone > 0 else [], gamma=cfg["decay_factor"])
    if recovery:
        train_data = Subset(train_data, range(start_step * cfg["effective_batch"], steps * cfg["effective_batch"]))
    data_loader = loader(train_data, cfg, cfg["effective_batch"])
    val_loader = loader(validation_data, cfg, cfg["micro_batch"]) if validation_data is not None else None
    best_score = recovery["checkpoint_validation"]["total"] if recovery else math.inf
    micro_batch = cfg["micro_batch"]
    best_path = out / "best.pt"
    started = time.monotonic()
    baseline_b = next(p for key, p in model.named_parameters() if key.endswith(".b")).detach().clone()
    csv_path = out / "training.csv"
    history = []
    with csv_path.open("a" if recovery else "w", newline="", buffering=1) as handle:
        writer = None
        for step, batch in enumerate(data_loader, start=start_step + 1):
            if step > steps:
                break
            before = time.monotonic()
            metrics, micro_batch = accumulated_step(model, teacher, optimizer, batch, micro_batch)
            scheduler.step()
            row = dict(step=step, seconds=elapsed_before + time.monotonic() - started, step_seconds=time.monotonic() - before,
                       lr=optimizer.param_groups[0]["lr"], micro_batch=micro_batch, **metrics)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not recovery:
                    writer.writeheader()
            writer.writerow(row)
            if step <= start_step + 10 or step % cfg["log_every"] == 0 or step == steps:
                event("training", experiment=name, **row)
                status("training", experiment=name, step=step, total_steps=steps,
                       **({"recovery": recovery} if recovery else {}), **{k: v for k, v in row.items() if k != "step"})
                guard_disk(cfg)
            if step <= start_step + 20 or step > steps - 20:
                history.append(row)
            if val_loader and (step % validation_every == 0 or step == steps):
                validation = validate(model, teacher, val_loader)
                with (out / "validation.jsonl").open("a") as f:
                    f.write(json.dumps(dict(step=step, **validation)) + "\n")
                event("validation", experiment=name, step=step, **validation)
                if validation["total"] < best_score:
                    save_best(model, best_path, cfg, step, validation, provenance)
                    best_score = validation["total"]
    if step < steps:
        raise RuntimeError(f"Training loader exhausted before requested updates: {step}/{steps}")
    delta = float((next(p for key, p in model.named_parameters() if key.endswith(".b")).detach() - baseline_b).abs().max())
    if delta <= 0:
        raise RuntimeError("LoRA parameters did not update")
    segment_seconds = time.monotonic() - started
    result = dict(experiment=name, steps=steps, seconds=segment_seconds + (recovery["previous_compute_seconds"] if recovery else 0), micro_batch=micro_batch,
                  trainable_parameters=sum(p.numel() for p in parameters), max_lora_b_change=delta,
                  best_validation=None if not math.isfinite(best_score) else best_score,
                  early_mean_loss=float(np.mean([r["total"] for r in history[:min(20, len(history))]])),
                  late_mean_loss=float(np.mean([r["total"] for r in history[-min(20, len(history)):]])),
                  estimated_150k_hours=segment_seconds / (steps-start_step) * 150000 / 3600)
    if recovery:
        result.update(recovery=recovery, exact_uninterrupted_run=False, restored_from_step=start_step,
                      updates_this_segment=steps-start_step, segment_seconds=segment_seconds,
                      lineage_seconds=elapsed_before+segment_seconds,
                      actual_compute_updates=steps+recovery["discarded_updates"], early_loss_scope="current resumed segment")
    atomic_json(out / "summary.json", result)
    event("training_finished", **result)
    del data_loader, val_loader, optimizer, scheduler
    return result

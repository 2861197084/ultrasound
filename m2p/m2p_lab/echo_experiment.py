"""Configured continuous EchoXFlow experiment within the M2P repository."""
from __future__ import annotations

import csv
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from . import runtime as rt
from .data import ContinuousMaskPairs
from .model import M2P
from .training import accumulated_step, loader
from .echo_evaluation import evaluate_model, infer, sample, validate_points


def save_latest(model, optimizer, scheduler, cfg, step, best, elapsed, path=None):
    path = Path(path) if path is not None else rt.RUN / "training/latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    rt.guard_disk(cfg, 150 * 1024**2)
    payload = dict(format="m2p-resume-v1", adapter=model.adapter_state(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), config=cfg, step=step, best=best, elapsed=elapsed,
        next_sampling_index=step * cfg["effective_batch"], python_rng=random.getstate(),
        numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    restored = torch.load(temp, map_location="cpu", weights_only=False)
    if restored["step"] != step or any(not torch.equal(v, restored["adapter"][k]) for k, v in payload["adapter"].items()):
        raise RuntimeError("Resume checkpoint verification failed")
    temp.replace(path)


def probe(cfg, split, records):
    rt.status("preflight", training_started=False, gpu=1)
    rt.seed_everything(cfg["seed"])
    model = M2P(cfg).cuda()
    teacher = model.frozen_teacher()
    initial = model.adapter_state()
    pilot_ids = ("4ch_572566bb81dc2c4ccc69", "4ch_0917026331be00685e6e", "4ch_2f6f3e0d2b5ddc6c8f49")
    checks = []
    lookup = {r["sample_id"]: r for r in split["train"]}
    for sid in pilot_ids:
        item = lookup[sid]
        images, ref, valid, meta, _ = sample(cfg, item)
        pred, _, _, timing = infer(model, images, ref[:, meta["ed_frame"]], meta["ed_frame"])
        repeat, _, _, _ = infer(model, images, ref[:, meta["ed_frame"]], meta["ed_frame"])
        if not np.array_equal(pred, repeat):
            raise ValueError("Repeated inference changed")
        short, r2, _, m2, _ = sample(cfg, item, "two_frames")
        endpoint, _, _, _ = infer(model, short, r2[:, 0], 0)
        if not np.array_equal(endpoint, pred[:, [meta["ed_frame"], meta["es_frame"]]]):
            raise ValueError("Window parity failed")
        checks.append(dict(sample_id=sid, frames=len(images), ed=meta["ed_frame"], repeat_exact=True,
                           windows_exact=True, **timing))
        # Confirm mask/image/query geometry visually without using test labels.
        from PIL import Image, ImageDraw
        mask_record = next(r for r in records if r["sample_id"] == sid)
        with np.load(mask_record["mask_path"]) as z:
            masks = z["masks"]
        canvas = Image.new("RGB", (336*3, 356))
        for col, frame in enumerate(dict.fromkeys([meta["ed_frame"], meta["es_frame"], len(images)-1])):
            rgb = np.repeat(images[frame, ..., None], 3, axis=-1)
            rgb[masks[frame]] = (rgb[masks[frame]]*.7 + np.array([0, 100, 0])*.3).astype(np.uint8)
            panel = Image.fromarray(rgb)
            draw = ImageDraw.Draw(panel)
            for xy in ref[valid[:, frame], frame] * 335:
                x, y = xy; draw.ellipse((x-2, y-2, x+2, y+2), outline="lime")
            for xy in pred[:, frame] * 335:
                x, y = xy; draw.ellipse((x-1, y-1, x+1, y+1), fill="red")
            canvas.paste(panel, (336*col, 20))
            ImageDraw.Draw(canvas).text((336*col+4, 3), f"frame {frame}; green=mask/ref red=initial M2P", fill="white")
        path = rt.RUN / "preflight_visuals" / (sid + ".png")
        path.parent.mkdir(exist_ok=True)
        canvas.save(path)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    dl = loader(ContinuousMaskPairs(records, cfg, length=40 * cfg["effective_batch"]), cfg, cfg["effective_batch"])
    started = time.monotonic()
    probe_rows = []
    last = started
    for step, batch in enumerate(dl, 1):
        start = time.monotonic()
        values, micro = accumulated_step(model, teacher, optimizer, batch, cfg["micro_batch"])
        probe_rows.append(dict(step=step, data_wait_seconds=start-last, compute_seconds=time.monotonic()-start, **values))
        last = time.monotonic()
    changed = any(not torch.equal(v, model.adapter_state()[k]) for k, v in initial.items())
    if not changed:
        raise RuntimeError("No probe parameter update")
    rt.atomic_json(rt.RUN / "preflight.json", dict(passed=True, checks=checks, loader_workers=cfg["loader_workers"],
        micro_batch=micro, effective_batch=cfg["effective_batch"], updates=len(probe_rows),
        seconds=time.monotonic()-started, rows=probe_rows, weights_discarded=True))
    del dl, model, teacher, optimizer
    torch.cuda.empty_cache()
    rt.event("preflight_passed", updates=40, weights_discarded=True)


def train_run(cfg, split, records, provenance, resume=False):
    out = rt.RUN / "training"
    out.mkdir(exist_ok=True)
    rt.seed_everything(cfg["seed"])
    model = M2P(cfg).cuda()
    teacher = model.frozen_teacher()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [cfg["decay_step"]], gamma=cfg["decay_factor"])
    step, elapsed_before, best = 0, 0., None
    if (out / "latest.pt").exists():
        if not resume:
            raise ValueError("Training exists: use --resume; refusing overwrite")
        saved = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        if saved["config"] != cfg:
            raise ValueError("Resume configuration changed")
        model.load_adapter(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step, best, elapsed_before = saved["step"], saved["best"], saved["elapsed"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        # A loader may have prefetched beyond this position; replay from the saved index.
        if saved["next_sampling_index"] != step * cfg["effective_batch"]:
            raise ValueError("Resume sampling position mismatch")
        del saved
        rt.event("resume", step=step, optimizer_restored=True, scheduler_restored=True, rng_restored=True)
    elif resume:
        raise ValueError("No full resume checkpoint")
    elif (out / "training.csv").exists():
        raise ValueError("Uncheckpointed run exists; preserve it explicitly before starting anew")
    else:
        rt.status("initial_validation", step=0, validation_videos=len(split["validation"]))
        initial = validate_points(model, split["validation"], cfg)
        rt.atomic_json(out / "initial_validation.json", dict(step=0, eligible_for_selection=False, **initial))
        rt.event("initial_validation", es_l2_epe_px=initial["es_l2_epe_px"])
    if step == cfg["echo_steps"]:
        if not (out / "summary.json").exists():
            rt.atomic_json(out / "summary.json", dict(complete=True, steps=step,
                sampled_pairs=step*cfg["effective_batch"], best=best, seconds=elapsed_before,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        return
    # Use a private generator so starting/restarting workers cannot consume model RNG.
    dl = loader(ContinuousMaskPairs(records, cfg, start_index=step * cfg["effective_batch"]), cfg, cfg["effective_batch"])
    dl.generator = torch.Generator().manual_seed(cfg["seed"])
    csv_path = out / "training.csv"
    if resume:
        # Keep interrupted progress as evidence, but do not mix discarded steps in canonical history.
        for file, column in ((csv_path, "step"), (out / "sampled_pairs.csv", "step")):
            if file.exists():
                shutil.copy2(file, file.with_suffix(f".before_resume_{step}.csv"))
                with file.open() as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames
                    rows = [r for r in reader if int(r[column]) <= step]
                with file.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    started = last = time.monotonic()
    model.train()
    with csv_path.open("a", newline="", buffering=1) as f, (out / "sampled_pairs.csv").open("a", newline="", buffering=1) as pair_file:
        writer = None
        pairs = csv.writer(pair_file)
        if not resume:
            pairs.writerow(["step", "sampling_index", "sample_id", "source_frame", "target_frame", "augmentation_retries"])
        for step, batch in enumerate(dl, step + 1):
            start = time.monotonic()
            wait_seconds = start - last
            lr_used = optimizer.param_groups[0]["lr"]
            torch.cuda.reset_peak_memory_stats()
            values, micro = accumulated_step(model, teacher, optimizer, batch, cfg["micro_batch"])
            scheduler.step()
            row = dict(step=step, seconds=elapsed_before+time.monotonic()-started, data_wait_seconds=wait_seconds,
                       compute_seconds=time.monotonic()-start, learning_rate=lr_used, micro_batch=micro,
                       peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3, **values)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not resume:
                    writer.writeheader()
            writer.writerow(row)
            for idx, sid, i, j, retries in zip(batch["sampling_index"].tolist(), batch["record_id"],
                    batch["pair_frames"][0].tolist(), batch["pair_frames"][1].tolist(), batch["augmentation_retries"].tolist()):
                pairs.writerow([step, idx, sid, i, j, retries])
            if step <= 10 or step % cfg["log_every"] == 0:
                rt.event("training", **row)
                rt.status("training", total_steps=cfg["echo_steps"], **row)
                rt.guard_disk(cfg)
            if step % cfg["echo_validation_every"] == 0 or step == cfg["echo_steps"]:
                rt.status("validation", step=step, total_steps=cfg["echo_steps"], validation_videos=len(split["validation"]))
                validation = validate_points(model, split["validation"], cfg)
                rt.atomic_json(out / "validation" / f"step_{step:05d}.json", dict(step=step, **validation))
                rt.event("validation", step=step, es_l2_epe_px=validation["es_l2_epe_px"])
                if best is None or validation["es_l2_epe_px"] < best["es_l2_epe_px"]:
                    best = dict(step=step, es_l2_epe_px=validation["es_l2_epe_px"])
                    rt.save_best(model, out / "best.pt", cfg, step, best, provenance)
                save_latest(model, optimizer, scheduler, cfg, step, best, elapsed_before+time.monotonic()-started)
            last = time.monotonic()
    if step != cfg["echo_steps"]:
        raise ValueError("Training did not reach the requested number of updates")
    rt.atomic_json(out / "summary.json", dict(complete=True, steps=step, sampled_pairs=step*cfg["effective_batch"],
        best=best, seconds=elapsed_before+time.monotonic()-started,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    del dl, model, teacher, optimizer, scheduler
    torch.cuda.empty_cache()


def run_experiment(config_path, stage="all", resume=False):
    cfg = rt.load_config(config_path)
    rt.configure(cfg)
    lock = (rt.RUN / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with (rt.RUN / "commands.log").open("a") as f:
            f.write(rt.timestamp() + " " + shlex.join([sys.executable, "-m", "m2p_lab.pipeline", *sys.argv[1:]]) + "\n")
        if stage == "report":
            from .echo_report import report
            report(cfg)
            return
        rt.check_gpu(cfg)
        rt.guard_disk(cfg)
        mask_path = rt.RUN / "preparation/manifest.json"
        if not mask_path.exists():
            subprocess.run([str(rt.ROOT / ".venv-data/bin/python"), "-u", "-m", "m2p_lab.echo_masks",
                            "--config", str(config_path)], cwd=rt.ROOT, check=True)
        prep = json.loads(mask_path.read_text())
        if not prep["complete"] or prep["split_sha256"] != rt.file_sha(cfg["split_manifest"]):
            raise ValueError("Mask preparation or fixed split invalid")
        split = json.loads(Path(cfg["split_manifest"]).read_text())["groups"]
        for r in prep["records"]:
            if rt.file_sha(r["mask_path"]) != r["mask_sha256"] or rt.file_sha(r["image_path"]) != r["image_sha256"]:
                raise ValueError("Training cache changed")
        if {r["sample_id"] for r in prep["records"]} != {r["sample_id"] for r in split["train"]}:
            raise ValueError("Training split mismatch")
        provenance = dict(config=cfg, config_sha256=rt.file_sha(config_path),
            split_sha256=rt.file_sha(cfg["split_manifest"]), masks_manifest_sha256=rt.file_sha(mask_path),
            initial_weights_sha256=rt.file_sha(cfg["initial_weights"]), implementation="independent M2P reproduction",
            environment={"python":sys.version, "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}},
            requirements_sha256=rt.file_sha(rt.ROOT / "requirements-train.txt"),
            sources={str(p.relative_to(rt.ROOT)): rt.file_sha(p) for p in sorted((rt.ROOT / "m2p_lab").glob("*.py"))},
            metric_sources={str(p): rt.file_sha(p) for p in (rt.ROOT.parent / "EchoTracker2/evaluate.py",
                rt.ROOT.parent / "EchoTracker2/evaluation_data.py", rt.ROOT.parent / "EchoTracker2/utils/evaluate.py")})
        path = rt.RUN / "provenance.json"
        if not path.exists() or (not (rt.RUN / "training/initial_validation.json").exists() and stage in ("all", "probe", "train")):
            rt.atomic_json(path, provenance)
            for relative in provenance["sources"]:
                target = rt.RUN / "source_snapshot" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rt.ROOT / relative, target)
        elif stage in ("all", "train") and json.loads(path.read_text()) != provenance:
            raise ValueError("Source/configuration changed since experiment start")
        if stage in ("all", "probe") and not (rt.RUN / "preflight.json").exists():
            probe(cfg, split, prep["records"])
        if stage in ("all", "train"):
            if not json.loads((rt.RUN / "preflight.json").read_text())["passed"]:
                raise ValueError("Preflight required")
            train_run(cfg, split, prep["records"], provenance, resume)
        if stage in ("all", "evaluate"):
            completed_training = json.loads((rt.RUN / "training/summary.json").read_text())
            if not completed_training["complete"] or completed_training["steps"] != cfg["echo_steps"]:
                raise ValueError("The fixed training budget must finish before test evaluation")
            rt.seed_everything(cfg["seed"])
            model = M2P(cfg).cuda()
            evaluate_model(model, "dinov2", cfg, split["test"], raw=True)
            saved = torch.load(rt.RUN / "training/best.pt", map_location="cpu", weights_only=True)
            model.load_adapter(saved["adapter"])
            evaluate_model(model, "m2p", cfg, split["test"])
            del model
            from .echo_report import report
            report(cfg)
            rt.status("complete", report=str(rt.RUN / "REPORT_zh.md"))
    except BaseException as error:
        rt.atomic_json(rt.RUN / "failure.json", dict(time=rt.timestamp(), reason=repr(error), traceback=traceback.format_exc()))
        rt.status("failed", reason=repr(error))
        raise
    finally:
        lock.close()

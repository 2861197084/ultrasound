from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import traceback

import torch

from .data import VOSPairs, index_natural
from .evaluation import benchmark_path, evaluate_benchmarks
from .model import M2P
from .recovery import prepare_best_resume
from .runtime import (ROOT, RUN, atomic_json, check_gpu, event, file_sha,
                      guard_disk, load_config, seed_everything, status)
from .training import fixed_validation_subset, loader, train, validate


def check_inputs(cfg, stage):
    required = [ROOT / "assets/dinov2_vitb14_pretrain.pth",
                ROOT / "vendor" / ("dinov2-" + cfg["dinov2_commit"]) / "dinov2/hub/backbones.py"]
    if stage in ("all", "evaluate"):
        required.append(ROOT / "assets/tapvid_evaluation_datasets.py")
        for name in ("davis", "rgb_stacking"):
            benchmark_path(name)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required reproduction resource is missing: {path}")


def code_provenance(cfg, config_path):
    paths = sorted(list((ROOT / "m2p_lab").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
                   + [config_path])
    hashes = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): file_sha(p) for p in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    result = dict(code_sha256=digest, files=hashes, base_weights_sha256=file_sha(ROOT / "assets/dinov2_vitb14_pretrain.pth"),
                  dinov2_commit=cfg["dinov2_commit"], tapnet_commit=cfg["tapnet_commit"],
                  physical_gpu=1, query_order="t,x,y", trajectory_order="t,point,xy",
                  inference_uses_masks=False, implementation="Independent paper reproduction; unpublished choices are documented in config")
    atomic_json(RUN / "provenance.json", result)
    return result


def load_best(model):
    saved = torch.load(RUN / "natural/best.pt", map_location="cpu", weights_only=True)
    expected = {k: v for k, v in saved["config"].items() if k != "natural_validation_mode"}
    actual = {k: v for k, v in model.config.items() if k != "natural_validation_mode"}
    if expected != actual:
        raise RuntimeError("Configuration differs from the saved natural-video checkpoint")
    model.load_adapter(saved["adapter"])
    model.eval()
    event("best_loaded", experiment="natural", step=saved["step"], validation=saved["validation"])
    return saved


def run(config_path=None, stage="all", resume_best=False):
    if stage not in ("all", "train", "evaluate"):
        raise ValueError(f"Unknown stage: {stage}")
    if stage == "evaluate" and resume_best:
        raise ValueError("--resume-best applies only to training")
    config_path = Path(config_path or ROOT / "configs/m2p_v2b14.json").resolve()
    cfg = load_config(config_path)
    check_inputs(cfg, stage)
    if (stage == "evaluate" or resume_best) and not (RUN / "natural/best.pt").is_file():
        raise FileNotFoundError(f"Natural-video checkpoint is missing: {RUN / 'natural/best.pt'}")
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another experiment pipeline is already running")
        if stage in ("all", "train"):
            if (RUN / "natural/summary.json").exists():
                raise RuntimeError("A completed natural-video experiment already exists; refusing to overwrite its results")
            if (RUN / "natural/training.csv").exists() and not resume_best:
                raise RuntimeError("Partial natural-video history exists; use --resume-best for explicitly labelled recovery")
        command = [sys.executable, "-m", "m2p_lab.pipeline", "--config", str(config_path), "--stage", stage]
        if resume_best:
            command.append("--resume-best")
        atomic_json(RUN / "process.json", dict(pid=os.getpid(), command=shlex.join(command), physical_gpu=1,
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(), stage=stage, resume_best=resume_best))
        event("pipeline_started", stage=stage, resume_best=resume_best)
        try:
            guard_disk(cfg)
            if stage in ("all", "train"):
                status("preparing_data")
                manifest = index_natural()
                cfg["natural_validation_mode"] = manifest["validation_mode"]
                event("natural_data_indexed", validation_mode=manifest["validation_mode"], **manifest["counts"])
                training = VOSPairs(manifest["train"], cfg)
                validation = VOSPairs(manifest["validation"], cfg, validation=True,
                                      synthetic_validation=manifest["validation_mode"].startswith("synthetic"))
                validation = fixed_validation_subset(validation, "natural")
            check_gpu(cfg)
            seed_everything(cfg["seed"])
            model = M2P(cfg).cuda()
            if stage in ("all", "evaluate"):
                status("evaluating_baseline")
                evaluate_benchmarks(model, "dinov2_baseline", raw=True)
            if stage in ("all", "train"):
                seed_everything(cfg["seed"])
                teacher = model.frozen_teacher()
                recovery = None
                if resume_best:
                    saved = load_best(model)
                    recovery_loader = loader(validation, cfg, cfg["micro_batch"])
                    recovered_validation = validate(model, teacher, recovery_loader)
                    del recovery_loader
                    if not all(math.isclose(recovered_validation[key], value, rel_tol=1e-4, abs_tol=1e-4)
                               for key, value in saved["validation"].items()):
                        raise RuntimeError("Reloaded checkpoint does not reproduce its original validation objective")
                    atomic_json(RUN / "natural/recovery_validation_check.json", dict(passed=True, step=saved["step"],
                                expected=saved["validation"], actual=recovered_validation,
                                teacher="Original frozen initialization", optimizer_restored=False))
                    _, recovery = prepare_best_resume(RUN / "natural", cfg, cfg["natural_steps"])
                    event("recovery_validation_passed", step=saved["step"], validation=recovered_validation)
                atomic_json(RUN / "configuration.json", cfg)
                provenance = code_provenance(cfg, config_path)
                if recovery:
                    provenance = dict(provenance, recovery=recovery)
                train(model, teacher, training, validation, cfg, "natural", cfg["natural_steps"],
                      cfg["natural_validation_every"], provenance, recovery=recovery)
                del teacher
            if stage in ("all", "evaluate"):
                load_best(model)
                status("evaluating_natural_best")
                evaluate_benchmarks(model, "natural_best")
            guard_disk(cfg)
            status("complete", requested_stage=stage)
            event("experiment_complete", stage=stage)
        except BaseException as error:
            atomic_json(RUN / "failure.json", dict(error=str(error), traceback=traceback.format_exc()))
            status("failed", error=str(error))
            event("experiment_failed", error=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2P natural-video training and TAP-Vid evaluation", allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m2p_v2b14.json")
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--resume-best", action="store_true", help="Continue best adapter weights with a fresh optimizer; not an exact resume")
    arguments = parser.parse_args()
    run(arguments.config, arguments.stage, arguments.resume_best)

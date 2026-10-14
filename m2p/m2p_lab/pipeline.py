from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

import torch

from . import runtime
from .data import VOSPairs, index_natural
from .evaluation import benchmark_path, evaluate_benchmarks, evaluate_echoxflow, compare_echoxflow
from .model import M2P
from .recovery import prepare_best_resume
from .runtime import (ROOT, atomic_json, check_gpu, event, file_sha, configure_run,
                      guard_disk, load_config, scientific_config, seed_everything, status)
from .training import fixed_validation_subset, loader, train, validate


def check_inputs(cfg, stage):
    if cfg.get("dataset", "natural") not in ("natural", "echoxflow"):
        raise ValueError("Unknown training dataset")
    if stage == "prepare":
        return
    required = [ROOT / "assets/dinov2_vitb14_pretrain.pth",
                ROOT / "vendor" / ("dinov2-" + cfg["dinov2_commit"]) / "dinov2/hub/backbones.py"]
    if cfg.get("dataset", "natural") == "natural" and stage in ("all", "evaluate"):
        required.append(ROOT / "assets/tapvid_evaluation_datasets.py")
        for name in ("davis", "rgb_stacking"):
            benchmark_path(name)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required reproduction resource is missing: {path}")


def code_provenance(cfg, config_path, gpu_info, manifest):
    paths = sorted(list((ROOT / "m2p_lab").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
                   + [config_path, ROOT / "requirements-train.txt"])
    hashes = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): file_sha(p) for p in paths}
    result = dict(code_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), files=hashes,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  base_weights_sha256=file_sha(ROOT / "assets/dinov2_vitb14_pretrain.pth"),
                  dinov2_commit=cfg["dinov2_commit"], tapnet_commit=cfg["tapnet_commit"], gpu=gpu_info,
                  query_order="t,x,y", trajectory_order="t,point,xy", inference_uses_masks=False,
                  data_fingerprint=manifest.get("fingerprint"),
                  implementation="Independent paper reproduction; unpublished choices are documented in config")
    atomic_json(runtime.RUN / "provenance.json", result)
    return result


def load_best(model, name="natural"):
    saved = torch.load(runtime.RUN / name / "best.pt", map_location="cpu", weights_only=True)
    if scientific_config(saved["config"]) != scientific_config(model.config):
        raise RuntimeError("Experiment configuration differs from the saved checkpoint")
    model.load_adapter(saved["adapter"])
    model.eval()
    event("best_loaded", experiment=name, step=saved["step"], validation=saved["validation"])
    return saved


def run(config_path=None, stage="all", resume_best=False):
    if stage not in ("all", "prepare", "train", "evaluate"):
        raise ValueError(f"Unknown stage: {stage}")
    if stage in ("evaluate", "prepare") and resume_best:
        raise ValueError("--resume-best applies only to training")
    config_path = Path(config_path or ROOT / "configs/m2p_v2b14.json").resolve()
    cfg = load_config(config_path)
    out = configure_run(cfg)
    name = cfg.get("dataset", "natural")
    check_inputs(cfg, stage)
    if (stage == "evaluate" or resume_best) and not (out / name / "best.pt").is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {out / name / 'best.pt'}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another experiment pipeline is already running in this directory")
        if stage in ("all", "train"):
            if (out / name / "summary.json").exists():
                raise RuntimeError("Completed training already exists; refusing to overwrite its results")
            if (out / name / "training.csv").exists() and not resume_best:
                raise RuntimeError("Partial training exists; use --resume-best for explicitly labelled recovery")
        command = [sys.executable, "-B", "-m", "m2p_lab.pipeline", "--config", str(config_path), "--stage", stage]
        if resume_best:
            command.append("--resume-best")
        atomic_json(out / "process.json", dict(pid=os.getpid(), command=shlex.join(command), gpu=cfg.get("gpu", 1),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(), stage=stage, resume_best=resume_best))
        event("pipeline_started", dataset=name, stage=stage, resume_best=resume_best)
        try:
            guard_disk(cfg)
            manifest = {}
            if name == "echoxflow":
                from .echoxflow import EchoXFlowPairs, prepare_echoxflow
                status("preparing_data")
                manifest = prepare_echoxflow(cfg)
                if stage in ("all", "train"):
                    training = EchoXFlowPairs([r for r in manifest["records"] if r["split"] == "train"], cfg)
                    validation = EchoXFlowPairs([r for r in manifest["records"] if r["split"] == "validation"], cfg, validation=True)
                    validation = fixed_validation_subset(validation, name, strict=True)
                steps, validation_every = cfg["train_steps"], cfg["validation_every"]
            else:
                if stage in ("all", "train", "prepare"):
                    status("preparing_data")
                    manifest = index_natural()
                    cfg["natural_validation_mode"] = manifest["validation_mode"]
                    event("natural_data_indexed", validation_mode=manifest["validation_mode"], **manifest["counts"])
                    training = VOSPairs(manifest["train"], cfg)
                    validation = VOSPairs(manifest["validation"], cfg, validation=True,
                                          synthetic_validation=manifest["validation_mode"].startswith("synthetic"))
                    validation = fixed_validation_subset(validation, name)
                steps, validation_every = cfg["natural_steps"], cfg["natural_validation_every"]
            if stage == "prepare":
                (out / "failure.json").unlink(missing_ok=True)
                status("prepared", dataset=name, counts=manifest["counts"])
                return
            gpu_info = check_gpu(cfg)
            seed_everything(cfg["seed"])
            model = M2P(cfg).cuda()
            baseline = None
            if stage in ("all", "evaluate"):
                status("evaluating_baseline")
                if name == "echoxflow":
                    baseline = evaluate_echoxflow(model, manifest, "dinov2_baseline", raw=True)
                else:
                    evaluate_benchmarks(model, "dinov2_baseline", raw=True)
            if stage in ("all", "train"):
                seed_everything(cfg["seed"])
                teacher = model.frozen_teacher()
                recovery = None
                if resume_best:
                    saved = load_best(model, name)
                    recovery_loader = loader(validation, cfg, cfg["micro_batch"])
                    recovered_validation = validate(model, teacher, recovery_loader)
                    del recovery_loader
                    if not all(math.isclose(recovered_validation[key], value, rel_tol=1e-4, abs_tol=1e-4)
                               for key, value in saved["validation"].items()):
                        raise RuntimeError("Reloaded checkpoint does not reproduce its original validation objective")
                    atomic_json(out / name / "recovery_validation_check.json", dict(passed=True, step=saved["step"],
                                expected=saved["validation"], actual=recovered_validation,
                                teacher="Original frozen initialization", optimizer_restored=False))
                    _, recovery = prepare_best_resume(out / name, cfg, steps)
                    event("recovery_validation_passed", step=saved["step"], validation=recovered_validation)
                atomic_json(out / "configuration.json", cfg)
                provenance = code_provenance(cfg, config_path, gpu_info, manifest)
                if recovery:
                    provenance = dict(provenance, recovery=recovery)
                train(model, teacher, training, validation, cfg, name, steps,
                      validation_every, provenance, recovery=recovery)
                del teacher
                torch.cuda.empty_cache()
            if stage in ("all", "evaluate"):
                saved = load_best(model, name)
                status("evaluating_best")
                if name == "echoxflow":
                    trained = evaluate_echoxflow(model, manifest, "m2p_best")
                    comparison = compare_echoxflow(baseline, trained, cfg)
                    comparison.update(best_step=saved["step"], checkpoint_sha256=file_sha(out / name / "best.pt"))
                    atomic_json(out / "evaluation/comparison.json", comparison)
                else:
                    evaluate_benchmarks(model, "natural_best")
            guard_disk(cfg)
            (out / "failure.json").unlink(missing_ok=True)
            status("complete", requested_stage=stage)
            event("experiment_complete", stage=stage)
        except BaseException as error:
            atomic_json(out / "failure.json", dict(error=str(error), traceback=traceback.format_exc()))
            status("failed", error=str(error))
            event("experiment_failed", error=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2P masked training and independent point evaluation", allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m2p_v2b14.json")
    parser.add_argument("--stage", choices=("all", "prepare", "train", "evaluate"), default="all")
    parser.add_argument("--resume-best", action="store_true", help="Continue best adapter weights with a fresh optimizer; not an exact resume")
    arguments = parser.parse_args()
    run(arguments.config, arguments.stage, arguments.resume_best)

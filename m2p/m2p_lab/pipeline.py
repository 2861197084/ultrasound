from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import torch

from .data import EchoPairs, VOSPairs, index_natural
from .evaluation import evaluate_benchmarks, evaluate_echo
from .model import M2P
from .report import make_report
from .recovery import prepare_best_resume
from .runtime import (ROOT, RUN, atomic_json, check_gpu, event, file_sha,
                      guard_disk, load_config, seed_everything, status)
from .training import fixed_validation_subset, loader, train, validate


def preparation_processes(script):
    pids = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = (directory / "cmdline").read_bytes().split(b"\0")
            if os.fsencode(script) in args:
                pids.append(int(directory.name))
        except OSError:
            pass
    return pids


def wait_for_preparation(script, ready, arguments, logname):
    """Reuse running preparation jobs, then retry incomplete assets at most twice."""
    last_log = 0
    while preparation_processes(script):
        if time.monotonic() - last_log > 300:
            event("waiting_for_preparation", script=script, pids=preparation_processes(script))
            last_log = time.monotonic()
        time.sleep(20)
    for attempt in range(3):
        if ready():
            return
        if attempt == 2:
            raise RuntimeError(f"Preparation remains incomplete: {script}")
        event("preparation_retry", script=script, attempt=attempt + 1)
        with (RUN / "preparation" / logname).open("a") as log:
            subprocess.run([str(ROOT / ".venv-data/bin/python"), "-u", script, *arguments], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=False)


def assets_ready():
    required = ("dinov2_weights", "dinov2_source", "tapvid_source", "davis2017", "tapvid_davis", "tapvid_rgb_stacking",
                "youtubevos2019_train", "youtubevos2019_valid")
    return all((RUN / "preparation" / (name + ".json")).exists() for name in required)


def code_provenance(cfg):
    paths = sorted(list((ROOT / "m2p_lab").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
                   + [ROOT / "configs/m2p_v2b14.json"])
    hashes = {str(p.relative_to(ROOT)): file_sha(p) for p in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    result = dict(code_sha256=digest, files=hashes, base_weights_sha256=file_sha(ROOT / "assets/dinov2_vitb14_pretrain.pth"),
                  dinov2_commit=cfg["dinov2_commit"], tapnet_commit=cfg["tapnet_commit"],
                  physical_gpu=1, query_order="t,x,y", trajectory_order="t,point,xy",
                  inference_uses_masks=False, implementation="Independent paper reproduction; unpublished choices are documented in config")
    atomic_json(RUN / "provenance.json", result)
    return result


def load_best(model, name):
    saved = torch.load(RUN / name / "best.pt", map_location="cpu", weights_only=True)
    model.load_adapter(saved["adapter"])
    model.eval()
    event("best_loaded", experiment=name, step=saved["step"], validation=saved["validation"])
    return saved


def run(resume_best=False):
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / "experiment.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another experiment pipeline is already running")
    cfg = load_config()
    atomic_json(RUN / "configuration.json", cfg)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    command = "python -m m2p_lab.pipeline" + (" --resume-best" if resume_best else "")
    atomic_json(RUN / "process.json", dict(pid=os.getpid(), command=command, physical_gpu=1, boot_id=boot_id, resume_best=resume_best))
    event("pipeline_started", pid=os.getpid(), boot_id=boot_id, resume_best=resume_best)
    try:
        suites = ET.parse(RUN / "preflight_tests.xml").getroot().findall("testsuite")
        if not suites or sum(int(s.attrib["tests"]) for s in suites) < 10 or any(int(s.attrib.get("failures", 0)) or int(s.attrib.get("errors", 0)) for s in suites):
            raise RuntimeError("Implementation tests have not passed")
        preflight = json.loads((RUN / "preflight/summary.json").read_text())
        if not preflight.get("passed") or preflight["steps"] != 200:
            raise RuntimeError("The 200-update preflight has not passed")
        status("preparing_data", note="Implementation and 200-update preflight passed; waiting for full paper datasets")
        wait_for_preparation("scripts/prepare_m2p_assets.py", assets_ready, [], "download.log")
        guard_disk(cfg)
        manifest = index_natural()
        cfg["natural_validation_mode"] = manifest["validation_mode"]
        atomic_json(RUN / "configuration.json", cfg)
        event("natural_data_indexed", validation_mode=manifest["validation_mode"], **manifest["counts"])
        training = VOSPairs(manifest["train"], cfg)
        validation = VOSPairs(manifest["validation"], cfg, validation=True,
                              synthetic_validation=manifest["validation_mode"].startswith("synthetic"))
        validation = fixed_validation_subset(validation, "natural")
        if (RUN / "natural/summary.json").exists():
            raise RuntimeError("A completed natural-video experiment already exists; refusing to overwrite its results")
        if (RUN / "natural/training.csv").exists() and not resume_best:
            raise RuntimeError("Partial natural-video history exists; use --resume-best for explicitly labelled recovery")
        check_gpu(cfg)
        seed_everything(cfg["seed"])
        provenance = code_provenance(cfg)
        model = M2P(cfg).cuda()
        initialization = model.adapter_state()
        teacher = model.frozen_teacher()
        status("evaluating_baseline")
        evaluate_benchmarks(model, "dinov2_baseline", raw=True)
        model.load_adapter(initialization)
        seed_everything(cfg["seed"])
        recovery = None
        if resume_best:
            saved_resume = load_best(model, "natural")
            recovery_loader = loader(validation, cfg, cfg["micro_batch"])
            recovered_validation = validate(model, teacher, recovery_loader)
            del recovery_loader
            if not all(math.isclose(recovered_validation[key], value, rel_tol=1e-4, abs_tol=1e-4)
                       for key, value in saved_resume["validation"].items()):
                raise RuntimeError("Reloaded checkpoint does not reproduce its original validation objective")
            atomic_json(RUN / "natural/recovery_validation_check.json", dict(passed=True, step=saved_resume["step"],
                        expected=saved_resume["validation"], actual=recovered_validation,
                        teacher="Original frozen initialization", optimizer_restored=False))
            _, recovery = prepare_best_resume(RUN / "natural", cfg, cfg["natural_steps"])
            provenance = dict(provenance, recovery=recovery)
            event("recovery_validation_passed", step=saved_resume["step"], validation=recovered_validation)
        train(model, teacher, training, validation, cfg, "natural", cfg["natural_steps"],
              cfg["natural_validation_every"], provenance, recovery=recovery)
        status("evaluating_natural_final")
        evaluate_benchmarks(model, "natural_final")
        saved = load_best(model, "natural")
        natural_adapter = saved["adapter"]
        evaluate_benchmarks(model, "natural_best")
        del teacher
        make_report()
        status("preparing_echo")
        echo_path = ROOT / "subset_manifests/a4c_full/training_manifest.json"
        if not echo_path.is_file():
            raise RuntimeError("Legacy EchoXFlow training cache is absent. Rebuild data for an explicit --config before running a new experiment.")
        echo = json.loads(echo_path.read_text())
        records = {split: [r for r in echo["records"] if r["split"] == split] for split in ("train", "validation", "test")}
        atomic_json(RUN / "echo_manifest.json", echo)
        echo_train = EchoPairs(records["train"], cfg)
        echo_val = fixed_validation_subset(EchoPairs(records["validation"], cfg, validation=True), "echo_direct")
        model.load_adapter(initialization)
        evaluate_echo(model, records["test"], "dinov2_baseline", raw=True)
        model.load_adapter(natural_adapter)
        evaluate_echo(model, records["test"], "natural_zero_shot")
        for name, initial_state in (("echo_direct", initialization), ("echo_transfer", natural_adapter)):
            seed_everything(cfg["seed"])
            model.load_adapter(initial_state)
            teacher = model.frozen_teacher()
            experiment_provenance = dict(provenance, initialization=name,
                                         echo_revision=echo["revision"], validation_split="official fold 3",
                                         natural_teacher_checkpoint_step=saved["step"] if name == "echo_transfer" else None)
            train(model, teacher, echo_train, echo_val, cfg, name, cfg["echo_steps"], cfg["echo_validation_every"], experiment_provenance)
            evaluate_echo(model, records["test"], name + "_final")
            load_best(model, name)
            evaluate_echo(model, records["test"], name + "_best")
            del teacher
            make_report()
        guard_disk(cfg)
        status("complete", report=str(RUN / "REPORT.md"))
        event("experiment_complete", report=str(RUN / "REPORT.md"))
    except BaseException as error:
        atomic_json(RUN / "failure.json", dict(error=str(error), traceback=traceback.format_exc()))
        status("failed", error=str(error), note="Existing best checkpoints and completed results are retained")
        event("experiment_failed", error=str(error))
        make_report()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=("all", "probe", "train", "evaluate", "report"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-best", action="store_true", help="Continue retained best weights with a fresh optimizer; preserve interrupted logs")
    arguments = parser.parse_args()
    if arguments.config:
        from .echo_experiment import run_experiment
        run_experiment(arguments.config, arguments.stage, arguments.resume)
    else:
        run(resume_best=arguments.resume_best)

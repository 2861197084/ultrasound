from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from . import GPU_UUID

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/m2p_reproduction"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def event(kind, **values):
    row = dict(time=timestamp(), event=kind, **values)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "events.jsonl").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu(config):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("GPU isolation was changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device must be visible")
    driver = ctypes.CDLL("libcuda.so.1")
    bus = ctypes.create_string_buffer(64)
    if driver.cuInit(0) or driver.cuDeviceGetPCIBusId(bus, 64, 0):
        raise RuntimeError("Cannot verify CUDA device identity")
    if bus.value.decode().lower() != "0000:07:00.0":
        raise RuntimeError("Visible CUDA device is not physical GPU 1")
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    free, total = torch.cuda.mem_get_info()
    info = dict(uuid=GPU_UUID, physical_device=1, logical_device=0, pci_bus=bus.value.decode(),
                name=torch.cuda.get_device_name(0), free_bytes=free, total_bytes=total,
                torch=torch.__version__, cuda=torch.version.cuda)
    event("gpu_verified", **info)
    return info


def guard_disk(config, additional=0):
    free = shutil.disk_usage(ROOT).free
    if free < config["reserve_gib"] * 1024**3 + additional:
        raise RuntimeError("Disk reserve reached; refusing further writes")
    size = sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())
    if size + additional > config["artifact_budget_gib"] * 1024**3:
        raise RuntimeError("Experiment artifact budget reached")
    return free


def load_config(path=None):
    return json.loads(Path(path or ROOT / "configs/m2p_v2b14.json").read_text())


def configure(config):
    global RUN
    RUN = Path(config["output"]).resolve()
    RUN.mkdir(parents=True, exist_ok=True)


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_best(model, path, config, step, metrics, provenance):
    """Atomically replace one adapter checkpoint after verifying deserialization."""
    state = model.adapter_state()
    payload = dict(format="m2p-adapter-v1", adapter=state, config=config, step=step,
                   validation=metrics, provenance=provenance, created=timestamp())
    guard_disk(config, sum(v.numel() * v.element_size() for v in state.values()) * 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        readback = torch.load(temp, map_location="cpu", weights_only=True)
        if readback["step"] != step or set(readback["adapter"]) != set(state):
            raise RuntimeError("Checkpoint metadata mismatch")
        for name, value in state.items():
            if not torch.equal(value, readback["adapter"][name]):
                raise RuntimeError(f"Checkpoint tensor mismatch: {name}")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    event("best_saved", path=str(path), step=step, bytes=path.stat().st_size, validation=metrics)
    return payload


def status(stage, **values):
    atomic_json(RUN / "status.json", dict(updated=timestamp(), stage=stage, **values))

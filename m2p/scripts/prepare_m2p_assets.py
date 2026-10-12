"""Fetch pinned research assets with resumable downloads and a disk reserve."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import threading
import time
import zipfile
from html import unescape

import requests

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
STATE = ROOT / "runs/m2p_reproduction/preparation"
RESERVE = 35 * 1024**3
LOCK = threading.Lock()
DINO_SHA = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
TAP_SHA = "730cda1c730877cfedbe01bf87fb1cadb78a565d"
YTVOS_MIRROR = "https://huggingface.co/datasets/yinloonga/YouTube_VOS_2019/resolve/7a0eb80c729b0dbbceaea15950e86ed606ab21ed/"
EXPECTED_SHA256 = {
    "youtubevos2019_train": "5a81246005cfc2245bcd1f7e8fff41868963dfc3873e92bb810ea5f35a7cf8c0",
    "youtubevos2019_valid": "3c847cd8f64ec6219db88671681303e84537e5478d53a88534592761ba74eb03",
    "dinov2_weights": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
}
SOURCES = {
    "dinov2_weights": ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth", 346378731, "dinov2_vitb14_pretrain.pth", None),
    "dinov2_source": (f"https://codeload.github.com/facebookresearch/dinov2/zip/{DINO_SHA}", None, "dinov2_source.zip", ROOT / "vendor"),
    "tapvid_source": (f"https://raw.githubusercontent.com/google-deepmind/tapnet/{TAP_SHA}/tapnet/tapvid/evaluation_datasets.py", None, "tapvid_evaluation_datasets.py", None),
    "tapvid_license": (f"https://raw.githubusercontent.com/google-deepmind/tapnet/{TAP_SHA}/LICENSE", None, "TAPNET_LICENSE", None),
    "davis2017": ("https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip", 832766765, "DAVIS-2017-trainval-480p.zip", ROOT.parent / "data/natural"),
    "tapvid_davis": ("https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip", 1668710491, "tapvid_davis.zip", ROOT.parent / "data/benchmarks"),
    "tapvid_rgb_stacking": ("https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip", 187291581, "tapvid_rgb_stacking.zip", ROOT.parent / "data/benchmarks"),
    "youtubevos2019_train": (YTVOS_MIRROR + "train.tar", 9258444800, "ytvos2019_train.tar", ROOT.parent / "data/youtubevos2019"),
    "youtubevos2019_valid": (YTVOS_MIRROR + "valid.tar", 1296824320, "ytvos2019_valid.tar", ROOT.parent / "data/youtubevos2019"),
}


def log(**record):
    with LOCK:
        print(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record)), flush=True)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def reserve(extra=0):
    if shutil.disk_usage(ROOT).free < RESERVE + extra:
        raise RuntimeError("Insufficient free space above the 35 GiB reserve")


def stream(session, url, offset):
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    if url.startswith("gdrive:"):
        file_id = url.split(":", 1)[1]
        # Follow the public large-file confirmation form, without authentication.
        probe = session.get(f"https://drive.google.com/uc?export=download&id={file_id}", timeout=(30, 90), stream=True)
        probe.raise_for_status()
        if "text/html" in probe.headers.get("Content-Type", ""):
            page = probe.text
            probe.close()
            form = re.search(r'<form[^>]+action="([^"]+)"', page)
            params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
            if not form or "id" not in params:
                raise RuntimeError("Google Drive did not return a public download form")
            return session.get(unescape(form[1]), params={k: unescape(v) for k, v in params.items()}, headers=headers, stream=True, timeout=(30, 120))
        if offset:
            direct = probe.url
            probe.close()
            return session.get(direct, headers=headers, stream=True, timeout=(30, 120))
        return probe
    return session.get(url, headers=headers, stream=True, timeout=(30, 120))


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ranged_download(name, url, path, expected):
    """Small retriable byte ranges avoid losing progress on interrupted streams."""
    if path.exists() and path.stat().st_size == expected:
        return
    partial = path.with_suffix(path.suffix + ".part")
    journal = path.with_suffix(path.suffix + ".ranges.json")
    block = 8 * 1024**2
    if journal.exists():
        data = json.loads(journal.read_text())
        if data["url"] != url or data["size"] != expected:
            raise RuntimeError("Range journal source changed")
        completed = set(data["completed"])
    else:
        prefix = partial.stat().st_size if partial.exists() else 0
        completed = set(range(prefix // block))
    reserve(max(0, expected - len(completed) * block))
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, expected)
    state_lock = threading.Lock()
    last = [0.0]
    def record():
        atomic_json(journal, dict(url=url, size=expected, block=block, completed=sorted(completed)))
    record()
    def chunk(index):
        start, end = index * block, min(expected, (index + 1) * block) - 1
        for attempt in range(30):
            try:
                query = "&" if "?" in url else "?"
                target = url + query + f"m2p_range={start}-{end}&attempt={attempt}"
                with requests.get(target, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                                  timeout=(30, 120), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{expected}":
                        raise RuntimeError("Server did not honor the exact requested range")
                    data = response.raw.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Incomplete ranged response")
                reserve(len(data))
                written = os.pwrite(fd, data, start)
                if written != len(data):
                    raise RuntimeError("Short local write")
                with state_lock:
                    completed.add(index)
                    record()
                    if time.monotonic() - last[0] > 30:
                        log(asset=name, state="range_progress", completed_chunks=len(completed),
                            total_chunks=math.ceil(expected / block), bytes=min(len(completed) * block, expected), total=expected)
                        last[0] = time.monotonic()
                return
            except Exception as error:
                if attempt == 29:
                    raise RuntimeError(f"Range {index} failed: {error}") from error
                time.sleep(min(1 + attempt, 8))
    try:
        pending = [i for i in range(math.ceil(expected / block)) if i not in completed]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for _ in pool.map(chunk, pending):
                pass
        os.fsync(fd)
    finally:
        os.close(fd)
    partial.replace(path)
    journal.unlink()


def download(name, url, path, expected):
    partial = path.with_suffix(path.suffix + ".part")
    if path.exists() and (expected is None or path.stat().st_size == expected):
        return
    session = requests.Session()
    for attempt in range(40):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            with stream(session, url, offset) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Received an HTML page instead of the requested asset")
                if offset and response.status_code == 200:
                    offset = 0
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Invalid resume range")
                total = expected or (int(response.headers.get("Content-Length", 0)) + offset)
                reserve(max(0, total - offset))
                last = time.monotonic()
                written = offset
                with partial.open("ab" if offset else "wb") as f:
                    for block in response.iter_content(4 * 1024**2):
                        if not block:
                            continue
                        reserve(len(block))
                        f.write(block)
                        written += len(block)
                        if time.monotonic() - last >= 30:
                            log(asset=name, state="downloading", bytes=written, total=total)
                            last = time.monotonic()
                    f.flush()
                    os.fsync(f.fileno())
                if total and partial.stat().st_size != total:
                    raise RuntimeError(f"Size mismatch: {partial.stat().st_size} != {total}")
            partial.replace(path)
            return
        except Exception as error:
            log(asset=name, state="retry", attempt=attempt + 1, error=str(error))
            if attempt == 39:
                raise
            time.sleep(min(2**attempt, 15))


def extract(path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    files = 0
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            reserve(sum(x.file_size for x in entries))
            for entry in entries:
                if not (destination / entry.filename).resolve().is_relative_to(root):
                    raise ValueError("Unsafe ZIP member")
                if not entry.is_dir() and (destination / entry.filename).exists():
                    raise FileExistsError(f"Refusing to overwrite an existing asset: {destination / entry.filename}")
            archive.extractall(destination)
            files = sum(not e.is_dir() for e in entries)
    else:
        with tarfile.open(path) as archive:
            entries = archive.getmembers()
            reserve(sum(e.size for e in entries if e.isfile()))
            for entry in entries:
                if not (destination / entry.name).resolve().is_relative_to(root) or not (entry.isfile() or entry.isdir()):
                    raise ValueError("Unsafe TAR member")
                if entry.isfile() and (destination / entry.name).exists():
                    raise FileExistsError(f"Refusing to overwrite an existing asset: {destination / entry.name}")
            archive.extractall(destination, filter="data")
            files = sum(e.isfile() for e in entries)
    return files


def prepare(name):
    url, expected, filename, destination = SOURCES[name]
    marker = STATE / (name + ".json")
    if marker.exists():
        info = json.loads(marker.read_text())
        if Path(info["retained_path"]).exists():
            log(asset=name, state="already_prepared")
            return info
    path = ASSETS / filename
    log(asset=name, state="starting")
    if expected and expected > 100 * 1024**2 and not url.startswith("gdrive:"):
        ranged_download(name, url, path, expected)
    else:
        download(name, url, path, expected)
    size, digest = path.stat().st_size, sha256(path)
    if name in EXPECTED_SHA256 and digest != EXPECTED_SHA256[name]:
        raise RuntimeError(f"Asset SHA256 mismatch: {name}")
    info = dict(name=name, source=url, archive_bytes=size, sha256=digest, retained_path=str(path), complete=True)
    if name.startswith("youtubevos"):
        info["mirror_reason"] = "The official Google Drive training file returned Quota exceeded. This public mirror is revision-pinned and checked against its published LFS SHA256."
        info["official_source"] = "https://youtube-vos.org/dataset/vos/"
    if destination:
        info["extracted_files"] = extract(path, destination)
        info["retained_path"] = str(destination)
        atomic_json(marker, info)
        path.unlink()
    else:
        atomic_json(marker, info)
    log(asset=name, state="complete", bytes=size, sha256=digest)
    return info


def main():
    ASSETS.mkdir(exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    # Small assets are ready first; long dataset transfers continue concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        order = ["youtubevos2019_train", "youtubevos2019_valid", "davis2017"] + [n for n in SOURCES if n not in ("youtubevos2019_train", "youtubevos2019_valid", "davis2017")]
        futures = {pool.submit(prepare, name): name for name in order}
        results, failures = {}, {}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as error:
                failures[name] = str(error)
                log(asset=name, state="failed", error=str(error))
        atomic_json(STATE / "assets_summary.json", dict(complete=not failures, assets=results, failures=failures, dinov2_commit=DINO_SHA, tapnet_commit=TAP_SHA))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

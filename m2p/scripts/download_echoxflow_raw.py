"""Fetch or verify the pinned LV subset using official metadata.

Use m2p/.venv-data/bin/python. Commands: plan, fetch, verify.
Plans, TAR indices and verification results stay in memory. Progress goes to
stdout; only original data and its reading metadata are saved. Existing stores
are reused after checking official decoded-array hashes, without state files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time

import numpy as np
import requests
import zarr

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/echoxflow"
REV = "d907c536cae16750b9e46f269a27befd4979ebdd"
CODE_REV = "b782e0a4e3dbaadf3c88fa572710bf4254d86143"
CATALOG_SHA = "2b0e80f83f8743f5babde82a3d8ded8ec4d607a1889a1b8b8cd204342116f83d"
DATASET = "Ahus-AIM/EchoXFlow"
LOCK = threading.Lock()
MiB, GiB = 1024**2, 1024**3


class RemoteTar(io.RawIOBase):
    """Seekable, bounded-memory range reader; never falls back to a full GET."""

    def __init__(self, url, size, index_block_kib=256):
        self.url, self.size = url, size
        self.session = requests.Session()
        self.direct_url = None
        self.pos = 0
        self.buffer_start, self.buffer = -1, b""
        self.network_bytes, self.requests_count = 0, 0
        self.index_block_bytes = index_block_kib * 1024

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("Negative offset")
        return self.pos

    def fetch(self, start, end):
        if not 0 <= start <= end < self.size:
            raise ValueError("Range outside archive")
        for attempt in range(5):
            try:
                url = self.direct_url or self.url + f"?subset_range={start}-{end}&attempt={attempt}&nonce={time.time_ns()}"
                with self.session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}, timeout=(20, 75), stream=True) as response:
                    self.requests_count += 1
                    if response.status_code in (401, 403):
                        self.direct_url = None
                    if response.status_code != 206:
                        raise RuntimeError(f"Expected HTTP 206, got {response.status_code}")
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server returned an unexpected byte range")
                    data = response.raw.read(end - start + 2)
                    self.network_bytes += len(data)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Incomplete byte range")
                    self.direct_url = response.url
                    return data
            except Exception as error:
                # A stale signed response or a broken TLS connection must not
                # keep all retries pinned to the same cached redirect URL.
                self.direct_url = None
                if attempt == 4:
                    detail = str(error) if isinstance(error, RuntimeError) else type(error).__name__
                    raise RuntimeError(f"Range {start}-{end} failed ({detail})") from None
                time.sleep(min(2 ** attempt, 8))

    def read(self, count=-1):
        if count < 0:
            raise ValueError("Unbounded remote reads disabled")
        count = min(count, self.size - self.pos)
        if count <= 0:
            return b""
        end = self.pos + count
        if not (self.buffer_start <= self.pos and end <= self.buffer_start + len(self.buffer)):
            self.buffer_start = self.pos
            self.buffer = self.fetch(self.pos, min(self.size, self.pos + max(count, self.index_block_bytes)) - 1)
        result = self.buffer[self.pos - self.buffer_start:end - self.buffer_start]
        self.pos = end
        return result

    def close(self):
        self.session.close()
        super().close()


def safe_name(name, exam):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or tuple(path.parts[:2]) != ("exams", exam):
        raise ValueError("Unexpected archive path")
    return path


def now():
    return datetime.now(timezone.utc).isoformat()



def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)



def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))



def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * MiB), b""): h.update(b)
    return h.hexdigest()



def array_sha(a):
    h = hashlib.sha256(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()



def event(kind, **kw):
    row = dict(time=now(), event=kind, **kw)
    with LOCK:
        print(json.dumps(row, ensure_ascii=False), flush=True)



def reserve(extra=0, floor=35):
    if shutil.disk_usage(DEST).free < floor * GiB + extra:
        raise RuntimeError("disk_reserve_reached")



def source_bootstrap():
    """Use only the three original metadata files, without cleanup records."""
    source = DEST / "source"
    for name in ("croissant.json", "splits.csv", "LICENSE"):
        if not (source / name).is_file():
            raise RuntimeError(f"missing_pinned_source_metadata:{source / name}")
    if sha(source / "croissant.json") != CATALOG_SHA:
        raise RuntimeError("source_catalog_checksum_mismatch")
    return source


def build_plan():
    source_bootstrap()
    catalog = read(DEST / "source/croissant.json")
    sets = {r["@id"]: r["data"] for r in catalog["recordSet"]}
    records = {r["recordings/recording_id"]: r for r in sets["recordings"]}
    ann_ids = {rid for rid, r in records.items() if "2d_left_ventricular_strain" in r["recordings/content_types"]}
    links = [r for r in sets["recording_links"] if r["recording_links/source_recording_id"] in ann_ids]
    b_ids = {r["recording_links/linked_recording_id"] for r in links}
    chosen = ann_ids | b_ids
    arrays = [a for a in sets["arrays"] if a["arrays/recording_id"] in chosen]
    by_id = {a["arrays/array_id"]: a for a in arrays}
    folds = {r["exam_id"]: int(r["five_fold_split"]) for r in csv.DictReader((DEST / "source/splits.csv").open())}
    grouped = defaultdict(list)
    for a in arrays: grouped[a["arrays/recording_id"]].append(a)
    stores = []
    for rid in sorted(chosen):
        r = records[rid]
        stores.append(dict(recording_id=rid, exam_id=r["recordings/exam_id"], path=r["recordings/zarr_path"],
             fold=folds[r["recordings/exam_id"]], role="annotation" if rid in ann_ids else "video",
             content_types=r["recordings/content_types"], arrays=grouped[rid]))
    link_rows = []
    for l in links:
        aid, bid, view = (l[f"recording_links/{k}"] for k in ("source_recording_id", "linked_recording_id", "panel_role_id"))
        assert view in {"2ch", "3ch", "4ch"}
        contour = by_id[aid + f"/data/{view}_contour"]
        target = by_id.get(l["recording_links/linked_frame_array_id"])
        status = "resolved"
        if target is None: status = "pending_multiplane_mapping"
        elif not all(contour["arrays/shape"]): status = "empty_contour"
        assert folds[records[aid]["recordings/exam_id"]] == folds[records[bid]["recordings/exam_id"]]
        link_rows.append(dict(annotation_recording=aid, video_recording=bid, view=view, status=status,
             fold=folds[records[aid]["recordings/exam_id"]],
             annotation_path=records[aid]["recordings/zarr_path"], video_path=records[bid]["recordings/zarr_path"],
             contour_path=f"data/{view}_contour", contour_times=f"timestamps/{view}_contour",
             declared_frames=l["recording_links/linked_frames_array_path"], declared_times=l["recording_links/linked_timestamps_array_path"],
             available_frame_arrays=[p for p in records[bid]["recordings/array_paths"] if p.startswith("data/2d_brightness_mode")],
             image_sha256=target["arrays/data_sha256"] if target else None,
             official_link=l))
    exams = sorted({r["exam_id"] for r in stores})
    counts = dict(annotation_stores=len(ann_ids), video_stores=len(b_ids), zarr_stores=len(stores), exams=len(exams), arrays=len(arrays),
         links_by_view=dict(Counter(l["view"] for l in link_rows)), links_by_status=dict(Counter(l["status"] for l in link_rows)))
    assert [counts[k] for k in ("annotation_stores", "video_stores", "zarr_stores", "exams", "arrays")] == [446, 877, 1323, 336, 13085]
    assert counts["links_by_view"] == {"4ch":446, "2ch":446, "3ch":278}
    assert counts["links_by_status"] == {"resolved":976, "pending_multiplane_mapping":193, "empty_contour":1}
    plan = dict(schema=1, dataset=DATASET, revision=REV, created=now(), counts=counts, stores=stores,
         exams=[dict(exam_id=e) for e in exams], selection="All original LV annotation stores and linked image stores; no quality exclusion or payload deduplication",
         indexed=False, reserve_gib=35)
    catalog["name"] += " — complete LV linked raw subset"
    catalog["description"] += f" Local subset of pinned revision {REV}; original complete catalog retained under ../source. Raw recording files are unchanged."
    for rs in catalog["recordSet"]:
        if rs["@id"] == "recordings": rs["data"] = [records[x] for x in sorted(chosen)]
        elif rs["@id"] == "arrays": rs["data"] = arrays
        elif rs["@id"] == "recording_links": rs["data"] = links
    return plan, catalog, folds


def check_index(index, exam, archive_bytes):
    if index["revision"] != REV or index["archive_bytes"] != archive_bytes: raise ValueError("index_provenance_mismatch")
    seen = set()
    for e in index["members"]:
        safe_name(e["name"], exam)
        if e["name"] in seen: raise ValueError("duplicate_tar_member")
        seen.add(e["name"])
        if not 0 <= e["offset"] <= e["offset"] + e["size"] <= archive_bytes: raise ValueError("invalid_member_bounds")
    return index



def make_remote(exam, size):
    return RemoteTar(f"https://huggingface.co/datasets/{DATASET}/resolve/{REV}/exams/{exam}.tar", size, 64)



def index_exam(row):
    """Read TAR headers in memory; never download or persist a whole exam TAR."""
    exam, size = row["exam_id"], row["archive_bytes"]
    reserve(); remote = make_remote(exam, size); entries=[]; last=time.monotonic()
    try:
        with tarfile.open(fileobj=remote, mode="r:") as archive:
            for m in archive:
                safe_name(m.name, exam)
                if m.isfile(): entries.append(dict(name=m.name, size=m.size, offset=m.offset_data))
                elif not m.isdir(): raise ValueError("unexpected_tar_special_file")
                if time.monotonic()-last>30:
                    event("index_progress", exam=exam, entries=len(entries)); last=time.monotonic()
        return check_index(dict(revision=REV, archive_bytes=size, members=entries),exam,size)
    finally:
        remote.close()


def archive_sizes():
    url=f"https://huggingface.co/api/datasets/{DATASET}/tree/{REV}/exams?limit=1000"
    sizes={}
    while url:
        with requests.get(url,timeout=30) as response:
            response.raise_for_status()
            for row in response.json():
                if row["type"]=="file" and row["path"].endswith(".tar"):
                    sizes[Path(row["path"]).stem]=row["size"]
            url=response.links.get("next",{}).get("url")
    return sizes


def attach_members(stores, index):
    for store in stores:
        prefix=store["path"]+"/"
        files=[e for e in index["members"] if e["name"].startswith(prefix)]
        names={e["name"] for e in files}
        required={prefix+".zattrs",prefix+".zgroup"}
        required.update(prefix+a["arrays/array_path"]+"/.zarray" for a in store["arrays"])
        if not required<=names:raise ValueError(f"missing_official_tar_members:{store['recording_id']}")
        store["members"]=files;store["payload_bytes"]=sum(e["size"] for e in files)


def inspect_store(store, root):
    group=zarr.open_group(str(root/store["path"]),mode="r")
    declarations={a["name"]:a for a in group.attrs["arrays"]}
    results=[];quality=[]
    assert group.attrs["recording_id"]==store["recording_id"]
    assert group.attrs["exam_id"]==store["exam_id"]
    for info in store["arrays"]:
        name=info["arrays/array_path"];a=np.asarray(group[name][:]);decl=declarations[name]
        h=array_sha(a)
        catalog_hash=info.get("arrays/data_sha256")
        if list(a.shape)!=info["arrays/shape"] or str(a.dtype)!=info["arrays/dtype"] or (catalog_hash and h!=catalog_hash):
            raise ValueError(f"official_array_mismatch:{store['recording_id']}:{name}")
        if decl["shape"]!=list(a.shape) or decl["dtype"]!=str(a.dtype) or decl.get("data_sha256")!=h:
            raise ValueError(f"official_recording_array_mismatch:{store['recording_id']}:{name}")
        results.append(dict(path=name,shape=list(a.shape),dtype=str(a.dtype),sha256=h,
                            checksum_source="official_recording_attributes_and_catalog" if catalog_hash else "official_recording_attributes"))
        if not a.size: quality.append(dict(array=name,issue="empty_source_array")); continue
        if np.issubdtype(a.dtype,np.floating) and not np.isfinite(a).all():
            quality.append(dict(array=name,issue="source_nonfinite_values",count=int((~np.isfinite(a)).sum())))
        if name.startswith("timestamps/") and a.ndim==1 and len(a)>1 and np.any(np.diff(a)<=0):
            quality.append(dict(array=name,issue="source_timestamps_not_strictly_increasing",count=int((np.diff(a)<=0).sum())))
        if name in ("data/2ch_contour","data/3ch_contour","data/4ch_contour"):
            if a.ndim!=3 or a.shape[-1]!=2 or a.shape[1]%5:
                quality.append(dict(array=name,issue="unexpected_contour_layout"))
            ts=np.asarray(group[name.replace("data/","timestamps/",1)][:])
            if len(ts)!=len(a):quality.append(dict(array=name,issue="contour_timestamp_length_mismatch"))
        if name.startswith("data/2d_brightness_mode"):
            ts=np.asarray(group[name.replace("data/","timestamps/",1)][:])
            if a.ndim!=3 or len(ts)!=len(a):quality.append(dict(array=name,issue="image_layout_or_timestamp_mismatch"))
    return results,quality



def completed(store):
    if not (DEST/"raw"/store["path"]).is_dir():return False
    try:
        for member in store.get("members",[]):
            p=DEST/"raw"/member["name"]
            if not p.is_file() or p.stat().st_size!=member["size"]:return False
        inspect_store(store,DEST/"raw")
    except Exception:
        return False
    return True


def intervals(files):
    spans=[]
    for e in sorted(files,key=lambda e:e["offset"]):
        for a in range(e["offset"],e["offset"]+e["size"],8*MiB):
            b=min(e["offset"]+e["size"],a+8*MiB)
            if spans and a-spans[-1][1]<=64*1024 and b-spans[-1][0]<=8*MiB:spans[-1][1]=b
            else:spans.append([a,b])
    return spans



def commit_staging(store):
    staging=DEST/".staging"
    for e in store["members"]:
        p=staging/e["name"]
        if not p.is_file() or p.stat().st_size!=e["size"]:raise ValueError("incomplete_staging_members")
    inspect_store(store,staging)
    final=DEST/"raw"/store["path"];final.parent.mkdir(parents=True,exist_ok=True)
    if final.exists():shutil.rmtree(final)
    (staging/store["path"]).replace(final)


def fetch_exam(row, stores):
    needed=[s for s in stores if not completed(s)]
    if not needed:return dict(exam=row["exam_id"],status="already_verified",stores=len(stores),network_bytes=0)
    attach_members(needed,index_exam(row))
    # Recover interrupted stores only after checking all members and array hashes.
    remaining=[]
    for s in needed:
        if not (DEST/".staging"/s["path"]).exists():remaining.append(s);continue
        try:commit_staging(s);event("staging_recovered",recording_id=s["recording_id"])
        except Exception:remaining.append(s)
    needed=remaining
    if not needed:return dict(exam=row["exam_id"],status="staging_verified_recovered",stores=len(stores),network_bytes=0)
    reserve(sum(s["payload_bytes"] for s in needed))
    staging=DEST/".staging";files=[e for s in needed for e in s["members"]]
    for s in needed:
        p=staging/s["path"]
        if p.exists():shutil.rmtree(p)
    for e in files:
        p=staging/e["name"];p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("wb") as f:f.truncate(e["size"])
    remote=make_remote(row["exam_id"],row["archive_bytes"]);started=time.monotonic();last=started
    try:
        for a,b in intervals(files):
            reserve();block=remote.fetch(a,b-1)
            for e in files:
                lo,hi=max(a,e["offset"]),min(b,e["offset"]+e["size"])
                if hi>lo:
                    with (staging/e["name"]).open("r+b") as f:
                        f.seek(lo-e["offset"]);f.write(block[lo-a:hi-a])
            if time.monotonic()-last>30:
                event("transfer_progress",exam=row["exam_id"],network_bytes=remote.network_bytes,seconds=round(time.monotonic()-started,1));last=time.monotonic()
        for s in needed:
            commit_staging(s)
        return dict(exam=row["exam_id"],status="downloaded_verified",stores=len(needed),payload_bytes=sum(s["payload_bytes"] for s in needed),
                    network_bytes=remote.network_bytes,requests=remote.requests_count,seconds=round(time.monotonic()-started,1))
    finally:
        event("exam_transfer_total",exam=row["exam_id"],network_bytes=remote.network_bytes,requests=remote.requests_count)
        remote.close()



def fetch_all(workers, limit=None):
    plan,catalog,folds=build_plan()
    raw=DEST/"raw";raw.mkdir(parents=True,exist_ok=True)
    if not (raw/"croissant.json").exists():dump(raw/"croissant.json",catalog)
    if not (raw/"splits.csv").exists():
        with (raw/"splits.csv").open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["exam_id","five_fold_split"])
            w.writerows((e["exam_id"],folds[e["exam_id"]]) for e in plan["exams"])
    if not (raw/"LICENSE").exists():shutil.copy2(DEST/"source/LICENSE",raw/"LICENSE")
    by_exam=defaultdict(list)
    for store in plan["stores"]:by_exam[store["exam_id"]].append(store)
    rows=plan["exams"][:limit] if limit else plan["exams"]
    selected=[s for row in rows for s in by_exam[row["exam_id"]]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        valid=dict(zip((s["recording_id"] for s in selected),pool.map(completed,selected)))
    needed=[row for row in rows if any(not valid[s["recording_id"]] for s in by_exam[row["exam_id"]])]
    failures=[]
    if needed:
        sizes=archive_sizes()
        for row in needed:row["archive_bytes"]=sizes[row["exam_id"]]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(fetch_exam,row,by_exam[row["exam_id"]]):row["exam_id"] for row in needed}
            for f in as_completed(fs):
                try:event("exam_complete",**f.result())
                except Exception as e:
                    failures.append(dict(exam=fs[f],error=str(e)));event("exam_failed",**failures[-1])
    if failures:raise RuntimeError(f"{len(failures)} exam failures; rerun fetch")
    # Empty staging directories carry no resume information.
    staging=DEST/".staging"
    if staging.exists():
        for p in sorted(staging.rglob("*"),key=lambda p:len(p.parts),reverse=True):
            if p.is_dir() and not any(p.iterdir()):p.rmdir()
        if not any(staging.iterdir()):staging.rmdir()
    event("fetch_finished",requested_exams=len(rows),downloaded_exams=len(needed),already_verified_exams=len(rows)-len(needed))


def verify_all(workers):
    """Check original arrays and reading metadata; print results without reports."""
    plan,catalog,folds=build_plan();failures=[];results=[]
    def verify(store):
        arrays,notes=inspect_store(store,DEST/"raw")
        files=[p for p in (DEST/"raw"/store["path"]).rglob("*") if p.is_file()]
        return dict(arrays=len(arrays),files=len(files),bytes=sum(p.stat().st_size for p in files),source_notes=len(notes))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(verify,s):s["recording_id"] for s in plan["stores"]}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as e:failures.append(dict(recording_id=fs[f],error=str(e)))
            if (len(results)+len(failures))%100==0:event("verify_progress",stores=len(results),failures=len(failures))
    expected={s["path"] for s in plan["stores"]}
    actual={str(p.relative_to(DEST/"raw")) for p in (DEST/"raw/exams").rglob("*.zarr") if p.is_dir()}
    metadata_ok=read(DEST/"raw/croissant.json")["recordSet"]==catalog["recordSet"]
    with (DEST/"raw/splits.csv").open() as f:
        actual_folds={r["exam_id"]:int(r["five_fold_split"]) for r in csv.DictReader(f)}
    metadata_ok &= actual_folds=={e["exam_id"]:folds[e["exam_id"]] for e in plan["exams"]}
    metadata_ok &= sha(DEST/"raw/LICENSE")==sha(DEST/"source/LICENSE")
    result=dict(complete=not failures and expected==actual and metadata_ok,stores=len(results),
        arrays=sum(r["arrays"] for r in results),files=sum(r["files"] for r in results),
        payload_bytes=sum(r["bytes"] for r in results),source_notes=sum(r["source_notes"] for r in results),
        failures=failures,missing_stores=sorted(expected-actual),unexpected_stores=sorted(actual-expected),
        reading_metadata_matches_official_selection=bool(metadata_ok))
    event("verification_complete",**result)
    if not result["complete"]:raise RuntimeError("verification_failed")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["plan","fetch","verify"])
    p.add_argument("--workers",type=int,default=4);p.add_argument("--limit",type=int)
    a=p.parse_args()
    if a.workers<1 or (a.limit is not None and a.limit<1):p.error("workers and limit must be positive")
    if a.command=="plan":event("selection",**build_plan()[0]["counts"])
    elif a.command=="fetch":fetch_all(a.workers,a.limit)
    else:verify_all(a.workers)


if __name__=="__main__":main()

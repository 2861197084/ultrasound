"""Original data integrity and Range downloads; no real dataset is modified."""
import json
from pathlib import Path
import sys
import tempfile
import io
import tarfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import download_echoxflow_raw as d
from download_echoxflow_raw import RemoteTar


class IntegrityTests(unittest.TestCase):

    def test_range_partition_covers_large_and_empty_members_once(self):
        files=[dict(offset=512,size=0),dict(offset=1024,size=500),dict(offset=8192,size=19*d.MiB),dict(offset=30*d.MiB,size=1)]
        spans=d.intervals(files)
        self.assertTrue(all(0<b-a<=8*d.MiB for a,b in spans))
        for f in files:
            self.assertEqual(sum(max(0,min(b,f["offset"]+f["size"])-max(a,f["offset"])) for a,b in spans),f["size"])


    def test_bootstrap_uses_only_original_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";source.mkdir()
            for name in ("croissant.json","splits.csv","LICENSE"):
                (source/name).write_text("canonical official metadata")
            catalog_sha=d.sha(source/"croissant.json")
            before={p.name:(d.sha(p),p.stat().st_mtime_ns) for p in source.iterdir()}
            with patch.object(d,"DEST",root),patch.object(d,"CATALOG_SHA",catalog_sha):
                self.assertEqual(d.source_bootstrap(),source)
                self.assertEqual(before,{p.name:(d.sha(p),p.stat().st_mtime_ns) for p in source.iterdir()})
                self.assertEqual(sorted(p.name for p in root.iterdir()),["source"])
                (source/"croissant.json").write_text("corrupted")
                with self.assertRaisesRegex(RuntimeError,"source_catalog_checksum_mismatch"):d.source_bootstrap()
                (source/"splits.csv").unlink()
                with self.assertRaisesRegex(RuntimeError,"missing_pinned_source_metadata"):d.source_bootstrap()


    def test_tar_path_and_bounds(self):
        for name in ("/etc/passwd","exams/e/../../outside","exams/other/x"):
            with self.assertRaises(ValueError):d.safe_name(name,"e")
        with self.assertRaises(ValueError):
            d.check_index(dict(revision=d.REV,archive_bytes=100,members=[dict(name="exams/e/x",offset=99,size=2)]),"e",100)

    def test_official_record_hash_and_resume_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,"DEST",Path(tmp)):
            root=Path(tmp)/".staging";path="exams/exam_test/recording_test.zarr"
            g=zarr.open_group(str(root/path),mode="w",zarr_format=2)
            a=np.arange(8,dtype=np.float32)
            g.create_array("data/ecg",data=a)
            decl=dict(name="data/ecg",shape=[8],dtype="float32",data_sha256=d.array_sha(a))
            g.attrs.update(dict(recording_id="recording_test",exam_id="exam_test",arrays=[decl]))
            info={"arrays/array_path":"data/ecg","arrays/shape":[8],"arrays/dtype":"float32","arrays/data_sha256":""}
            store=dict(path=path,recording_id="recording_test",exam_id="exam_test",arrays=[info])
            out,_=d.inspect_store(store,root)
            self.assertEqual(out[0]["checksum_source"],"official_recording_attributes")
            info["arrays/data_sha256"]="0"*64
            with self.assertRaises(ValueError):d.inspect_store(store,root)
            info["arrays/data_sha256"]=""
            g["data/ecg"][0]=99
            with self.assertRaises((AssertionError,ValueError)):d.inspect_store(store,root)
            g["data/ecg"][0]=0
            members=[dict(name=str(p.relative_to(root)),size=p.stat().st_size,offset=0) for p in (root/path).rglob("*") if p.is_file()]
            store.update(members=members,payload_bytes=sum(m["size"] for m in members))
            d.commit_staging(store)
            self.assertTrue(d.completed(store))
            self.assertFalse((Path(tmp)/"state").exists())
            chunk=next(p for p in (Path(tmp)/"raw"/path/"data/ecg").iterdir() if not p.name.startswith("."))
            data=bytearray(chunk.read_bytes());data[-1]^=1;chunk.write_bytes(data)
            self.assertFalse(d.completed(store))

    def test_range_download_and_resume_need_no_saved_reports(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,"DEST",Path(tmp)):
            root=Path(tmp);source=root/"input";path="exams/exam_test/recording_test.zarr"
            group=zarr.open_group(str(source/path),mode="w",zarr_format=2)
            values=np.arange(8,dtype=np.float32)
            group.create_array("data/ecg",data=values)
            group.attrs.update(dict(recording_id="recording_test",exam_id="exam_test",
                arrays=[dict(name="data/ecg",shape=[8],dtype="float32",data_sha256=d.array_sha(values))]))
            store=dict(path=path,recording_id="recording_test",exam_id="exam_test",arrays=[{
                "arrays/array_path":"data/ecg","arrays/shape":[8],"arrays/dtype":"float32","arrays/data_sha256":d.array_sha(values)}])
            buffer=io.BytesIO()
            with tarfile.open(fileobj=buffer,mode="w") as archive:
                for p in sorted((source/path).rglob("*")):
                    if p.is_file():archive.add(p,arcname=str(p.relative_to(source)))
            payload=buffer.getvalue()
            class Remote(io.BytesIO):
                network_bytes=0
                requests_count=0
                def fetch(self,start,end):
                    assert 0<=start<=end<len(payload)
                    self.network_bytes+=end-start+1;self.requests_count+=1
                    return payload[start:end+1]
            row=dict(exam_id="exam_test",archive_bytes=len(payload))
            with patch.object(d,"make_remote",side_effect=lambda *args:Remote(payload)):
                result=d.fetch_exam(row,[store])
            self.assertEqual(result["status"],"downloaded_verified")
            self.assertTrue(d.completed(store))
            for p in (source/path).rglob("*"):
                if p.is_file():self.assertEqual(p.read_bytes(),(root/"raw"/p.relative_to(source)).read_bytes())
            with patch.object(d,"make_remote",side_effect=AssertionError("verified data must be reused")):
                self.assertEqual(d.fetch_exam(row,[store])["status"],"already_verified")
            self.assertFalse(list(root.rglob("*.json")))

    def test_http_200_is_rejected_without_body_read(self):
        class Response:
            status_code=200
            headers={}
            url="https://invalid.example/file"
            def __enter__(self):return self
            def __exit__(self,*args):pass
            @property
            def raw(self):raise AssertionError("whole-object response must never be read")
        class Session:
            def get(self,*args,**kw):return Response()
            def close(self):pass
        remote=RemoteTar("https://invalid.example/file",1024)
        remote.session.close();remote.session=Session()
        with patch("download_echoxflow_raw.time.sleep",lambda _:None):
            with self.assertRaises(RuntimeError):remote.fetch(0,15)
        self.assertEqual(remote.network_bytes,0)
        remote.close()


if __name__=="__main__":unittest.main(verbosity=2)

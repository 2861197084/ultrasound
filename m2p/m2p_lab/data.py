from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .geometry import boundary_distances, grouped_queries
from .runtime import ROOT, RUN, atomic_json


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "little")


def find_folder(root, name):
    paths = [p for p in root.rglob(name) if p.is_dir() and (p / "JPEGImages").exists()]
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {name}/JPEGImages directory under {root}, found {paths}")
    return paths[0]


def build_vos_records(root, split, dataset):
    records = []
    if dataset == "youtubevos":
        folder = find_folder(root, split)
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            candidates = list(root.rglob(f"{split}/meta.json"))
            if len(candidates) != 1:
                raise RuntimeError("YouTube-VOS metadata missing")
            meta_path = candidates[0]
        videos = json.loads(meta_path.read_text())["videos"]
        for video, info in sorted(videos.items()):
            objects = []
            visible_ids = {}
            if split != "train":
                for p in (folder / "Annotations" / video).glob("*.png"):
                    with Image.open(p) as im:
                        visible_ids[p.stem] = set(int(x) for x in np.unique(np.asarray(im)))
            for oid, detail in sorted(info["objects"].items()):
                frames = [f for f in detail["frames"] if (folder / "Annotations" / video / (f + ".png")).exists()
                          and (folder / "JPEGImages" / video / (f + ".jpg")).exists()
                          and (split == "train" or int(oid) in visible_ids.get(f, set()))]
                if frames:
                    objects.append(dict(object_id=int(oid), frames=sorted(set(frames))))
            if objects:
                records.append(dict(dataset=dataset, split=split, video=video, root=str(folder), objects=objects))
    else:
        folder = root / "DAVIS"
        videos = (folder / "ImageSets/2017" / (split + ".txt")).read_text().split()
        for video in videos:
            annotation_dir = folder / "Annotations/480p" / video
            objects = {}
            for mask_path in sorted(annotation_dir.glob("*.png")):
                with Image.open(mask_path) as image:
                    ids = np.unique(np.asarray(image))
                for oid in ids:
                    if 0 < oid < 255:
                        objects.setdefault(int(oid), []).append(mask_path.stem)
            records.append(dict(dataset=dataset, split=split, video=video, root=str(folder),
                                objects=[dict(object_id=oid, frames=frames) for oid, frames in sorted(objects.items())]))
    return records


def index_natural():
    train = build_vos_records(ROOT.parent / "data/youtubevos2019", "train", "youtubevos")
    val = build_vos_records(ROOT.parent / "data/youtubevos2019", "valid", "youtubevos")
    davis = build_vos_records(ROOT.parent / "data/natural", "train", "davis")
    train = [r for r in train if any(len(o["frames"]) >= 2 for o in r["objects"])]
    if len(train) != 3471 or len(davis) != 60:
        raise RuntimeError(f"Unexpected training video count: YouTube-VOS={len(train)}, DAVIS={len(davis)}")
    if {r["video"] for r in train} & {r["video"] for r in val}:
        raise RuntimeError("YouTube-VOS training/validation overlap")
    davis_val = set((ROOT.parent / "data/natural/DAVIS/ImageSets/2017/val.txt").read_text().split())
    if {r["video"] for r in davis} & davis_val:
        raise RuntimeError("DAVIS training/TAP-Vid test overlap")
    real_val = [r for r in val if any(len(o["frames"]) >= 2 for o in r["objects"])]
    # Official VOS validation releases often expose only initialization masks.
    # Synthetic pairs stay within official validation images, never TAP-Vid data.
    mode = "real_temporal_masks" if real_val else "synthetic_affine_from_official_validation_masks"
    chosen_val = real_val if real_val else val
    payload = dict(train=train + davis, validation=chosen_val, validation_mode=mode,
                   counts=dict(youtube_train=len(train), davis_train=len(davis),
                               youtube_validation=len(chosen_val), validation_real_pair_videos=len(real_val)),
                   note="Validation masks select checkpoints; no benchmark point annotations are used for model selection.")
    atomic_json(RUN / "natural_manifest.json", payload)
    return payload


def load_frame(record, frame):
    root, video = Path(record["root"]), record["video"]
    infix = "480p" if record["dataset"] == "davis" else ""
    with Image.open(root / "JPEGImages" / infix / video / (frame + ".jpg")) as im:
        image = im.convert("RGB")
    with Image.open(root / "Annotations" / infix / video / (frame + ".png")) as im:
        annotation = np.array(im)
    return image, annotation


def affine_pair(image, mask, cfg, rng, params=None):
    size = cfg["train_size"]
    if params is None:
        params = (float(rng.uniform(-cfg["affine_degrees"], cfg["affine_degrees"])),
                  [int(rng.uniform(-cfg["affine_translate"], cfg["affine_translate"]) * size) for _ in range(2)],
                  float(rng.uniform(*cfg["affine_scale"])))
    angle, translate, scale = params
    image = TF.affine(image, angle, translate, scale, [0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
    mask = TF.affine(mask, angle, translate, scale, [0.0, 0.0], interpolation=InterpolationMode.NEAREST, fill=0)
    return image, mask, params


def prepare_pair(im0, im1, mask0, mask1, cfg, rng, augment=False, synthetic=False):
    size = cfg["train_size"]
    images = [im.resize((size, size), Image.Resampling.BILINEAR) for im in (im0, im1)]
    masks = [Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
             for mask in (mask0, mask1)]
    if synthetic:
        images[1], masks[1], _ = affine_pair(images[1], masks[1], cfg, rng)
    if augment:
        for i in range(2):
            images[i], masks[i], _ = affine_pair(images[i], masks[i], cfg, rng)
            strength = cfg["color_jitter"]
            images[i] = TF.adjust_brightness(images[i], float(rng.uniform(1 - strength, 1 + strength)))
            images[i] = TF.adjust_contrast(images[i], float(rng.uniform(1 - strength, 1 + strength)))
            images[i] = TF.adjust_saturation(images[i], float(rng.uniform(1 - strength, 1 + strength)))
            images[i] = TF.adjust_hue(images[i], float(rng.uniform(-0.05, 0.05)))
    masks = [np.asarray(m) > 0 for m in masks]
    if min(m.sum() for m in masks) < cfg["groups"] * cfg["points_per_group"]:
        raise ValueError("Foreground too small after transformation")
    queries = grouped_queries(masks[0], rng, cfg["groups"], cfg["points_per_group"],
                              cfg["sample_max_pixels"], cfg["kmeans_iterations"])
    return dict(image0=TF.to_tensor(images[0]), image1=TF.to_tensor(images[1]),
                mask0=torch.from_numpy(masks[0].copy()).float(), mask1=torch.from_numpy(masks[1].copy()).float(),
                queries=torch.from_numpy(queries),
                distance0=torch.from_numpy(boundary_distances(masks[0])),
                distance1=torch.from_numpy(boundary_distances(masks[1])))


class VOSPairs(Dataset):
    def __init__(self, records, cfg, validation=False, synthetic_validation=False, length=None):
        self.records, self.cfg, self.validation = records, cfg, validation
        self.synthetic = synthetic_validation
        self.length = length or (len(records) * cfg["validation_pairs_per_video"] if validation else cfg["natural_steps"] * cfg["effective_batch"])
        if not records:
            raise ValueError("No video records")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(stable_seed((self.cfg["seed"], "val" if self.validation else "train", index)))
        fixed_video = (index // self.cfg["validation_pairs_per_video"]) % len(self.records)
        last_error = None
        for attempt in range(50):
            # Validation stays in its designated video even if a pair is invalid.
            record = self.records[fixed_video if self.validation else int(rng.integers(len(self.records))) ]
            objects = [o for o in record["objects"] if len(o["frames"]) >= (1 if self.synthetic else 2)]
            if not objects:
                continue
            obj = objects[int(rng.integers(len(objects)))]
            frames = rng.choice(obj["frames"], 2, replace=self.synthetic)
            if self.synthetic:
                frames[1] = frames[0]
            try:
                image0, ann0 = load_frame(record, frames[0])
                image1, ann1 = load_frame(record, frames[1])
                pair = prepare_pair(image0, image1, ann0 == obj["object_id"], ann1 == obj["object_id"],
                                    self.cfg, rng, augment=not self.validation, synthetic=self.synthetic)
                pair["record_id"] = record["dataset"] + ":" + record["video"]
                return pair
            except ValueError as error:
                last_error = error
        raise RuntimeError(f"Cannot sample valid pair for index {index}, video {record['video']}: {last_error}")

"""Official-training-only, class/track-disjoint GTSRB splits in raw pixel space."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
from PIL import Image
import torch

TRAIN_PREFIX = "GTSRB/Final_Training/Images"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def training_records(archive):
    records = []
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP paths")
        names = set(names)
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "Test" in name:
                raise ValueError("unsafe or test archive member")
        for label in range(43):
            base = f"{TRAIN_PREFIX}/{label:05d}"
            with source.open(f"{base}/GT-{label:05d}.csv") as handle:
                for row in csv.DictReader(io.TextIOWrapper(handle), delimiter=";"):
                    filename = row["Filename"]
                    if not re.fullmatch(r"\d{5}_\d{5}\.ppm", filename) or int(row["ClassId"]) != label:
                        raise ValueError("invalid GTSRB track/class annotation")
                    image_id = f"{base}/{filename}"
                    if image_id not in names:
                        raise ValueError("missing training image")
                    records.append({"image_id": image_id, "track_id": f"{label:05d}/{filename.split('_')[0]}", "label": label})
    records.sort(key=lambda r: r["image_id"])
    if len({r["image_id"] for r in records}) != len(records):
        raise ValueError("duplicate image IDs")
    return records


def make_split(records, *, seed=42, target=4000, probe_seed=20260920):
    tracks = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(records):
        tracks[row["label"]][row["track_id"]].append(index)
    rng = np.random.default_rng(seed)
    fraction = target / len(records)
    private, dev = [], []
    allocation = {}
    for label, groups in sorted(tracks.items()):
        ordered = sorted(groups)
        if len(ordered) < 2:
            raise ValueError("each class needs at least two tracks")
        rng.shuffle(ordered)
        # Prefix closest to proportional image target; never split a track.
        class_target = fraction * sum(len(v) for v in groups.values())
        counts = np.cumsum([len(groups[t]) for t in ordered])
        n = min(range(1, len(ordered)), key=lambda k: (abs(counts[k-1] - class_target), k))
        selected = set(ordered[:n])
        for track, ids in groups.items():
            (dev if track in selected else private).extend(ids)
        allocation[str(label)] = {"private_images": sum(len(groups[t]) for t in ordered[n:]),
                                  "dev_images": int(counts[n-1]), "private_tracks": len(ordered)-n, "dev_tracks": n}
    private, dev = sorted(private), sorted(dev)
    ordered = np.random.default_rng(probe_seed).permutation(dev).tolist()
    if len(ordered) < 1024:
        raise ValueError("insufficient public development data")
    indices = {"private": private, "dev": dev, "probe": ordered[:512], "trajectory": ordered[:1024]}
    split = {"records": records, "indices": indices, "class_counts": allocation, "split_seed": seed,
             "probe_seed": probe_seed, "target_dev_images": target, "split_unit": "class_and_track",
             "privacy_unit": "image", "official_test_loaded": False}
    split["hashes"] = {name: digest([records[i] for i in ids]) for name, ids in indices.items()}
    split["split_hash"] = digest({"records": records, "indices": indices, "seed": seed})
    validate_split(split)
    return split


def validate_split(split):
    rows, ids = split["records"], split["indices"]
    private, dev = set(ids["private"]), set(ids["dev"])
    if private & dev or private | dev != set(range(len(rows))):
        raise ValueError("incomplete or overlapping image split")
    if {rows[i]["track_id"] for i in private} & {rows[i]["track_id"] for i in dev}:
        raise ValueError("track leakage")
    for name, size in [("probe", 512), ("trajectory", 1024)]:
        if len(ids[name]) != size or len(set(ids[name])) != size or not set(ids[name]) <= dev:
            raise ValueError("invalid public subset")
    for name, values in ids.items():
        if split["hashes"][name] != digest([rows[i] for i in values]):
            raise ValueError("ordered subset hash mismatch")
    if split["official_test_loaded"]:
        raise ValueError("official test forbidden")


class GTSRBTrainOnly:
    def __init__(self, archive, split_path, augmentation):
        self.split = json.loads(Path(split_path).read_text())
        validate_split(self.split)
        self.augmentation = augmentation
        # Decode/resample once on the remote VM, not for every Poisson minibatch.
        self.pixels = np.empty((len(self.split["records"]), 96, 96, 3), dtype=np.uint8)
        with zipfile.ZipFile(archive) as source:
            for index, row in enumerate(self.split["records"]):
                with source.open(row["image_id"]) as handle:
                    image = Image.open(handle).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
                    self.pixels[index] = np.asarray(image)

    def batch(self, ids, *, augment=False):
        from torchvision import transforms
        pixels = torch.from_numpy(self.pixels[np.asarray(ids)].copy()).permute(0, 3, 1, 2).float().div_(255)
        labels = torch.tensor([self.split["records"][int(i)]["label"] for i in ids], dtype=torch.long)
        if augment:
            a = self.augmentation
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=a["degrees"], translate=tuple(a["translate"]), scale=tuple(a["scale"]),
                                        interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=a["brightness"], contrast=a["contrast"], saturation=a["saturation"], hue=a["hue"]),
            ])
            pixels = torch.stack([transform(x) for x in pixels])
        if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("images must remain unnormalized [0,1] pixels")
        return pixels, labels

"""Train-only artifacts: official test bytes are never packaged for tuning."""
from __future__ import annotations

import hashlib
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

SPLIT_SEED = 20260919


def indices_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def prepare_train_only(archive, destination):
    """Read precisely the public CIFAR train member, never extract the test member."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "train_only.npz"
    if target.exists():
        return json.loads((destination / "splits.json").read_text())
    with tarfile.open(archive) as tar:
        member = tar.extractfile("cifar-100-python/train")
        raw = pickle.load(member, encoding="bytes")
    x = raw[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1)
    y = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if x.shape != (50000,32,32,3) or len(np.unique(y)) != 100:
        raise ValueError("unexpected CIFAR-100 train shape")
    rng = np.random.default_rng(SPLIT_SEED)
    dev, private, probe = [], [], []
    extra = set(rng.permutation(100)[:12].tolist())
    for label in range(100):
        ids = rng.permutation(np.flatnonzero(y == label))
        dev.extend(ids[:50].tolist())
        private.extend(ids[50:].tolist())
        probe.extend(ids[:6 if label in extra else 5].tolist())
    arrays = {"private": sorted(private), "dev": sorted(dev), "probe": sorted(probe)}
    if len(set(private) & set(dev)) or not set(probe) <= set(dev):
        raise ValueError("split overlap")
    split = {"split_seed": SPLIT_SEED, "indices": arrays,
             "hashes": {k:indices_hash(v) for k,v in arrays.items()},
             "counts": {k:len(v) for k,v in arrays.items()}, "official_test_in_bundle": False}
    np.savez_compressed(target, pixels=x, labels=y)
    (destination / "splits.json").write_text(json.dumps(split, indent=2)+"\n")
    return split


class TrainOnlyData:
    def __init__(self, root):
        root = Path(root)
        self.split = json.loads((root / "splits.json").read_text())
        raw = np.load(root / "train_only.npz", allow_pickle=False)
        self.pixels, self.labels = raw["pixels"], raw["labels"]
        for name, ids in self.split["indices"].items():
            if indices_hash(ids) != self.split["hashes"][name]:
                raise ValueError("split hash mismatch")
        self.transform = None

    def batch(self, ids, *, augment=False):
        if augment:
            if self.transform is None:
                from timm.data.auto_augment import rand_augment_transform
                # Source appendix A.2: 2 operations, magnitude 9, p=.5 per op.
                # No MixUp/CutMix: preserve independent per-record queries.
                randaug = rand_augment_transform("rand-m9-n2", {
                    "translate_const": int(32*.45), "img_mean": (124,116,104),
                    "interpolation": Image.Resampling.BICUBIC})
                self.transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                    transforms.RandomHorizontalFlip(), randaug, transforms.ToTensor()])
            x = torch.stack([self.transform(Image.fromarray(self.pixels[i])) for i in ids])
        else:
            x = torch.from_numpy(self.pixels[np.asarray(ids)].copy()).permute(0,3,1,2).float()/255
        return x, torch.from_numpy(self.labels[np.asarray(ids)].copy()).long()

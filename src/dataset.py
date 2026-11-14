"""Dataset helpers: build a PERSON-ONLY YOLO dataset and its yaml."""
from __future__ import annotations

import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_coco128() -> Path:
    dl = DATA / "_dl"
    target = dl / "coco128"
    if target.exists():
        return target
    dl.mkdir(parents=True, exist_ok=True)
    z = dl / "coco128.zip"
    print(f"Downloading {COCO128_URL} ...")
    urllib.request.urlretrieve(COCO128_URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(dl)
    return target


def build_person_dataset(source: Path, person_ids=(0,), val_frac=0.2, seed=0, out=None) -> Path:
    """Copy `source` (YOLO format: images/**, labels/**) into data/person_dataset keeping ONLY
    the person boxes (re-labelled as class 0). Images without people are kept as negatives."""
    out = Path(out) if out else DATA / "person_dataset"
    if out.exists():
        shutil.rmtree(out)
    imgs = sorted(p for p in (source / "images").rglob("*") if p.suffix.lower() in IMG_EXT)
    rng = random.Random(seed)
    rng.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_frac))
    splits = {"val": imgs[:n_val], "train": imgs[n_val:]}
    stats = {}
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_box = 0
        for im in items:
            rel = im.relative_to(source / "images")
            lab = (source / "labels" / rel).with_suffix(".txt")
            lines = []
            if lab.exists():
                for ln in lab.read_text().splitlines():
                    p = ln.split()
                    if p and int(float(p[0])) in person_ids:
                        lines.append("0 " + " ".join(p[1:5]))
            shutil.copy(im, out / "images" / split / im.name)
            (out / "labels" / split / (im.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            n_box += len(lines)
        stats[split] = (len(items), n_box)
    print(f"person dataset -> {out}\n  train: {stats['train'][0]} images, {stats['train'][1]} person boxes"
          f"\n  val  : {stats['val'][0]} images, {stats['val'][1]} person boxes")
    return out


def ensure_yaml(dataset: Path | None = None) -> Path:
    """(Re)write data/person.yaml with the correct absolute path (so the folder can be moved)."""
    dataset = Path(dataset) if dataset else DATA / "person_dataset"
    y = DATA / "person.yaml"
    y.write_text(f"path: {dataset.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    return y

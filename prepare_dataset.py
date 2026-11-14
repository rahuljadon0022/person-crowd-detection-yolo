"""Build the person-only training dataset.

    python prepare_dataset.py                       # downloads COCO128 and keeps only 'person'
    python prepare_dataset.py --source my_yolo_ds   # your own YOLO dataset (needs images/ and labels/)
    python prepare_dataset.py --source crowdhuman_yolo --person-ids 0

For a serious run use a big person dataset (COCO person subset, CrowdHuman, WiderPerson ...)
converted to YOLO format, and pass it with --source.
"""
import argparse
from pathlib import Path

from src.dataset import build_person_dataset, download_coco128, ensure_yaml

ap = argparse.ArgumentParser()
ap.add_argument("--source", default=None, help="YOLO-format dataset folder (images/, labels/)")
ap.add_argument("--person-ids", type=int, nargs="+", default=[0], help="class ids that mean 'person' in the source")
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

src = Path(a.source) if a.source else download_coco128()
ds = build_person_dataset(src, tuple(a.person_ids), a.val_frac, a.seed)
print("yaml:", ensure_yaml(ds))

#Above 

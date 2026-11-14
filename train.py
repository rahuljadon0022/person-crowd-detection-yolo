"""Fine-tune YOLO as a PERSON-ONLY detector.

    python train.py                                   # defaults (CPU friendly)
    python train.py --model weights/yolo11l.pt --epochs 100 --imgsz 960 --batch 8 --device 0   # GPU run

The best checkpoint is copied to weights/best_person.pt. Use it with:  python detect.py --weights weights/best_person.pt ...
IMPORTANT: only prefer it over the pretrained yolo11l.pt if evaluate.py shows it is better (it needs a LARGE person dataset to win).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from src.dataset import ensure_yaml

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "weights" / "yolo11s.pt"))
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default=None, help="'cpu', '0' (GPU) ...")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--freeze", type=int, default=10, help="freeze first N layers (backbone) - good for small data")
ap.add_argument("--time", type=float, default=None, help="max training time in hours (overrides epochs)")
ap.add_argument("--name", default="person_yolo")
a = ap.parse_args()

data = ensure_yaml()
model = YOLO(a.model)
kw = dict(
    data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    workers=a.workers, freeze=a.freeze, project=str(ROOT / "runs"), name=a.name, exist_ok=True,
    optimizer="AdamW", lr0=5e-4, cos_lr=True, patience=15, plots=True, seed=0, deterministic=True,
    # augmentations that help crowded scenes
    mosaic=1.0, close_mosaic=5, fliplr=0.5, scale=0.5, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
)
if a.time:
    kw["time"] = a.time
model.train(**kw)

best = ROOT / "runs" / a.name / "weights" / "best.pt"
if best.exists():
    shutil.copy(best, ROOT / "weights" / "best_person.pt")
    print("saved ->", ROOT / "weights" / "best_person.pt")

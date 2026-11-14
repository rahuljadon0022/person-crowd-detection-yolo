"""YOLO person detector + crowd pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .crowd import CrowdResult, group_people
from .draw import draw_result

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "weights" / "yolo11l.pt"     # most accurate (pretrained COCO, we keep only class 0 = person)
FALLBACK_WEIGHTS = ROOT / "weights" / "yolo11s.pt"    # faster

PERSON_CLASS = 0  # class 0 is "person" both in COCO and in our fine-tuned 1-class model


def resolve_weights(path: str | None) -> str:
    if path:
        return str(path)
    for cand in (DEFAULT_WEIGHTS, FALLBACK_WEIGHTS):
        if cand.exists():
            return str(cand)
    return str(DEFAULT_WEIGHTS)  # not found -> ultralytics downloads it into weights/ on first run


class PersonCrowdPipeline:
    def __init__(
        self,
        weights: str | None = None,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 960,
        device: str | None = None,
        tta: bool = False,
        link_factor: float = 1.5,
        min_group_size: int = 2,
    ):
        from ultralytics import YOLO

        self.weights = resolve_weights(weights)
        self.model = YOLO(self.weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device, self.tta = device, tta
        self.link_factor, self.min_group_size = link_factor, min_group_size

    def detect_persons(self, frame_bgr: np.ndarray):
        """Return (boxes_xyxy [N,4], scores [N]) for PERSONS ONLY."""
        res = self.model.predict(
            frame_bgr,
            classes=[PERSON_CLASS],      # <- nothing but people
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            augment=self.tta,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()

    def process(self, frame_bgr: np.ndarray, mode: str = "crowd", show_isolated: bool = False):
        """mode='crowd'  -> highlight crowds only (isolated people are not marked)
        mode='person' -> plain person detection, every person boxed
        Returns (annotated_frame, info_dict)."""
        boxes, scores = self.detect_persons(frame_bgr)
        crowd = group_people(boxes, self.link_factor, self.min_group_size)
        out = draw_result(frame_bgr, boxes, scores, crowd, mode=mode, show_isolated=show_isolated)
        info = {
            "persons": int(len(boxes)),
            "crowds": crowd.n_crowds,
            "isolated": crowd.n_isolated,
            "crowd_sizes": [g["size"] for g in crowd.groups],
        }
        return out, info

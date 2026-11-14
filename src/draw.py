"""Drawing helpers."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .crowd import CrowdResult


def _palette(k: int):
    cols = []
    for i in range(max(k, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return cols


def _label(img, text, x, y, color, scale):
    t = max(1, int(round(scale * 2)))
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    y0 = max(y, th + bl + 2)
    cv2.rectangle(img, (x, y0 - th - bl - 2), (x + tw + 4, y0), color, -1)
    cv2.putText(img, text, (x + 2, y0 - bl), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t, cv2.LINE_AA)


def draw_result(frame, boxes, scores, crowd: CrowdResult, mode="crowd", show_isolated=False):
    img = frame.copy()
    H, W = img.shape[:2]
    s = max(0.45, min(H, W) / 900)          # scale text/lines with image size
    th = max(1, int(round(s * 2)))
    cols = _palette(crowd.n_crowds)

    if mode == "person":
        for (x1, y1, x2, y2), sc in zip(boxes, scores):
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, (0, 200, 0), th)
            _label(img, f"person {sc:.2f}", p1[0], p1[1], (0, 200, 0), s * 0.6)
    else:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            gid = crowd.labels[i]
            if gid == -1:
                if show_isolated:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (170, 170, 170), 1)
                continue
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), cols[gid - 1], max(1, th - 1))
        for g in crowd.groups:
            x1, y1, x2, y2 = [int(v) for v in g["bbox"]]
            pad = int(6 * s)
            c = cols[g["id"] - 1]
            cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), c, th + 2)
            _label(img, f"Crowd {g['id']} ({g['size']} people)", x1 - pad, y1 - pad, c, s * 0.75)

    if mode == "person":
        txt = f"Persons: {len(boxes)}"
    else:
        txt = f"Crowds: {crowd.n_crowds}  |  People in crowds: {len(boxes) - crowd.n_isolated}  |  Isolated (ignored): {crowd.n_isolated}"
    (tw, tht), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, s * 0.8, th)
    cv2.rectangle(img, (0, 0), (tw + 16, tht + bl + 14), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, tht + 7), cv2.FONT_HERSHEY_SIMPLEX, s * 0.8, (255, 255, 255), th, cv2.LINE_AA)
    return img

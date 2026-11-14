"""Evaluate one or more weight files on the person-only validation split.

    python evaluate.py                                        # yolo11l vs yolo11s (+ best_person.pt if you trained one)
    python evaluate.py --weights weights/best_person.pt --split val --imgsz 960

Reports (person class only):
  * Precision / Recall / F1 at the operating confidence (IoU 0.5)
  * AP50 and AP50-95 (COCO style, own implementation so a pretrained 80-class model and a
    1-class fine-tuned model are scored in exactly the same way)
  * Crowd-level accuracy: the same grouping rule is applied to the ground-truth boxes and to the
    detections; we compare number of crowds (exact match, MAE) and per-person "in crowd / isolated"
    classification (precision / recall / F1).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.crowd import group_people
from src.dataset import DATA, ensure_yaml

ROOT = Path(__file__).resolve().parent
IOUS = np.linspace(0.5, 0.95, 10)


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def greedy_match(pred, gt, thr):
    """pred sorted by conf desc. returns tp flags and pred->gt index (or -1)."""
    tp = np.zeros(len(pred), bool); pair = -np.ones(len(pred), int)
    if len(pred) == 0 or len(gt) == 0:
        return tp, pair
    m = iou_mat(pred, gt); used = set()
    for i in range(len(pred)):
        order = np.argsort(-m[i])
        for j in order:
            if m[i, j] < thr:
                break
            if j not in used:
                used.add(j); tp[i] = True; pair[i] = j
                break
    return tp, pair


def average_precision(tp, conf, n_gt):
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    o = np.argsort(-conf); tp = tp[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(~tp)
    rec = ctp / n_gt; prec = ctp / (ctp + cfp)
    rec = np.concatenate([[0], rec, [1]]); prec = np.concatenate([[1], prec, [0]])
    prec = np.flip(np.maximum.accumulate(np.flip(prec)))
    x = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(x, rec, prec)))     # 101-point interpolation


def load_gt(lab: Path, W, H):
    out = []
    if lab.exists():
        for ln in lab.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                cx, cy, w, h = map(float, p[1:5])
                out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(out).reshape(-1, 4)


def evaluate(weights, split, imgsz, op_conf, iou_nms, link_factor, min_group, device):
    model = YOLO(weights)
    ds = DATA / "person_dataset"
    imgs = sorted((ds / "images" / split).glob("*"))
    all_tp = [[] for _ in IOUS]; all_conf = []; n_gt = 0
    tp_op = fp_op = fn_op = 0
    cnt_err = []; cnt_exact = 0
    mem = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for p in imgs:
        im = cv2.imread(str(p)); H, W = im.shape[:2]
        gt = load_gt(ds / "labels" / split / (p.stem + ".txt"), W, H)
        r = model.predict(im, classes=[0], conf=0.001, iou=0.6, imgsz=imgsz, device=device, verbose=False, max_det=300)[0]
        if r.boxes is not None and len(r.boxes):
            pb = r.boxes.xyxy.cpu().numpy(); pc = r.boxes.conf.cpu().numpy()
        else:
            pb = np.zeros((0, 4)); pc = np.zeros(0)
        o = np.argsort(-pc); pb, pc = pb[o], pc[o]
        n_gt += len(gt); all_conf.append(pc)
        for k, t in enumerate(IOUS):
            all_tp[k].append(greedy_match(pb, gt, t)[0])
        # operating point
        keep = pc >= op_conf; ob = pb[keep]
        tp, pair = greedy_match(ob, gt, 0.5)
        tp_op += tp.sum(); fp_op += (~tp).sum(); fn_op += len(gt) - tp.sum()
        # crowd level
        g_gt = group_people(gt, link_factor, min_group); g_pr = group_people(ob, link_factor, min_group)
        cnt_err.append(abs(g_gt.n_crowds - g_pr.n_crowds)); cnt_exact += int(g_gt.n_crowds == g_pr.n_crowds)
        for i, j in enumerate(pair):
            if j < 0:
                continue
            a = g_pr.labels[i] != -1; b = g_gt.labels[j] != -1
            mem["tp" if a and b else "fp" if a and not b else "fn" if b and not a else "tn"] += 1
    conf_all = np.concatenate(all_conf) if all_conf else np.zeros(0)
    aps = [average_precision(np.concatenate(all_tp[k]), conf_all, n_gt) for k in range(len(IOUS))]
    P = tp_op / max(tp_op + fp_op, 1); R = tp_op / max(tp_op + fn_op, 1)
    mp = mem["tp"] / max(mem["tp"] + mem["fp"], 1); mr = mem["tp"] / max(mem["tp"] + mem["fn"], 1)
    return {
        "weights": str(weights), "split": split, "images": len(imgs), "gt_persons": int(n_gt),
        "operating_conf": op_conf, "precision": round(P, 4), "recall": round(R, 4),
        "f1": round(2 * P * R / max(P + R, 1e-9), 4),
        "AP50": round(aps[0], 4), "AP50_95": round(float(np.mean(aps)), 4),
        "crowd_count_exact_match": round(cnt_exact / max(len(imgs), 1), 4),
        "crowd_count_MAE": round(float(np.mean(cnt_err)), 4),
        "crowd_member_precision": round(mp, 4), "crowd_member_recall": round(mr, 4),
        "crowd_member_f1": round(2 * mp * mr / max(mp + mr, 1e-9), 4),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=None)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.35, help="operating confidence for P/R and crowd metrics")
    ap.add_argument("--link-factor", type=float, default=1.5)
    ap.add_argument("--min-group", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "metrics.json"))
    a = ap.parse_args()
    ensure_yaml()
    ws = a.weights or [w for w in (ROOT / "weights" / "yolo11l.pt", ROOT / "weights" / "yolo11s.pt", ROOT / "weights" / "best_person.pt") if w.exists()]
    results = [evaluate(w, a.split, a.imgsz, a.conf, 0.5, a.link_factor, a.min_group, a.device) for w in ws]
    keys = ["precision", "recall", "f1", "AP50", "AP50_95", "crowd_count_exact_match", "crowd_count_MAE", "crowd_member_f1"]
    print(f"\n{'weights':32s}" + "".join(f"{k:>24s}" for k in keys))
    for r in results:
        print(f"{Path(r['weights']).name:32s}" + "".join(f"{r[k]:>24}" for k in keys))
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print("\nsaved ->", a.out)

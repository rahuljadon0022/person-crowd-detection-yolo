# Person & Crowd Detection with YOLO

Two tasks, one pipeline:

1. **Crowd detection** – find groups of people. A person sitting alone is *not* a crowd and is ignored.
   (Room with 2 groups + 1 isolated person → **2 crowds**, 1 isolated.)
2. **Person-only detection** – on an image, a video or the laptop camera, detect **only persons**
   (no chairs, bottles, laptops …). Done by restricting YOLO to class 0 (`person`).

## How it works

```
image / video frame / webcam
        │
   YOLO11 (class 0 = person only)  ──►  person boxes
        │
   crowd grouping (src/crowd.py)
        │   two people are linked if their box centres are closer than
        │   link_factor × (their own size)  → scale-invariant (near/far people both work)
        │   crowds = connected groups of linked people (>= min_group people)
        ▼
   crowds drawn in different colours, isolated people ignored
```

## Setup

```bash
pip install -r requirements.txt
```

The big weight files are **not inside the zip** (size limit). `weights/yolo11l.pt` is downloaded automatically from GitHub the first time you run
`detect.py` / `app.py` (needs internet once, ~50 MB). The fine-tuned demo model `finetuned_coco128_demo.pt` is sent as a separate file; put it in `weights/`.

## Run

```bash
# image (crowd mode)                         -> outputs/<name>_out.jpg
python detect.py --source samples/val_000000000257.jpg
# plain person detection (every person boxed)
python detect.py --source samples/val_000000000257.jpg --mode person
# a folder of images
python detect.py --source samples/
# video file                                  -> outputs/<name>_out.mp4
python detect.py --source my_video.mp4
# laptop camera (press q to quit)
python detect.py --source 0 --show
# browser app: upload image / upload video / live webcam
python app.py          # open http://127.0.0.1:7860
```

Useful options: `--conf 0.35` (detection threshold), `--imgsz 960` (bigger = better for far/small people),
`--tta` (more accurate, ~3x slower), `--link-factor 1.5` (bigger = looser groups),
`--min-group 2` (use 3 if you only want "3 or more people" to count as a crowd), `--show-isolated`.

**Speed vs accuracy.** Default model is `weights/yolo11l.pt` (most accurate, ~1.5-3 s/image on a CPU, real-time on a GPU).
For a smooth live webcam on a CPU use the small model: `python detect.py --source 0 --weights weights/yolo11s.pt --imgsz 640 --show`.

## Training and evaluation

```bash
python prepare_dataset.py        # builds a PERSON-ONLY dataset (data/person_dataset, 103 train / 25 val images)
python train.py --epochs 25 --device cpu     # fine-tune (GPU: --device 0 --imgsz 960 --epochs 100)
python evaluate.py               # precision / recall / AP / crowd-level accuracy on the val split
python tests/test_crowd.py       # unit tests of the grouping logic (2 groups + 1 isolated -> 2 crowds, ...)
```

`evaluate.py` scores every model the same way (person class only) and also checks the **crowd logic**: the same grouping
rule is applied to ground-truth boxes and to detections, and the number of crowds / who-is-in-a-crowd are compared.

### Results (same 25 validation images, 50 persons, conf 0.35, imgsz 960)

| model | precision | recall | AP50 | AP50-95 | crowd-count exact match | in-crowd vs isolated F1 |
|---|---|---|---|---|---|---|
| **yolo11l (pretrained, person only) – default** | 0.891 | 0.82 | **0.882** | **0.686** | **96 %** | **0.985** |
| yolo11s (pretrained, person only) | 0.927 | 0.76 | 0.859 | 0.651 | 96 % | 0.966 |
| yolo11m (pretrained, person only; weights not bundled) | 0.872 | 0.82 | 0.859 | 0.683 | 92 % | 0.968 |
| yolo11s fine-tuned by us on 103 images (`weights/finetuned_coco128_demo.pt`) | 0.673 | 0.74 | 0.735 | 0.471 | 88 % | 0.946 |

Raw numbers: `outputs/metrics_*.json`. Training curves: `runs/person_yolo/`.

### Please read – honest notes

* **Why the pretrained model is the default.** I did train/fine-tune a person-only model, but with only 103 training images
  it is *worse* than the pretrained COCO model (table above): the 1-class output layer has to be learned from scratch and
  the data is tiny. So the shipped detector is the pretrained model restricted to class 0, which is also the standard,
  reliable way to get "person only". To beat it you need a **large person/crowd dataset** (COCO person subset, CrowdHuman,
  WiderPerson …) and ideally a GPU: convert it to YOLO format, `python prepare_dataset.py --source <folder>`,
  `python train.py --device 0 --imgsz 960 --epochs 100`, then check `python evaluate.py` and use it with `--weights weights/best_person.pt`
  only if its numbers are better.
* **The validation numbers are optimistic and noisy.** The 128 images come from COCO train2017, which the pretrained models
  were trained on, and the val split is only 25 images / 50 persons. Treat the table as a sanity check and a way to
  compare settings, not as a final accuracy claim. For a real number, evaluate on your own room/video footage with labelled boxes.
* **What "crowd" means is a rule, not a learned class.** `link_factor` and `min-group` can be tuned to your scene
  (seated rows, distant cameras …). The webcam path uses the same code as the video path; it was tested on images, a video file
  and the web app's image tab, but not with a physical camera.

## Folder layout

```
crowd_detection/
├── detect.py            CLI: image / folder / video / webcam
├── app.py               browser app (upload image, upload video, live webcam)
├── prepare_dataset.py   person-only dataset builder
├── train.py             fine-tuning script
├── evaluate.py          metrics (detection + crowd level)
├── src/                 crowd.py (grouping), detector.py (YOLO + pipeline), draw.py, dataset.py
├── tests/test_crowd.py  unit tests
├── weights/             yolo11l.pt (default), yolo11s.pt (fast), finetuned_coco128_demo.pt
├── data/                person_dataset/ (+ person.yaml), original COCO128 in data/_dl/
├── samples/             test images + test video
├── outputs/             example results + metrics json
└── runs/                training logs / curves
```

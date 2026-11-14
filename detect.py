"""Run person / crowd detection on an image, a folder of images, a video file or the laptop camera.

    python detect.py --source samples/room.jpg                 # image  (crowd mode)
    python detect.py --source samples/                         # every image in a folder
    python detect.py --source my_video.mp4                     # video  -> outputs/my_video_out.mp4
    python detect.py --source 0 --show                         # laptop webcam (press q to quit)
    python detect.py --source room.jpg --mode person           # plain person detection (all people boxed)

Useful switches
    --conf 0.35        detection confidence (lower = finds more people, more false positives)
    --imgsz 960        inference size (bigger = better for small / far people, slower)
    --tta              test-time augmentation (a bit more accurate, ~3x slower)
    --link-factor 1.5  how close people must be to belong to the same crowd (bigger = looser)
    --min-group 2      minimum people to call something a crowd
    --show-isolated    also draw isolated people (thin grey) in crowd mode
"""
import argparse
import time
from pathlib import Path

import cv2

from src.detector import PersonCrowdPipeline

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
ROOT = Path(__file__).resolve().parent

#Below function main() is the entry point of the script. It parses command-line arguments, 
# initializes the PersonCrowdPipeline with the specified parameters, and processes the input source 
# (image, folder, video, or camera) accordingly.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    #above line creates an ArgumentParser object to handle command-line arguments, using the module's docstring as 
    #the description and formatting it for better readability.
    ap.add_argument("--source", required=True, help="image / folder / video path, or camera index (0)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--mode", choices=["crowd", "person"], default="crowd")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--link-factor", type=float, default=1.5)
    ap.add_argument("--min-group", type=int, default=3)
    ap.add_argument("--show-isolated", action="store_true")
    ap.add_argument("--show", action="store_true", help="open a preview window")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()

    pipe = PersonCrowdPipeline(a.weights, a.conf, a.iou, a.imgsz, a.device, a.tta, a.link_factor, a.min_group)
    #above line initializes an instance of the PersonCrowdPipeline class with the specified command-line arguments,
    # setting up the detection model and parameters for processing images or videos.
    print("weights:", pipe.weights)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    #above line creates the output directory specified by the user (or defaults to "outputs") if it doesn't already exist,
    # ensuring that the processed results can be saved without errors.
    src = a.source

    # ---------- camera ----------
    #below block checks if the source is a digit (indicating a camera index) and opens the corresponding webcam for live detection.   
    #If the source is not a digit, it checks if it's a valid file or directory and processes images or videos accordingly.
    
    if src.isdigit():
        run_stream(pipe, cv2.VideoCapture(int(src)), None, a, live=True)
        return
    p = Path(src)
    if not p.exists():
        raise SystemExit(f"source not found: {src}")

    # ---------- images ----------
    #below block checks if the source is a directory or an image file and processes each image using the PersonCrowdPipeline,
    # saving the results to the output directory and displaying them if requested.
    if p.is_dir() or p.suffix.lower() in IMG:
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG) if p.is_dir() else [p]
        #above line creates a list of image files to process. If the source is a directory,
        # it collects all image files with supported extensions;
        # if the source is a single image file, it creates a list containing just that file
        for f in files:
            im = cv2.imread(str(f))
            if im is None:
                print("cannot read", f); continue
            t = time.time()
            res, info = pipe.process(im, a.mode, a.show_isolated)
            dst = out / f"{f.stem}_out.jpg"
            cv2.imwrite(str(dst), res)
            print(f"{f.name}: {info}  ({time.time() - t:.2f}s) -> {dst}")
            if a.show:
                cv2.imshow("result", res); cv2.waitKey(0)
        return
    #the above block processes each image file by reading it, applying the detection pipeline,
    #saving the processed image to the output directory,
    #and printing the detection summary along with the processing time. 
    #If the --show flag is set, it displays the processed image in a window.    

    # ---------- video ----------
    if p.suffix.lower() in VID:
        run_stream(pipe, cv2.VideoCapture(str(p)), out / f"{p.stem}_out.mp4", a, live=False)
        return
    raise SystemExit("unsupported source type")
#Above block checks if the source is a video file with a supported extension and processes it using the run_stream function,
# saving the output video to the specified output directory. 
# If the source type is unsupported, it raises a SystemExit error with an appropriate message.
   

#below function run_stream() handles the processing of video streams, either from a file or a live camera feed. 
def run_stream(pipe, cap, dst, a, live):
    #here cap is a cv2.VideoCapture object that reads frames from the video source (file or camera),
    # dst is the output file path for saving the processed video (if applicable),
    # a contains the command-line arguments, and live indicates whether the stream is from a live camera feed.
    if not cap.isOpened():
        raise SystemExit("cannot open video / camera")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if dst else None
    n, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res, info = pipe.process(frame, a.mode, a.show_isolated)
            n += 1
            if writer:
                writer.write(res)
            if a.show or live:
                cv2.imshow("person / crowd detection  (q = quit)", res)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if n % 25 == 0:
                print(f"frame {n}: {info}  ({n / (time.time() - t0):.1f} FPS)")
    finally:
        cap.release()
        if writer:
            writer.release(); print("saved ->", dst)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

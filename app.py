"""Web app (runs locally in your browser):  python app.py   ->  open http://127.0.0.1:7860

Tabs: upload an image | upload a video | live laptop webcam.
"""
import tempfile

import cv2
import gradio as gr
#gradio is a Python library that allows you to quickly create user interfaces for
#machine learning models, enabling users to interact with the model through a web browser.

from src.detector import PersonCrowdPipeline

pipe = PersonCrowdPipeline()   # picks weights/best_person.pt if present, else yolo11m.pt
#PersonCrowdPipeline  an automated pipeline that detects people and groups them into crowds.
#It uses a YOLO model for person detection and applies a grouping algorithm based on proximity to identify crowds. 
#The pipeline can be configured with various parameters such as confidence threshold, link factor, minimum group size, and inference image size.


#_cfg function sets the configuration parameters for the PersonCrowdPipeline instance.
def _cfg(conf, link, min_group, imgsz):
    pipe.conf, pipe.link_factor, pipe.min_group_size, pipe.imgsz = conf, link, int(min_group), int(imgsz)

#summary function calcuates and returns a summary of the detection results, 
# including the number of persons detected, the number of crowds identified, 
# the sizes of those crowds, and the number of isolated people not counted as part of any crowd.
def _summary(info):
    return (f"Persons detected: {info['persons']}\nCrowds: {info['crowds']}  (sizes: {info['crowd_sizes']})\n"
            f"Isolated people (not counted as crowd): {info['isolated']}")

#on_image function processes an uploaded image, applies the detection pipeline, 
#and returns the processed image along with a summary of the detection results.
def on_image(img_rgb, mode, conf, link, min_group, imgsz):
    if img_rgb is None:
        return None, "Upload an image first."
    _cfg(conf, link, min_group, imgsz)
    res, info = pipe.process(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), "person" if mode == "Person only" else "crowd")
    #Above line converts the input image from RGB to BGR format (as OpenCV uses BGR) and processes it through the PersonCrowdPipeline.
    # The mode determines whether to detect only persons or to also identify crowds. 
    # The function returns the processed image and a summary of the detection results.
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), _summary(info)
 #above line converts the processed image back to RGB format for display in the web app and returns
 #it along with the summary information.

#function on_video processes an uploaded video file,
# applies the detection pipeline frame by frame, and saves the processed video to a temporary file.
def on_video(path, mode, conf, link, min_group, imgsz, progress=gr.Progress()):
    if not path:
        return None
    _cfg(conf, link, min_group, imgsz)
    cap = cv2.VideoCapture(path)
    #Above line initializes a video capture object to read frames from the uploaded video file specified by the path.
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    #Above line retrieves the frames per second (FPS) of the video. If the FPS cannot be determined, it defaults to 25 FPS.
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #Above line retrieves the width and height of the video frames, converting them to integers for further processing.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1 
    #Above line retrieves the total number of frames in the video.
    #If the frame count cannot be determined, it defaults to 1 to avoid division by zero errors during progress tracking.
    dst = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name 
    #Above line creates a temporary file with an .mp4 suffix to store the processed video output. 
    #The delete=False parameter ensures that the file is not automatically deleted when closed, allowing it to be accessed later.
    w = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    #Above line initializes a video writer object to write the processed frames to the temporary output file.
    i = 0
    while True:
        ok, fr = cap.read()
        #Above line reads the next frame from the video capture object.
        if not ok:
            break
        res, _ = pipe.process(fr, "person" if mode == "Person only" else "crowd")
        #Above line processes the current frame using the PersonCrowdPipeline, 
        #applying either person-only detection or crowd detection based on the selected mode.
        w.write(res); i += 1
        #Above line writes the processed frame to the output video file and increments the frame counter.
        progress(i / total)
        #Above line updates the progress indicator in the web app, showing the user how much of the video has been processed as a fraction of the total number of frames.
    cap.release(); w.release()
    return dst


def on_webcam(frame_rgb, mode, conf, link, min_group, imgsz):
    if frame_rgb is None:
        return None
    _cfg(conf, link, min_group, imgsz)
    res, _ = pipe.process(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), "person" if mode == "Person only" else "crowd")
    #Above line converts the webcam frame from RGB to BGR format and processes it through the PersonCrowdPipeline,
    #applying the selected detection mode. The processed frame is then returned in RGB format for display in the web app.
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="Person & Crowd Detection (YOLO)") as demo:
    gr.Markdown("## Person & Crowd Detection (YOLO)\n**Crowd mode**: highlights groups of people, ignores isolated people.  "
                "**Person only**: boxes every person (no other objects).")
    with gr.Row():
        mode = gr.Radio(["Crowd detection", "Person only"], value="Crowd detection", label="Mode")
        conf = gr.Slider(0.1, 0.9, 0.35, step=0.05, label="Confidence")
        link = gr.Slider(0.5, 3.0, 1.5, step=0.1, label="Crowd link distance (bigger = looser groups)")
        min_group = gr.Slider(2, 6, 3, step=1, label="Min people per crowd")
        imgsz = gr.Dropdown([640, 960, 1280], value=960, label="Inference size")
    opts = [mode, conf, link, min_group, imgsz]
#Above function gr.Row() creates a horizontal layout for the user interface elements,
#allowing users to select the detection mode, confidence threshold, crowd link distance, minimum group size, 
#and inference image size. The selected options are stored in the opts list for use in the detection functions.
#Radio and Slider components allow users to interactively adjust these parameters,
#while the Dropdown component provides a selection of predefined inference sizes.




    with gr.Tab("Image"):
        i_in = gr.Image(type="numpy", label="Upload image")
        i_btn = gr.Button("Detect", variant="primary")
        i_out = gr.Image(label="Result"); i_txt = gr.Textbox(label="Summary", lines=3)
        i_btn.click(on_image, [i_in] + opts, [i_out, i_txt])
        #Above function gr.Tab() creates a tabbed interface for the web app, allowing users to switch between different input types 
        # (image, video, webcam).
        #The Image tab includes an image upload component, a button to trigger detection, 
        # and output components for displaying the processed image and a summary of the detection results.
        #The click event of the button is linked to the on_image function, 
        # which processes the uploaded image and updates the output components with the results.
    with gr.Tab("Video"):
        v_in = gr.Video(label="Upload video")
        v_btn = gr.Button("Process video", variant="primary")
        v_out = gr.Video(label="Result")
        v_btn.click(on_video, [v_in] + opts, v_out)
    with gr.Tab("Webcam (live)"):
        w_in = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
        w_out = gr.Image(label="Live result")
        w_in.stream(on_webcam, [w_in] + opts, w_out, stream_every=0.3, time_limit=300)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

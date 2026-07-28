import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import glob
import supervision as sv
from ultralytics import YOLO
import tempfile

# Page configuration
st.set_page_config(
    page_title="Mosquito detection hub",
    page_icon=":material/bug_report:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header Section
st.title("Mosquito detection hub", text_alignment="center")
st.write(
    "An interactive computer vision platform powered by YOLO to identify and localize mosquitoes in images, videos, and live feeds.",
    text_alignment="center"
)
st.space("large")

# Find available model paths
model_paths = glob.glob("runs/detect/*/weights/best.pt")
model_options = {os.path.basename(os.path.dirname(os.path.dirname(p))): p for p in model_paths}

# Add default or fallback model options if glob returns empty
if not model_options:
    model_options["Default (best.pt)"] = "best.pt"

# Sidebar Configuration
st.sidebar.title("Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose YOLO model weight",
    options=list(model_options.keys()),
    index=0
)
model_path = model_options[selected_model_name]

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.30,
    step=0.05
)

# Load the model helper
@st.cache_resource
def load_yolo_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}", icon=":material/error:")
        return None

model = load_yolo_model(model_path)

# Initialize annotation components
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator()

def run_prediction(image: Image.Image, conf: float):
    # Convert PIL Image to OpenCV BGR format
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run YOLO prediction
    results = model(img_bgr, conf=conf)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Check if there are any detections
    mosquito_count = len(detections)
    
    labels = [
        f"{model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id in zip(detections.confidence, detections.class_id)
    ]
    
    # Annotate frame
    annotated_bgr = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)
    annotated_bgr = label_annotator.annotate(scene=annotated_bgr, detections=detections, labels=labels)
    
    # Convert back to PIL Image for streamlit display
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), mosquito_count

# Layout tabs
tab_image, tab_video, tab_webcam, tab_samples = st.tabs([
    "Image detection",
    "Video detection",
    "Live webcam",
    "Sample gallery"
])

# --- Image Tab ---
with tab_image:
    st.subheader("Upload custom image")
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.subheader("Original image")
                st.image(image, width="stretch")
            
        with col2:
            with st.container(border=True):
                st.subheader("Processed image")
                if model is not None:
                    with st.spinner("Analyzing image..."):
                        result_img, count = run_prediction(image, confidence_threshold)
                    st.image(result_img, width="stretch")
                    
                    st.space("medium")
                    c1, c2 = st.columns(2)
                    c1.metric("Mosquitoes detected", count)
                    c2.metric("Selected model", selected_model_name)
                else:
                    st.warning("Model could not be loaded.", icon=":material/warning:")

# --- Video Tab ---
with tab_video:
    st.subheader("Upload custom video")
    uploaded_video = st.file_uploader("Upload a video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.subheader("Original video")
                st.video(uploaded_video)
            
        with col2:
            with st.container(border=True):
                st.subheader("Video analysis")
                if st.button("Start video analysis", icon=":material/play_arrow:"):
                    if model is not None:
                        # Target path for processed video
                        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                        
                        video_info = sv.VideoInfo.from_video_path(video_path)
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                        
                        # Function to process frame
                        def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
                            # Update progress
                            progress = min(1.0, float(index) / max(1.0, float(video_info.total_frames)))
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {index}/{video_info.total_frames}...")
                            
                            results = model(frame, imgsz=256, conf=confidence_threshold)[0]
                            detections = sv.Detections.from_ultralytics(results)
                            
                            labels = [
                                f"{model.names[class_id]} {confidence:0.2f}"
                                for confidence, class_id in zip(detections.confidence, detections.class_id)
                            ]
                            
                            frame = box_annotator.annotate(scene=frame, detections=detections)
                            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                            return frame
                        
                        with st.spinner("Processing video..."):
                            sv.process_video(source_path=video_path, target_path=output_path, callback=process_frame)
                        
                        progress_bar.progress(1.0)
                        status_text.success("Video processing complete!", icon=":material/check_circle:")
                        
                        # Read video content to offer download
                        with open(output_path, "rb") as f:
                            video_bytes = f.read()
                        
                        st.download_button(
                            label="Download annotated video",
                            data=video_bytes,
                            file_name="detected_mosquitoes.mp4",
                            mime="video/mp4",
                            icon=":material/download:"
                        )
                    else:
                        st.warning("Model could not be loaded.", icon=":material/warning:")

# --- Webcam Tab ---
with tab_webcam:
    st.subheader("Live webcam detection")
    picture = st.camera_input("Capture frame from webcam")
    
    if picture is not None:
        image = Image.open(picture)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("Webcam frame")
                st.image(image, width="stretch")
            
        with col2:
            with st.container(border=True):
                st.subheader("Detection results")
                if model is not None:
                    with st.spinner("Analyzing frame..."):
                        result_img, count = run_prediction(image, confidence_threshold)
                    st.image(result_img, width="stretch")
                    
                    st.space("medium")
                    c1, c2 = st.columns(2)
                    c1.metric("Mosquitoes detected", count)
                    c2.metric("Selected model", selected_model_name)
                else:
                    st.warning("Model could not be loaded.", icon=":material/warning:")

# --- Sample Gallery Tab ---
with tab_samples:
    st.subheader("Select from sample images")
    
    # Get files from test/images
    sample_images = glob.glob("test/images/*.jpg") + glob.glob("test/images/*.png") + glob.glob("test/images/*.jpeg")
    
    if sample_images:
        selected_sample = st.selectbox(
            "Choose a sample image to test",
            options=sample_images,
            format_func=lambda x: os.path.basename(x)
        )
        
        if selected_sample:
            image = Image.open(selected_sample)
            
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.subheader("Original sample")
                    st.image(image, width="stretch")
                
            with col2:
                with st.container(border=True):
                    st.subheader("Detection results")
                    if model is not None:
                        with st.spinner("Analyzing sample..."):
                            result_img, count = run_prediction(image, confidence_threshold)
                        st.image(result_img, width="stretch")
                        
                        st.space("medium")
                        c1, c2 = st.columns(2)
                        c1.metric("Mosquitoes detected", count)
                        c2.metric("Selected model", selected_model_name)
                    else:
                        st.warning("Model could not be loaded.", icon=":material/warning:")
    else:
        st.info("No sample images found in test/images directory.", icon=":material/info:")

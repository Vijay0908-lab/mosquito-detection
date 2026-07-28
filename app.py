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
    page_title="Mosquito Detection Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for rich aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0ea5e9 !important;
        color: white !important;
        border-color: #38bdf8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #38bdf8;
    }
    .card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #38bdf8; font-size: 2.8rem; font-weight: 800; margin-bottom: 0.5rem;">🦟 Mosquito Detection Hub</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">An interactive computer vision platform powered by YOLO to identify and localize mosquitoes in images, videos, and live feeds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Find available model paths
model_paths = glob.glob("runs/detect/*/weights/best.pt")
model_options = {os.path.basename(os.path.dirname(os.path.dirname(p))): p for p in model_paths}

# Add default or fallback model options if glob returns empty
if not model_options:
    model_options["Default (best.pt)"] = "best.pt"

# Sidebar Configuration
st.sidebar.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #38bdf8; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem;">Settings</h2>
        <hr style="border-color: #334155; margin-top: 0.5rem; margin-bottom: 1rem;">
    </div>
    """,
    unsafe_allow_html=True,
)

selected_model_name = st.sidebar.selectbox(
    "Choose YOLO Model Weight:",
    options=list(model_options.keys()),
    index=0
)
model_path = model_options[selected_model_name]

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold:",
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
        st.error(f"Error loading model from {path}: {e}")
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
    "🖼️ Image Detection",
    "📹 Video Detection",
    "📷 Live Webcam",
    "✨ Sample Gallery"
])

# --- Image Tab ---
with tab_image:
    st.subheader("Upload Custom Image")
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card"><h4>Original Image</h4></div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown('<div class="card"><h4>Processed Image</h4></div>', unsafe_allow_html=True)
            if model is not None:
                with st.spinner("Analyzing image..."):
                    result_img, count = run_prediction(image, confidence_threshold)
                st.image(result_img, use_container_width=True)
                
                # Show Stats
                st.markdown("---")
                c1, c2 = st.columns(2)
                c1.metric("Mosquitoes Detected", count)
                c2.metric("Selected Model", selected_model_name)
            else:
                st.warning("Model could not be loaded.")

# --- Video Tab ---
with tab_video:
    st.subheader("Upload Custom Video")
    uploaded_video = st.file_uploader("Upload a video (MP4, MOV, AVI)...", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.markdown('<div class="card"><h4>Video Process Settings</h4></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.video(uploaded_video)
            
        with col2:
            if st.button("Start Video Analysis 🚀"):
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
                        status_text.text(f"Processing Frame {index}/{video_info.total_frames}...")
                        
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
                    status_text.success("Video processing complete!")
                    
                    # Read video content to offer download
                    with open(output_path, "rb") as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        label="Download Annotated Video 📥",
                        data=video_bytes,
                        file_name="detected_mosquitoes.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.warning("Model could not be loaded.")

# --- Webcam Tab ---
with tab_webcam:
    st.subheader("Live Webcam Detection")
    picture = st.camera_input("Take a photo to detect mosquitoes")
    
    if picture is not None:
        image = Image.open(picture)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card"><h4>Webcam Frame</h4></div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown('<div class="card"><h4>Detection Results</h4></div>', unsafe_allow_html=True)
            if model is not None:
                with st.spinner("Analyzing frame..."):
                    result_img, count = run_prediction(image, confidence_threshold)
                st.image(result_img, use_container_width=True)
                
                # Show Stats
                st.markdown("---")
                c1, c2 = st.columns(2)
                c1.metric("Mosquitoes Detected", count)
                c2.metric("Selected Model", selected_model_name)
            else:
                st.warning("Model could not be loaded.")

# --- Sample Gallery Tab ---
with tab_samples:
    st.subheader("Select from Sample Images")
    
    # Get files from test/images
    sample_images = glob.glob("test/images/*.jpg") + glob.glob("test/images/*.png") + glob.glob("test/images/*.jpeg")
    
    if sample_images:
        selected_sample = st.selectbox(
            "Choose a sample image to test:",
            options=sample_images,
            format_func=lambda x: os.path.basename(x)
        )
        
        if selected_sample:
            image = Image.open(selected_sample)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card"><h4>Original Sample</h4></div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                
            with col2:
                st.markdown('<div class="card"><h4>Detection Results</h4></div>', unsafe_allow_html=True)
                if model is not None:
                    with st.spinner("Analyzing sample..."):
                        result_img, count = run_prediction(image, confidence_threshold)
                    st.image(result_img, use_container_width=True)
                    
                    # Show Stats
                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    c1.metric("Mosquitoes Detected", count)
                    c2.metric("Selected Model", selected_model_name)
                else:
                    st.warning("Model could not be loaded.")
    else:
        st.info("No sample images found in test/images directory.")

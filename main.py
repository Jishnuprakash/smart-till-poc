import streamlit as st
import cv2
import tempfile
from PIL import Image
import numpy as np
from yolo_tracker import YoloTracker

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Till POC - YOLOv8",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Tracker in Session State ---
if 'tracker' not in st.session_state:
    st.session_state.tracker = YoloTracker()

# --- Callback function to reset summary stats on mode change ---
def on_mode_change():
    """Resets the tracker's statistics when the user switches modes."""
    if 'tracker' in st.session_state:
        st.session_state.tracker.reset_summary_stats()

# --- Title and Description ---
st.title("🛒 Smart Till - Item Detector")
st.write("""
This application uses **YOLOv8** to detect and track common objects.
Upload an image or a video to begin.
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    st.info("Adjust the confidence to filter out uncertain detections.")

# --- Main Application Logic ---
# The radio button now has only two options and a callback function
mode = st.radio(
    "Choose Input Mode",
    ["Upload Image", "Upload Video"],
    horizontal=True,
    on_change=on_mode_change # This function is called when the selection changes
)

# --- Image Detection Mode ---
if mode == "Upload Image":
    st.subheader("🖼️ Detect Items in an Image")
    image_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"], key="image_uploader")

    if image_file:
        image = Image.open(image_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process the single frame
        annotated_frame, summary = st.session_state.tracker._process_frame(frame, confidence_threshold)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(annotated_frame, caption="Detected Items", channels="BGR")
            
        st.subheader("📦 Detected Items Summary")
        if not summary:
            st.info("No items detected.")
        else:
            cols = st.columns(len(summary) or 1)
            for idx, (item, count) in enumerate(summary.items()):
                with cols[idx]:
                    st.metric(label=item.capitalize(), value=count)

# --- Video Detection Mode ---
elif mode == "Upload Video":
    st.subheader("📹 Detect Items in a Video")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        video_placeholder = st.empty()
        summary_placeholder = st.empty()
        
        cap = cv2.VideoCapture(tfile.name)
        
        def frame_generator():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()
            
        # The track_video_stream method already resets stats, ensuring a fresh start
        for annotated_frame, summary in st.session_state.tracker.track_video_stream(frame_generator(), confidence_threshold):
            video_placeholder.image(annotated_frame, channels="BGR")
            with summary_placeholder.container():
                st.subheader("Live Item Count")
                if not summary:
                    st.info("No items detected yet.")
                else:
                    cols = st.columns(len(summary) or 1)
                    for idx, (item, count) in enumerate(summary.items()):
                        with cols[idx]:
                            st.metric(label=item.capitalize(), value=count)
        
        # Display Final Summary Table After Video Ends
        st.subheader("📊 Final Detection Summary")
        summary_df = st.session_state.tracker.get_summary_dataframe()
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No objects were detected in the video.")
            
    else:
        st.info("Please upload a video file to begin.")
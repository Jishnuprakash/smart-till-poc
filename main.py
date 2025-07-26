import streamlit as st
import cv2
from detectors.yolov8_tracker import YOLOv8WithTracker
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Smart Till POC", layout="wide")
st.title("🛒 Smart Till POC – Item Detection")

detector = YOLOv8WithTracker()
mode = st.radio("Choose input mode", ["Upload Video", "Live Webcam"])

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        item_summary = {}
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % 2 != 0:
                continue  # Skip every other frame

            annotated, counts = detector.detect_and_track(frame)

            for item, count in counts.items():
                item_summary[item] = item_summary.get(item, 0) + count

            stframe.image(annotated, channels="BGR", use_column_width=True)

        cap.release()
        st.subheader("📦 Item Count Summary")
        st.json(item_summary)

elif mode == "Live Webcam":
    class ItemDetectionTransformer(VideoTransformerBase):
        def __init__(self):
            self.item_summary = {}
            self.frame_id = 0

        def transform(self, frame):
            self.frame_id += 1
            if self.frame_id % 2 != 0:
                return frame.to_ndarray(format="bgr24")  # just pass through skipped frame

            image = frame.to_ndarray(format="bgr24")
            annotated, counts = detector.detect_and_track(image)

            for item, count in counts.items():
                self.item_summary[item] = self.item_summary.get(item, 0) + count

            return annotated

    webrtc_streamer(
        key="smart-till",
        video_transformer_factory=ItemDetectionTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

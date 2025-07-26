from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class YOLOv8WithTracker:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=30)

    def detect_and_track(self, frame):
        # Resize for speed
        resized = cv2.resize(frame, (640, 640))
        results = self.model(resized, verbose=False)

        detections = []
        for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
            score = conf.cpu().numpy().item()
            if score < 0.5:  # skip low-confidence detections
                continue
            bbox = box.cpu().numpy().tolist()
            class_id = int(cls.cpu().numpy().item())
            detections.append(([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], score, class_id))

        tracks = self.tracker.update_tracks(detections, frame=resized)

        tracked_items = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            class_id = track.det_class
            label = self.model.model.names[class_id]
            tracked_items[label] = tracked_items.get(label, set())
            tracked_items[label].add(track_id)

        counts = {label: len(ids) for label, ids in tracked_items.items()}
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))  # scale back
        return annotated_frame, counts

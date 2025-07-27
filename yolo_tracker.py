from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
from collections import defaultdict

class YoloTracker:
    def __init__(self, model_path="models/yolov8n.pt"):
        """
        Initializes the tracker with the YOLO model and summary statistics.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        # Call the reset method during initialization
        self.reset_summary_stats()

    def reset_summary_stats(self):
        """Resets the statistics for a new video or session."""
        self.frame_count = 0
        self.class_appearances = defaultdict(int)
        self.overall_tracked_ids = defaultdict(set)

    def _process_frame(self, frame: np.ndarray, confidence_threshold: float):
        """
        Core logic to process a single frame for detection, tracking, and stats gathering.
        """
        self.frame_count += 1
        results = self.model(frame, conf=confidence_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Annotation Logic
        labels = []
        for det in tracked_detections:
            class_id = det[3]
            tracker_id = det[4]
            class_name = self.model.model.names[class_id]
            label_text = f"#{tracker_id} {class_name}"
            labels.append(label_text)
            
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=tracked_detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=tracked_detections, labels=labels
        )
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, detections=tracked_detections
        )

        # Statistics Gathering
        current_frame_classes = set()
        for det in tracked_detections:
            class_id = det[3]
            tracker_id = det[4]
            class_name = self.model.model.names[class_id]
            current_frame_classes.add(class_name)
            self.overall_tracked_ids[class_name].add(tracker_id)
        
        for class_name in current_frame_classes:
            self.class_appearances[class_name] += 1
            
        # Live summary for the current frame
        live_summary = {name: len(ids) for name, ids in self.overall_tracked_ids.items() if len(ids) > 0}

        return annotated_frame, live_summary

    # --- THIS IS THE MISSING METHOD ---
    def get_summary_dataframe(self):
        """
        Generates a pandas DataFrame summarizing the entire detection session.
        """
        if self.frame_count == 0:
            return pd.DataFrame()

        summary_data = []
        sorted_classes = sorted(self.overall_tracked_ids.keys())

        for class_name in sorted_classes:
            total_unique_items = len(self.overall_tracked_ids[class_name])
            appearance_frames = self.class_appearances[class_name]
            presence_percentage = (appearance_frames / self.frame_count) * 100
            
            summary_data.append({
                "Object Class": class_name.capitalize(),
                "Total Unique Items": total_unique_items,
                "Frame Presence (%)": f"{presence_percentage:.2f}"
            })
        
        if not summary_data:
            return pd.DataFrame()
            
        return pd.DataFrame(summary_data)

    def track_video_stream(self, frame_generator, confidence_threshold):
        """
        A generator that processes frames from a video stream and yields results.
        """
        self.reset_summary_stats()
        for frame in frame_generator:
            yield self._process_frame(frame, confidence_threshold)
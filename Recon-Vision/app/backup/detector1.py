from ultralytics import YOLO
import cv2
from datetime import datetime

class Detector:
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf_threshold=0.25, alert_conf_threshold=0.5):
        self.model = YOLO(model_path)  # Initialize without device argument
        self.model.to(device)  # Move model to desired device (cpu or cuda)
        self.conf_threshold = conf_threshold
        self.alert_conf_threshold = alert_conf_threshold

    def detect(self, frame):
        """
        Run detection on a frame.

        Args:
            frame (np.array): BGR image

        Returns:
            detections: list of dict with keys ['box', 'class', 'confidence']
            annotated_frame: frame with bounding boxes and labels drawn
            alerts: list of alert dicts for person detections above alert_conf_threshold
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model(img_rgb)
        result = results[0]

        detections = []
        alerts = []

        annotated_frame = frame.copy()

        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            confidence = float(conf)
            label = self.model.names[cls_id]

            detection = {'box': (x1, y1, x2, y2), 'class': label, 'confidence': confidence}
            detections.append(detection)

            # Draw bounding box and label on annotated frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Generate alerts for person detections above alert_conf_threshold
            if label == "person" and confidence >= self.alert_conf_threshold:
                alert = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "class": label,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                }
                alerts.append(alert)

        return detections, annotated_frame, alerts

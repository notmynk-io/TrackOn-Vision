from ultralytics import YOLO
import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

class Detector:
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf_threshold=0.3, alert_conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.alert_conf_threshold = alert_conf_threshold
        self.class_names = self.model.names
        self.tracker = CentroidTracker(maxDisappeared=20)

        self.allowed_classes = {
            "person", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe"
        }

    def detect(self, frame):
        results = self.model(frame, device=self.device, conf=self.conf_threshold, classes=[0, 15, 16, 17, 18, 19, 20, 21, 22, 23])[0] # Filter for person and animals
        
        all_detections = []
        person_rects = []
        alerts = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = r
            cls_id = int(cls_id)
            cls_name = self.class_names.get(cls_id)

            if cls_name not in self.allowed_classes:
                continue

            detection = {
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "class": cls_name,
                "intruder_id": None # To be filled by tracker
            }
            all_detections.append(detection)

            if cls_name == "person" and conf >= self.alert_conf_threshold:
                person_rects.append(detection["box"])

        # Update tracker with person detections
        tracked_objects = self.tracker.update(person_rects)

        # Link tracked objects back to detections and create alerts
        for det in all_detections:
            if det["class"] == "person" and det["confidence"] >= self.alert_conf_threshold:
                # Find the corresponding tracked ID
                (x1, y1, x2, y2) = det["box"]
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                
                # This is a simplified matching. A more robust way would be to match rects directly.
                for objectID, centroid in tracked_objects.items():
                    if abs(cX - centroid[0]) < 20 and abs(cY - centroid[1]) < 20:
                        det["intruder_id"] = objectID
                        alerts.append(det)
                        break

        annotated_frame = self.draw_modern_overlay(frame, all_detections)
        return all_detections, annotated_frame, alerts

    def draw_modern_overlay(self, image, detections):
        overlay = image.copy()
        alpha = 0.6

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class"]
            conf = det["confidence"]
            intruder_id = det.get("intruder_id")

            color = (0, 220, 0)  # Bright green for animals
            if cls_name == "person":
                color = (0, 100, 255)  # Orange-red for person

            if intruder_id is not None:
                color = (0, 0, 255) # Bright red for tracked intruder

            thickness = 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            label = f"{cls_name} {conf:.2f}"
            if intruder_id is not None:
                label = f"ID: {intruder_id} | {conf:.2f}"

            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image
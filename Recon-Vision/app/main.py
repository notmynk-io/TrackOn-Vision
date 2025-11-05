from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import cv2
import asyncio
from .detector import Detector
import json
from typing import List

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Use webcam index 0 (local)
WEBCAM_INDEX = 0
cap = cv2.VideoCapture(WEBCAM_INDEX)

# Load your detector
detector = Detector(
    model_path='yolov8n.pt',
    device='cpu',
    conf_threshold=0.35,
    alert_conf_threshold=0.5,
)

import time
import os

# Cooldown for alerts (in seconds)
ALERT_COOLDOWN = 10  # 10 seconds
last_alert_time = 0

# Ensure the intruder_images directory exists
os.makedirs("app/intruder_images", exist_ok=True)

async def frame_generator():
    global last_alert_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Keep a copy of the original frame for saving
        original_frame = frame.copy()

        detections, annotated_frame, alerts = detector.detect(frame)

        current_time = time.time()
        if alerts and (current_time - last_alert_time) > ALERT_COOLDOWN:
            last_alert_time = current_time

            # Save the original frame as an intruder image
            timestamp = int(current_time)
            image_name = f"intruder_{timestamp}.jpg"
            image_path = os.path.join("app/intruder_images", image_name)
            cv2.imwrite(image_path, original_frame)

            alert_message = json.dumps({
                "type": "alert",
                "count": len(alerts),
                "details": alerts,
                "image_url": f"/intruder_images/{image_name}"  # URL to the image
            })
            await manager.broadcast(alert_message)
        
        # Also broadcast general stats
        stats = {
            "person": sum(1 for d in detections if d['class'] == 'person'),
            "bird": sum(1 for d in detections if d['class'] == 'bird'),
            "animal": sum(1 for d in detections if d['class'] not in ['person', 'bird'])
        }
        stats_message = json.dumps({"type": "stats", "data": stats})
        await manager.broadcast(stats_message)


        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Yield control to the event loop
        await asyncio.sleep(0.01)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

from starlette.responses import FileResponse

@app.get("/")
async def read_index():
    return FileResponse('app/index.html')

@app.get("/intruder_images/{image_name}")
async def get_intruder_image(image_name: str):
    return FileResponse(f'app/intruder_images/{image_name}')

@app.get("/warning.mp3")
async def read_warning_sound():
    return FileResponse('app/warning.mp3')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .detector1 import Detector

import numpy as np
import cv2
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./detections.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    class_name = Column(String)
    confidence = Column(Float)
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Border Surveillance Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = Detector(device='cpu', conf_threshold=0.3, alert_conf_threshold=0.5)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=422, detail="Invalid image file")

        detections, annotated, alerts = detector.detect(img)

        # Save detections and alerts into DB
        db = SessionLocal()

        for d in detections:
            det = Detection(
                timestamp=datetime.utcnow(),
                class_name=d['class'],
                confidence=d['confidence'],
                x1=d['box'][0],
                y1=d['box'][1],
                x2=d['box'][2],
                y2=d['box'][3]
            )
            db.add(det)

        db.commit()
        db.close()

        response = {
            "detections": detections,
            "alerts": alerts,
            "num_detections": len(detections),
            "num_alerts": len(alerts),
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
async def get_alerts(limit: int = 10):
    db = SessionLocal()
    alerts_query = (
        db.query(Detection)
        .filter(Detection.class_name == "person", Detection.confidence >= 0.5)
        .order_by(Detection.timestamp.desc())
        .limit(limit)
        .all()
    )
    alerts = [
        {
            "timestamp": d.timestamp.isoformat(),
            "class": d.class_name,
            "confidence": d.confidence,
            "box": [d.x1, d.y1, d.x2, d.y2]
        } for d in alerts_query
    ]
    db.close()
    return {"alerts": alerts}

@app.get("/hotspots")
async def get_hotspots():
    db = SessionLocal()
    # Aggregate count by hour for person alerts
    from sqlalchemy import func
    rows = (
        db.query(
            func.strftime('%Y-%m-%d %H:00', Detection.timestamp).label("hour"),
            func.count(Detection.id)
        )
        .filter(Detection.class_name == "person", Detection.confidence >= 0.5)
        .group_by("hour")
        .order_by("hour")
        .all()
    )
    hotspots = [{"hour": r[0], "count": r[1]} for r in rows]
    db.close()
    return {"hotspots": hotspots}

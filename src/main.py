import time
from typing import Callable

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# --- 1. Initialize Model and FastAPI App ---

print('Loading YOLOv8 model...')
model = YOLO('yolov8n.pt')
print("Model loaded successfully.")

app = FastAPI(
    title="FrameSight Object Detection API",
    description="API for real-time object detection using YOLOv8.",
    version="1.0.0",
)

# --- 2. Add Middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable) -> Response:
    """
    Adds a custome header X-Process-Time to the response with the request processing time.
    
    Args:
        request (Request): The incoming request object.
        call_next (Callable): The next function in the middleware chain.
        
    Returns:
        Response: The response object with the added "X-Process-Time" header.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

origins = ["*"] # For development, allow all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. API Endpoints ---

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)) -> dict:
    """
    Receives an image file, runs YOLOv8 inference, and returns the
    detected object data as JSON.

    Args:
        file (UploadFile): The image file to process.

    Returns:
        (dict): A JSON response containing a list of detected objects, each with its
        bounding box, confidence score, and class name.
    """

    # Read the image file from request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    # Run inference on the image
    results = model(image)
    result = results[0]

    # Extrac the detection data
    detections = []
    for box in result.boxes:
        detections.append({
            "box": box.xyxy[0].tolist(),
            "confidence": float(box.conf[0]),
            "class_id": int(box.cls[0]),
            "class_name": result.names[int(box.cls[0])],
        })
    
    return {'detections': detections}

@app.get("/")
def read_root() -> dict:
    """A simple endpoint to confirm the server is running."""

    return {"message": "FrameSight API is running" }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

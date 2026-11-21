import os
import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI(
    title="Vehicle & License Plate Recognition API",
    description="Extracts vehicle presence, vehicle type, and license plate text from uploaded images.",
    version="1.0.0"
)

# Enable CORS for universal global access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight YOLOv8 nano for CPU efficiency
yolo_model = YOLO("yolov8n.pt") 

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Class IDs for vehicles from the COCO dataset
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def extract_plate_text(gray_img):
    """Detect plate contour and extract alphanumeric characters."""
    bfilter = cv2.bilateralFilter(gray_img, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    cropped_plate = None
    if location is not None:
        mask = np.zeros(gray_img.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        if len(x) > 0 and len(y) > 0:
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_plate = gray_img[x1:x2+1, y1:y2+1]
    
    target_img = cropped_plate if cropped_plate is not None else gray_img
    ocr_results = reader.readtext(target_img)
    
    extracted_text = ""
    max_confidence = 0.0

    for (bbox, text, prob) in ocr_results:
        cleaned_text = "".join(e for e in text if e.isalnum()).upper()
        if len(cleaned_text) >= 3 and prob > max_confidence:
            extracted_text = cleaned_text
            max_confidence = float(prob)

    return extracted_text, round(max_confidence, 2)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "ANPR & Vehicle Detection API",
        "endpoint": "POST /api/v1/extract-plate"
    }


@app.post("/api/v1/extract-plate")
async def extract_plate(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    # 1. Vehicle Detection
    results = yolo_model(image)[0]
    is_vehicle = False
    detected_vehicle_type = None

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in VEHICLE_CLASS_IDS:
            is_vehicle = True
            detected_vehicle_type = VEHICLE_CLASS_IDS[cls_id]
            break

    # 2. License Plate Text Extraction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate_text, confidence = extract_plate_text(gray)

    return {
        "success": True,
        "is_vehicle": is_vehicle,
        "vehicle_type": detected_vehicle_type,
        "license_plate": {
            "detected": bool(plate_text),
            "text": plate_text if plate_text else None,
            "confidence": confidence if plate_text else 0.0
        }
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
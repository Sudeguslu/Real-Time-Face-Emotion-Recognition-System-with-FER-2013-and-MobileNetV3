import base64
import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from fastapi import HTTPException
from pydantic import BaseModel
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from fastapi import APIRouter
from database import EmotionRepository
from api.schemas.emotion import EmotionCreate, EmotionResponse

class FrameRequest(BaseModel):
    frame: str

router = APIRouter(prefix="/emotions", tags=["Captured Emotions"])
repo = EmotionRepository()


@router.get("/by-session/{session_id}", response_model=list[EmotionResponse])
def get_by_session(session_id: str):
    return repo.get_by_session(session_id)


@router.get("/count/{session_id}", response_model=dict)
def count_by_emotion(session_id: str):
    return repo.count_by_emotion(session_id)

@router.delete("/by-session/{session_id}", response_model=dict)
def delete_by_session(session_id: str):
    count = repo.delete_by_session(session_id)
    return {"deleted": count}

_model = None
_face_detector = None
_transform = None
EMOTIONS = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']

def get_model():
    global _model, _transform
    if _model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = models.mobilenet_v3_large(weights=None)
        _model.classifier[3] = nn.Linear(_model.classifier[3].in_features, 7)
        _model.load_state_dict(
            torch.load("best_emotion_mobilenetv3.pth", map_location=device, weights_only=True)
        )
        _model.to(device).eval()
        _transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return _model, _transform

def get_face_detector():
    global _face_detector
    if _face_detector is None:
        base_options = mp_python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.6,
        )
        _face_detector = mp_vision.FaceDetector.create_from_options(options)
    return _face_detector

@router.post("/analyze-frame", response_model=dict)
def analyze_frame(body: FrameRequest):
    try:
        img_data = base64.b64decode(body.frame)
        pil_img  = Image.open(io.BytesIO(img_data)).convert("RGB")
        np_img   = np.array(pil_img)
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz frame")

    face_detector = get_face_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
    result   = face_detector.detect(mp_image)

    if not result.detections:
        return {"emotion": None, "score": None, "faces": 0}

    model, transform = get_model()
    device = next(model.parameters()).device

    h, w = np_img.shape[:2]
    faces = []
    for detection in result.detections:
        bb = detection.bounding_box
        x1 = max(0, bb.origin_x)
        y1 = max(0, bb.origin_y)
        x2 = min(w, bb.origin_x + bb.width)
        y2 = min(h, bb.origin_y + bb.height)

        face_roi = pil_img.crop((x1, y1, x2, y2))
        tensor   = transform(face_roi).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].tolist()

        best_idx = int(np.argmax(probs))
        faces.append({
            "emotion": EMOTIONS[best_idx],
            "score": round(probs[best_idx], 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return {"faces": faces}

@router.post("/", response_model=dict, status_code=201)
def save_emotion(body: EmotionCreate):
    emotion_id = repo.save(
        emotion=body.emotion,
        session_id=body.sessionId,
        date=body.date,
    )
    return {"id": emotion_id}
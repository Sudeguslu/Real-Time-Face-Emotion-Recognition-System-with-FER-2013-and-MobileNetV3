from fastapi import APIRouter
from database import EmotionRepository
from api.schemas.emotion import EmotionCreate, EmotionResponse

router = APIRouter(prefix="/emotions", tags=["Captured Emotions"])
repo = EmotionRepository()


@router.get("/by-session/{session_id}", response_model=list[EmotionResponse])
def get_by_session(session_id: str):
    return repo.get_by_session(session_id)


@router.get("/count/{session_id}", response_model=dict)
def count_by_emotion(session_id: str):
    return repo.count_by_emotion(session_id)


@router.post("/", response_model=dict, status_code=201)
def save_emotion(body: EmotionCreate):
    emotion_id = repo.save(
        emotion=body.emotion,
        session_id=body.sessionId,
        date=body.date,
    )
    return {"id": emotion_id}


@router.delete("/by-session/{session_id}", response_model=dict)
def delete_by_session(session_id: str):
    count = repo.delete_by_session(session_id)
    return {"deleted": count}

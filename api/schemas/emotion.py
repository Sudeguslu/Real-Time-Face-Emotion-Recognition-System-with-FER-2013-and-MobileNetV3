from datetime import datetime
from pydantic import BaseModel


class EmotionCreate(BaseModel):
    emotion: str
    sessionId: str
    date: datetime | None = None


class EmotionResponse(BaseModel):
    id: str
    emotion: str
    sessionId: str
    date: datetime

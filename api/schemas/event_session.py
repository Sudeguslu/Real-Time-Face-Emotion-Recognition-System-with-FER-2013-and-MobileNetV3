from datetime import datetime
from pydantic import BaseModel


class EventSessionCreate(BaseModel):
    sessionName: str
    eventId: str
    duration: float = 0.0
    date: datetime | None = None


class EventSessionUpdate(BaseModel):
    duration: float


class EventSessionResponse(BaseModel):
    id: str
    sessionName: str
    eventId: str
    duration: float
    date: datetime

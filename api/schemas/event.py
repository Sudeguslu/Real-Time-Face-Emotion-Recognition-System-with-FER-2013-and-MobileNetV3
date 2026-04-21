from pydantic import BaseModel


class EventCreate(BaseModel):
    eventName: str


class EventResponse(BaseModel):
    id: str
    eventName: str

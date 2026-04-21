from fastapi import APIRouter, HTTPException
from database import EventRepository
from api.schemas.event import EventCreate, EventResponse

router = APIRouter(prefix="/events", tags=["Events"])
repo = EventRepository()


@router.get("/", response_model=list[EventResponse])
def get_all_events():
    return repo.get_all()


@router.get("/{event_id}", response_model=EventResponse)
def get_event(event_id: str):
    event = repo.get_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event bulunamadı")
    return event


@router.post("/", response_model=dict, status_code=201)
def create_event(body: EventCreate):
    event_id = repo.create(body.eventName)
    return {"id": event_id}


@router.delete("/{event_id}", status_code=204)
def delete_event(event_id: str):
    deleted = repo.delete(event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Event bulunamadı")

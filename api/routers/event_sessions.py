from fastapi import APIRouter, HTTPException
from database import EventSessionRepository
from api.schemas.event_session import EventSessionCreate, EventSessionUpdate, EventSessionResponse

router = APIRouter(prefix="/event-sessions", tags=["Event Sessions"])
repo = EventSessionRepository()


@router.get("/by-event/{event_id}", response_model=list[EventSessionResponse])
def get_sessions_by_event(event_id: str):
    return repo.get_by_event(event_id)


@router.get("/{session_id}", response_model=EventSessionResponse)
def get_session(session_id: str):
    session = repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session bulunamadı")
    return session


@router.post("/", response_model=dict, status_code=201)
def create_session(body: EventSessionCreate):
    session_id = repo.create(
        session_name=body.sessionName,
        event_id=body.eventId,
        duration=body.duration,
        date=body.date,
    )
    return {"id": session_id}


@router.put("/{session_id}/duration", response_model=dict)
def update_duration(session_id: str, body: EventSessionUpdate):
    updated = repo.update_duration(session_id, body.duration)
    if not updated:
        raise HTTPException(status_code=404, detail="Session bulunamadı")
    return {"updated": True}


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str):
    deleted = repo.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session bulunamadı")

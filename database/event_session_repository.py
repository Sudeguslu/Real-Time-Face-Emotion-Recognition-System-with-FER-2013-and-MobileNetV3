from datetime import datetime
from zoneinfo import ZoneInfo
from bson import ObjectId
from .client import get_db
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ISTANBUL = ZoneInfo("Europe/Istanbul")


class EventSessionRepository:
    def __init__(self):
        self._col = get_db()["eventSessions"]

    def create(self, session_name: str, event_id: str, duration: float, date: datetime | None = None) -> str:
        result = self._col.insert_one({
            "sessionName": session_name,
            "eventId":     event_id,
            "duration":    duration,
            "date":        date or datetime.now(ISTANBUL),
        })
        return str(result.inserted_id)

    def get_by_id(self, session_id: str) -> dict | None:
        doc = self._col.find_one({"_id": ObjectId(session_id)})
        return _serialize(doc)

    def get_by_event(self, event_id: str) -> list[dict]:
        return [_serialize(d) for d in self._col.find({"eventId": event_id})]

    def update_duration(self, session_id: str, duration: float) -> bool:
        result = self._col.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"duration": duration}},
        )
        return result.modified_count > 0

    def delete(self, session_id: str) -> bool:
        result = self._col.delete_one({"_id": ObjectId(session_id)})
        return result.deleted_count > 0


def _serialize(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc["id"] = str(doc.pop("_id"))
    if isinstance(doc.get("date"), datetime):
        dt = doc["date"]
        # MongoDB naive datetime döndürüyorsa UTC olarak işaretle
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        doc["date"] = dt.astimezone(ISTANBUL).isoformat()
    return doc
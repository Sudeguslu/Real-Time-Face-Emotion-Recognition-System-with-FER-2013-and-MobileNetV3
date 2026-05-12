from datetime import datetime
from zoneinfo import ZoneInfo
from bson import ObjectId
from .client import get_db
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


ISTANBUL = ZoneInfo("Europe/Istanbul")


class EmotionRepository:
    def __init__(self):
        self._col = get_db()["capturedEmotions"]

    def save(self, emotion: str, session_id: str, date: datetime | None = None) -> str:
        result = self._col.insert_one({
            "emotion":   emotion,
            "sessionId": session_id,
            "date":      date or datetime.now(ISTANBUL),
        })
        return str(result.inserted_id)

    def get_by_session(self, session_id: str) -> list[dict]:
        return [_serialize(d) for d in self._col.find({"sessionId": session_id})]

    def count_by_emotion(self, session_id: str) -> dict[str, int]:
        pipeline = [
            {"$match": {"sessionId": session_id}},
            {"$group": {"_id": "$emotion", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        return {doc["_id"]: doc["count"] for doc in self._col.aggregate(pipeline)}

    def delete_by_session(self, session_id: str) -> int:
        result = self._col.delete_many({"sessionId": session_id})
        return result.deleted_count


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
from datetime import datetime
from zoneinfo import ZoneInfo
from bson import ObjectId
from .client import get_db
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ISTANBUL = ZoneInfo("Europe/Istanbul")


class LogRepository:
    def __init__(self):
        self._col = get_db()["logs"]

    def save(self, exception_message: str, exception_detail: str = "") -> str:
        result = self._col.insert_one({
            "exceptionMessage": exception_message,
            "exceptionDetail":  exception_detail,
            "date":             datetime.now(ISTANBUL),
        })
        return str(result.inserted_id)

    def get_all(self, limit: int = 100) -> list[dict]:
        cursor = self._col.find().sort("date", -1).limit(limit)
        return [_serialize(d) for d in cursor]

    def get_by_id(self, log_id: str) -> dict | None:
        doc = self._col.find_one({"_id": ObjectId(log_id)})
        return _serialize(doc)


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
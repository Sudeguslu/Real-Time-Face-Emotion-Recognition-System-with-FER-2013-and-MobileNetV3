from datetime import datetime, timezone
from bson import ObjectId
from .client import get_db


class LogRepository:
    def __init__(self):
        self._col = get_db()["logs"]

    def save(self, exception_message: str, exception_detail: str = "") -> str:
        result = self._col.insert_one({
            "exceptionMessage": exception_message,
            "exceptionDetail":  exception_detail,
            "date":             datetime.now(timezone.utc),
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
    return doc

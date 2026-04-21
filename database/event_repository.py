from bson import ObjectId
from .client import get_db


class EventRepository:
    def __init__(self):
        self._col = get_db()["events"]

    def create(self, event_name: str) -> str:
        result = self._col.insert_one({"eventName": event_name})
        return str(result.inserted_id)

    def get_by_id(self, event_id: str) -> dict | None:
        doc = self._col.find_one({"_id": ObjectId(event_id)})
        return _serialize(doc)

    def get_all(self) -> list[dict]:
        return [_serialize(d) for d in self._col.find()]

    def delete(self, event_id: str) -> bool:
        result = self._col.delete_one({"_id": ObjectId(event_id)})
        return result.deleted_count > 0


def _serialize(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc["id"] = str(doc.pop("_id"))
    return doc

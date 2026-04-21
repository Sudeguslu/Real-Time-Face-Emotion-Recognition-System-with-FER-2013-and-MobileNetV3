from pymongo import MongoClient
from pymongo.database import Database
from .config import MONGO_URI, DB_NAME

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client


def get_db() -> Database:
    return get_client()[DB_NAME]


def close_connection() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None

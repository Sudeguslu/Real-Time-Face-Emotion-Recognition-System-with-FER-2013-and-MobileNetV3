from datetime import datetime
from pydantic import BaseModel


class LogCreate(BaseModel):
    exceptionMessage: str
    exceptionDetail: str = ""


class LogResponse(BaseModel):
    id: str
    exceptionMessage: str
    exceptionDetail: str
    date: datetime

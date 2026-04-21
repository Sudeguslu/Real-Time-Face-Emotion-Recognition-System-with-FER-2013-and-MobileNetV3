from fastapi import APIRouter, HTTPException
from database import LogRepository
from api.schemas.log import LogCreate, LogResponse

router = APIRouter(prefix="/logs", tags=["Logs"])
repo = LogRepository()


@router.get("/", response_model=list[LogResponse])
def get_all_logs(limit: int = 100):
    return repo.get_all(limit=limit)


@router.get("/{log_id}", response_model=LogResponse)
def get_log(log_id: str):
    log = repo.get_by_id(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log bulunamadı")
    return log


@router.post("/", response_model=dict, status_code=201)
def save_log(body: LogCreate):
    log_id = repo.save(body.exceptionMessage, body.exceptionDetail)
    return {"id": log_id}

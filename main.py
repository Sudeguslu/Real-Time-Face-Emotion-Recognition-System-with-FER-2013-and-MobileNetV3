from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import events, event_sessions, emotions, logs

app = FastAPI(
    title="Emotion Recognition API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events.router)
app.include_router(event_sessions.router)
app.include_router(emotions.router)
app.include_router(logs.router)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok"}

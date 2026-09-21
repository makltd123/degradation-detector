"""FastAPI-приложение Silent Degradation Detector."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import db, pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Silent Degradation Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunCreate(BaseModel):
    day_label: str | None = None
    config: dict = {}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/setup")
def setup():
    return pipeline.run_setup()


@app.post("/api/runs", status_code=202)
def create_run(body: RunCreate):
    try:
        return pipeline.start_run(body.day_label, body.config)
    except pipeline.RunBusy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except pipeline.SetupNeeded as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/runs")
def get_runs():
    return pipeline.list_runs()


@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: int):
    detail = pipeline.get_run(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    return detail


@app.get("/api/runs/{run_id}/diff")
def get_run_diff(run_id: int):
    diff = pipeline.diff_with_baseline(run_id)
    if diff is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    return diff

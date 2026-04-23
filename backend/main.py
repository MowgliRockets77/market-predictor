import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .database import init_db
from .scanner import run_full_scan
from .routes import router

scheduler = AsyncIOScheduler(timezone="America/New_York")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    scheduler.add_job(
        run_full_scan, "cron",
        day_of_week="mon-fri", hour="9-17", minute=0,
        id="hourly_scan", replace_existing=True,
    )
    scheduler.add_job(
        run_full_scan, "cron",
        day_of_week="mon-fri", hour=16, minute=5,
        id="eod_scan", replace_existing=True,
    )
    scheduler.start()
    asyncio.create_task(run_full_scan())
    yield
    scheduler.shutdown()

app = FastAPI(title="Market Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

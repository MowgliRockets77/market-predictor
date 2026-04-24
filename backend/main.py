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

    # Full scan every 5 minutes during market hours Mon-Fri
    scheduler.add_job(
        run_full_scan,
        "cron",
        day_of_week="mon-fri",
        hour="9-16",
        minute="0,5,10,15,20,25,30,35,40,45,50,55",
        id="frequent_scan",
        replace_existing=True,
    )

    # After-hours scan every 30 minutes to catch news
    scheduler.add_job(
        run_full_scan,
        "cron",
        day_of_week="mon-fri",
        hour="4-8,17-23",
        minute="0,30",
        id="afterhours_scan",
        replace_existing=True,
    )

    # Weekend scan every hour
    scheduler.add_job(
        run_full_scan,
        "cron",
        day_of_week="sat,sun",
        minute=0,
        id="weekend_scan",
        replace_existing=True,
    )

    scheduler.start()
    # Run scan immediately on startup
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

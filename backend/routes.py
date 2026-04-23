from fastapi import APIRouter, BackgroundTasks, HTTPException
from .database import (
    fetch_latest_scan,
    fetch_all_predictions,
    fetch_ticker_predictions,
    fetch_accuracy_stats,
    fetch_top_picks,
)
from .scanner import run_full_scan, get_last_scan

router = APIRouter()

@router.get("/api/scan/latest")
async def latest_scan():
    results, scan_time = get_last_scan()
    return {"scan_time": scan_time, "results": results}

@router.get("/api/scan/top")
async def top_picks(limit: int = 10):
    return fetch_top_picks(limit)

@router.get("/api/scan/trigger")
async def trigger_scan(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_full_scan)
    return {"message": "Scan started"}

@router.get("/api/stats")
async def accuracy_stats():
    return fetch_accuracy_stats()

@router.get("/api/ticker/{ticker}")
async def ticker_detail(ticker: str):
    preds = fetch_ticker_predictions(ticker.upper())
    if not preds:
        raise HTTPException(status_code=404, detail="No predictions found")
    finished    = [p for p in preds if p["actual"] is not None]
    accuracy    = round(sum(p["correct"] for p in finished) / len(finished) * 100, 1) if finished else None
    return {
        "ticker":           ticker.upper(),
        "predictions":      preds,
        "tracked_accuracy": accuracy,
        "total":            len(preds),
        "resolved":         len(finished),
    }

@router.get("/api/predictions")
async def all_predictions(limit: int = 500):
    return fetch_all_predictions(limit)

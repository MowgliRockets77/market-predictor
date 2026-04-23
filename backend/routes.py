from fastapi import APIRouter, BackgroundTasks, HTTPException
from .database import (
    fetch_latest_scan,
    fetch_all_predictions,
    fetch_ticker_predictions,
    fetch_accuracy_stats,
    fetch_top_picks,
    fetch_sentiment_history,
)
from .scanner import run_full_scan, get_last_scan
from .sentiment import get_sentiment

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

    # Build accuracy streak analysis
    streak_correct = 0
    streak_wrong   = 0
    for p in finished[:10]:
        if p["correct"] == 1:
            streak_correct += 1
        else:
            streak_wrong += 1

    # Sentiment history
    sentiment_history = fetch_sentiment_history(ticker.upper(), limit=10)

    return {
        "ticker":             ticker.upper(),
        "predictions":        preds,
        "tracked_accuracy":   accuracy,
        "total":              len(preds),
        "resolved":           len(finished),
        "recent_correct":     streak_correct,
        "recent_wrong":       streak_wrong,
        "sentiment_history":  sentiment_history,
    }

@router.get("/api/ticker/{ticker}/sentiment")
async def ticker_sentiment(ticker: str):
    """Live sentiment analysis with explanation for a single ticker."""
    data = get_sentiment(ticker.upper())

    # Build explanation
    score   = data["combined_score"]
    label   = data["label"]
    sources = data["source_scores"]
    counts  = data["source_counts"]

    # Most positive and negative sources
    scored_sources = [(s, v) for s, v in sources.items() if counts.get(s, 0) > 0]
    scored_sources.sort(key=lambda x: x[1], reverse=True)

    best_source  = scored_sources[0]  if scored_sources else None
    worst_source = scored_sources[-1] if scored_sources else None

    # Generate plain-English explanation
    if label == "POSITIVE":
        reason = f"News sentiment for {ticker.upper()} is broadly positive (score: {score:+.3f}). "
    elif label == "NEGATIVE":
        reason = f"News sentiment for {ticker.upper()} is broadly negative (score: {score:+.3f}). "
    else:
        reason = f"News sentiment for {ticker.upper()} is neutral (score: {score:+.3f}). "

    if best_source:
        reason += f"Most positive coverage came from {best_source[0].title()} (score: {best_source[1]:+.3f}). "
    if worst_source and worst_source != best_source:
        reason += f"Most negative coverage came from {worst_source[0].title()} (score: {worst_source[1]:+.3f}). "

    reason += f"Analysis based on {data['total_headlines']} headlines across {len(scored_sources)} sources."

    return {
        "ticker":          ticker.upper(),
        "combined_score":  score,
        "label":           label,
        "reason":          reason,
        "total_headlines": data["total_headlines"],
        "source_breakdown": [
            {
                "source":    s,
                "score":     round(v, 3),
                "headlines": counts.get(s, 0),
                "label":     "POSITIVE" if v >= 0.05 else "NEGATIVE" if v <= -0.05 else "NEUTRAL"
            }
            for s, v in sources.items()
        ],
        "timestamp": data["timestamp"],
    }

@router.get("/api/predictions")
async def all_predictions(limit: int = 500):
    return fetch_all_predictions(limit)

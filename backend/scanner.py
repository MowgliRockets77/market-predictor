import asyncio
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .database import save_prediction, update_actuals_for_ticker
from .sentiment import get_sentiment

logger = logging.getLogger(__name__)

FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")
AV_KEY      = os.getenv("ALPHA_VANTAGE_KEY", "")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "market-predictor/1.0"})

TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","AVGO","NFLX",
    "PLTR","JPM","XOM","LLY","UNH","JNJ","V","MA","COST","WMT",
    "PG","HD","ABBV","KO","MRK","PEP","ADBE","CSCO","CRM","ORCL",
    "INTC","QCOM","BAC","GS","MS","CVX","DIS","MCD","NKE","TXN",
    "AMGN","CAT","GE","IBM","RTX","SPGI","BLK","PANW","UBER","INTU",
    "ISRG","BKNG","NOW","AMAT","MU","SNOW","NET","CRWD","COIN","PYPL",
]

FEATURE_COLS = [
    "ret1","ret3","ret5","ret10",
    "sma_gap_5_10","sma_gap_10_20",
    "vol10","vol20","sentiment",
]

def fetch_price_history_finnhub(ticker: str) -> pd.Series | None:
    """Fetch 2 years of daily closes from Finnhub."""
    try:
        import time
        end   = int(time.time())
        start = end - 2 * 365 * 24 * 3600
        url   = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={start}&to={end}&token={FINNHUB_KEY}"
        r     = SESSION.get(url, timeout=10)
        data  = r.json()
        if data.get("s") != "ok" or not data.get("c"):
            return None
        closes = pd.Series(
            data["c"],
            index=pd.to_datetime(data["t"], unit="s", utc=True).tz_convert("America/New_York")
        )
        return closes.dropna()
    except Exception as e:
        logger.warning(f"Finnhub price fetch failed {ticker}: {e}")
        return None

def fetch_price_history_av(ticker: str) -> pd.Series | None:
    """Fetch daily closes from Alpha Vantage as fallback."""
    try:
        url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
               f"&symbol={ticker}&outputsize=full&apikey={AV_KEY}")
        r    = SESSION.get(url, timeout=15)
        data = r.json()
        ts   = data.get("Time Series (Daily)", {})
        if not ts:
            return None
        closes = pd.Series(
            {pd.Timestamp(d): float(v["5. adjusted close"]) for d, v in ts.items()}
        ).sort_index()
        return closes.dropna().tail(500)
    except Exception as e:
        logger.warning(f"Alpha Vantage fetch failed {ticker}: {e}")
        return None

def fetch_price_history(ticker: str) -> pd.Series | None:
    """Try Finnhub first, fall back to Alpha Vantage."""
    close = fetch_price_history_finnhub(ticker)
    if close is not None and len(close) >= 120:
        return close
    logger.info(f"[{ticker}] Finnhub failed, trying Alpha Vantage")
    return fetch_price_history_av(ticker)

def make_features(close: pd.Series, sentiment_score: float = 0.0) -> pd.DataFrame:
    df = pd.DataFrame({"Close": close})
    df["ret1"]          = df["Close"].pct_change(1)
    df["ret3"]          = df["Close"].pct_change(3)
    df["ret5"]          = df["Close"].pct_change(5)
    df["ret10"]         = df["Close"].pct_change(10)
    sma5                = df["Close"].rolling(5).mean()
    sma10               = df["Close"].rolling(10).mean()
    sma20               = df["Close"].rolling(20).mean()
    df["sma_gap_5_10"]  = sma5  / sma10  - 1
    df["sma_gap_10_20"] = sma10 / sma20  - 1
    df["vol10"]         = df["ret1"].rolling(10).std()
    df["vol20"]         = df["ret1"].rolling(20).std()
    df["sentiment"]     = sentiment_score
    df["target"]        = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna().copy()

def train_and_predict(feat_df: pd.DataFrame):
    X, y = feat_df[FEATURE_COLS], feat_df["target"]
    split = int(len(feat_df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    holdout_acc = float((model.predict(X_test) == y_test).mean()) if len(y_test) else float("nan")
    prob_up     = float(model.predict_proba(X.tail(1))[0][1])
    if prob_up >= 0.60:
        prediction, recommendation = "UP", "BUY"
    elif prob_up <= 0.40:
        prediction, recommendation = "DOWN", "SELL"
    else:
        prediction     = "UP" if prob_up >= 0.50 else "DOWN"
        recommendation = "HOLD"
    return prob_up, prediction, recommendation, holdout_acc

def process_ticker(ticker: str, scan_time: str) -> dict | None:
    try:
        close = fetch_price_history(ticker)
        if close is None or len(close) < 120:
            logger.warning(f"[{ticker}] insufficient price data")
            return None

        update_actuals_for_ticker(ticker, close)

        sentiment_data  = get_sentiment(ticker)
        sentiment_score = sentiment_data["combined_score"]
        sentiment_label = sentiment_data["label"]
        total_headlines = sentiment_data["total_headlines"]

        feat_df = make_features(close, sentiment_score)
        if len(feat_df) < 100:
            return None

        prob_up, prediction, recommendation, holdout_acc = train_and_predict(feat_df)

        if sentiment_label == "POSITIVE" and recommendation == "BUY":
            prob_up = min(prob_up * 1.05, 0.99)
        elif sentiment_label == "NEGATIVE" and recommendation == "SELL":
            prob_up = max(prob_up * 0.95, 0.01)

        price       = float(close.iloc[-1])
        ret5        = float(close.pct_change(5).iloc[-1] * 100)
        ret20       = float(close.pct_change(20).iloc[-1] * 100)
        pred_date   = close.index[-1].strftime("%Y-%m-%d")
        target_date = (close.index[-1] + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")

        save_prediction(
            scan_time, ticker, pred_date, target_date,
            prob_up, prediction, recommendation,
            price=price, ret5=ret5, ret20=ret20,
            holdout_acc=holdout_acc if not np.isnan(holdout_acc) else None,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            headline_count=total_headlines,
        )

        return {
            "ticker":          ticker,
            "price":           price,
            "prob_up":         round(prob_up * 100, 1),
            "prediction":      prediction,
            "recommendation":  recommendation,
            "ret5":            round(ret5, 2),
            "ret20":           round(ret20, 2),
            "holdout_acc":     round(holdout_acc * 100, 1) if not np.isnan(holdout_acc) else None,
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "headline_count":  total_headlines,
        }
    except Exception as e:
        logger.warning(f"[{ticker}] failed: {e}")
        return None

_scan_lock         = asyncio.Lock()
_last_scan_results: list = []
_last_scan_time:    str | None = None

async def run_full_scan():
    global _last_scan_results, _last_scan_time
    async with _scan_lock:
        scan_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting scan at {scan_time}")
        loop    = asyncio.get_event_loop()
        results = []
        for ticker in TICKERS:
            result = await loop.run_in_executor(None, process_ticker, ticker, scan_time)
            if result:
                results.append(result)
            await asyncio.sleep(0.5)
        results.sort(key=lambda x: x["prob_up"], reverse=True)
        _last_scan_results = results
        _last_scan_time    = scan_time
        logger.info(f"Scan complete: {len(results)} tickers")
        return results

def get_last_scan():
    return _last_scan_results, _last_scan_time

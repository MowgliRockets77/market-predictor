import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .database import save_prediction, update_actuals_for_ticker

logger = logging.getLogger(__name__)

# Fix for yfinance being blocked on cloud servers
yf.utils.get_json = lambda url, proxy=None, session=None: {}

import requests
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})

TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","AVGO","NFLX",
    "PLTR","JPM","XOM","LLY","UNH","JNJ","V","MA","COST","WMT",
    "PG","HD","ABBV","KO","MRK","PEP","ADBE","CSCO","CRM","ORCL",
    "INTC","QCOM","BAC","GS","MS","CVX","DIS","MCD","NKE","TXN",
    "AMGN","CAT","GE","IBM","RTX","SPGI","BLK","PANW","UBER","INTU",
    "ISRG","BKNG","NOW","AMAT","MU","SNOW","NET","CRWD","COIN","PYPL",
]

FEATURE_COLS = ["ret1","ret3","ret5","ret10","sma_gap_5_10","sma_gap_10_20","vol10","vol20"]

def make_features(close):
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
    df["target"]        = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna().copy()

def train_and_predict(feat_df):
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

def process_ticker(ticker, scan_time):
    try:
        tk = yf.Ticker(ticker, session=_SESSION)
        df = tk.history(period="2y", interval="1d", auto_adjust=True)
        if df.empty:
            return None
        close = df["Close"]
        if hasattr(close, "ndim") and close.ndim > 1:
            close = close.iloc[:, 0]
        close = pd.Series(close).dropna()
        if len(close) < 120:
            return None
        update_actuals_for_ticker(ticker, close)
        feat_df = make_features(close)
        if len(feat_df) < 100:
            return None
        prob_up, prediction, recommendation, holdout_acc = train_and_predict(feat_df)
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
        )
        return {
            "ticker": ticker, "price": price,
            "prob_up": round(prob_up * 100, 1),
            "prediction": prediction, "recommendation": recommendation,
            "ret5": round(ret5, 2), "ret20": round(ret20, 2),
            "holdout_acc": round(holdout_acc * 100, 1) if not np.isnan(holdout_acc) else None,
        }
    except Exception as e:
        logger.warning(f"[{ticker}] failed: {e}")
        return None

_scan_lock = asyncio.Lock()
_last_scan_results = []
_last_scan_time = None

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

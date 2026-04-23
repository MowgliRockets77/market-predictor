"""
Sentiment analyzer for stocks.
Sources: Yahoo Finance, Google News RSS, Bing News RSS, Reddit, SEC EDGAR, Finviz
Scoring: VADER (fast, finance-aware)
"""
import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_ANALYZER = SentimentIntensityAnalyzer()

# Add finance-specific words to VADER lexicon
FINANCE_LEXICON = {
    "bullish": 3.0, "bearish": -3.0,
    "surge": 2.5, "plunge": -2.5,
    "beat": 2.0, "miss": -2.0,
    "upgrade": 2.0, "downgrade": -2.0,
    "outperform": 2.0, "underperform": -2.0,
    "rally": 2.0, "selloff": -2.5,
    "buyback": 1.5, "layoffs": -2.0,
    "bankruptcy": -3.5, "fraud": -3.5,
    "record": 1.5, "loss": -1.5,
    "profit": 1.5, "revenue": 1.0,
    "growth": 1.5, "decline": -1.5,
    "innovative": 1.5, "lawsuit": -2.0,
}
_ANALYZER.lexicon.update(FINANCE_LEXICON)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

def _score(text: str) -> float:
    """Return compound VADER score -1 to +1."""
    if not text:
        return 0.0
    return _ANALYZER.polarity_scores(str(text))["compound"]

def _fetch_rss(url: str, timeout: int = 8) -> list[str]:
    """Fetch RSS feed and return list of headline strings."""
    try:
        r = _SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        headlines = []
        for item in root.iter("item"):
            title = item.find("title")
            desc  = item.find("description")
            text  = (title.text or "" if title is not None else "") + " " + \
                    (desc.text  or "" if desc  is not None else "")
            headlines.append(text.strip())
        return headlines[:20]
    except Exception as e:
        logger.debug(f"RSS fetch failed {url}: {e}")
        return []

# ── Source functions ──────────────────────────────────────────────────────────

def fetch_yahoo_news(ticker: str) -> list[str]:
    try:
        import yfinance as yf
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome110")
        tk = yf.Ticker(ticker, session=session)
        news = tk.news or []
        return [n.get("content", {}).get("title", "") or n.get("title", "") for n in news[:15]]
    except Exception as e:
        logger.debug(f"Yahoo news failed {ticker}: {e}")
        return []

def fetch_google_news(ticker: str, company: str = "") -> list[str]:
    query = quote(f"{ticker} stock {company}")
    url   = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    return _fetch_rss(url)

def fetch_bing_news(ticker: str) -> list[str]:
    query = quote(f"{ticker} stock market")
    url   = f"https://www.bing.com/news/search?q={query}&format=RSS"
    return _fetch_rss(url)

def fetch_reddit(ticker: str) -> list[str]:
    headlines = []
    subreddits = ["stocks", "investing", "wallstreetbets", "StockMarket"]
    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/search.json?q={ticker}&sort=new&limit=10&restrict_sr=1"
            r   = _SESSION.get(url, timeout=8, headers={"User-Agent": "market-predictor/1.0"})
            if r.status_code == 200:
                posts = r.json().get("data", {}).get("children", [])
                for p in posts[:5]:
                    d = p.get("data", {})
                    headlines.append(d.get("title", "") + " " + d.get("selftext", "")[:200])
        except Exception as e:
            logger.debug(f"Reddit {sub} failed: {e}")
    return headlines

def fetch_sec_edgar(ticker: str) -> list[str]:
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={(datetime.now()-timedelta(days=30)).strftime('%Y-%m-%d')}&enddt={datetime.now().strftime('%Y-%m-%d')}&forms=8-K"
    try:
        r = _SESSION.get(url, timeout=8)
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            return [h.get("_source", {}).get("file_date", "") + " " +
                    h.get("_source", {}).get("display_names", [""])[0]
                    for h in hits[:10]]
    except Exception as e:
        logger.debug(f"SEC EDGAR failed {ticker}: {e}")
    return []

def fetch_finviz(ticker: str) -> list[str]:
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        r   = _SESSION.get(url, timeout=8)
        if r.status_code != 200:
            return []
        from html.parser import HTMLParser

        class NewsParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.headlines = []
                self._in_news  = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if attrs.get("class") == "nn-tab-link":
                    self._in_news = True

            def handle_data(self, data):
                if self._in_news and data.strip():
                    self.headlines.append(data.strip())
                    self._in_news = False

        p = NewsParser()
        p.feed(r.text)
        return p.headlines[:15]
    except Exception as e:
        logger.debug(f"Finviz failed {ticker}: {e}")
        return []

# ── Main aggregator ───────────────────────────────────────────────────────────

def get_sentiment(ticker: str, company: str = "") -> dict:
    """
    Aggregate sentiment from all free sources.
    Returns dict with scores per source and combined score.
    """
    all_headlines: dict[str, list[str]] = {}

    all_headlines["yahoo"]   = fetch_yahoo_news(ticker)
    all_headlines["google"]  = fetch_google_news(ticker, company)
    all_headlines["bing"]    = fetch_bing_news(ticker)
    all_headlines["reddit"]  = fetch_reddit(ticker)
    all_headlines["sec"]     = fetch_sec_edgar(ticker)
    all_headlines["finviz"]  = fetch_finviz(ticker)

    source_scores: dict[str, float] = {}
    source_counts: dict[str, int]   = {}
    all_scores: list[float]         = []

    for source, headlines in all_headlines.items():
        scores = [_score(h) for h in headlines if h.strip()]
        if scores:
            avg = sum(scores) / len(scores)
            source_scores[source] = round(avg, 4)
            source_counts[source] = len(scores)
            all_scores.extend(scores)
        else:
            source_scores[source] = 0.0
            source_counts[source] = 0

    total_headlines = sum(source_counts.values())
    combined = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    # Classify
    if combined >= 0.05:
        label = "POSITIVE"
    elif combined <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "ticker":           ticker,
        "combined_score":   combined,
        "label":            label,
        "total_headlines":  total_headlines,
        "source_scores":    source_scores,
        "source_counts":    source_counts,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

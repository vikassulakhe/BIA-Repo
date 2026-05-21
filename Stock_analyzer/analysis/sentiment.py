# ==============================
# analysis/sentiment.py
# ==============================
import requests
from bs4 import BeautifulSoup
import os
from typing import Any, cast
SENTIMENT_BACKEND = os.environ.get('SENTIMENT_BACKEND', 'textblob').lower()

# Use VADER (preferred) or TextBlob for sentiment scoring.
_VADER = None
_USE_VADER = False
_TEXTBLOB = None
_USE_TEXTBLOB = False

if SENTIMENT_BACKEND in ('auto', 'vader'):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
        _USE_VADER = True
    except Exception:
        _USE_VADER = False
else:
    _USE_VADER = False

if SENTIMENT_BACKEND in ('auto', 'textblob'):
    try:
        from textblob import TextBlob
        _TEXTBLOB = TextBlob
        _USE_TEXTBLOB = True
    except Exception:
        _USE_TEXTBLOB = False
else:
    _USE_TEXTBLOB = False


def _clean_headlines_from_soup(soup, max_headlines=8):
    raw = [h.text.strip() for h in soup.find_all('a')]
    bad_tokens = set(['', 'Sign in', 'Home', 'For you', 'News', 'Search'])
    headlines = []
    seen = set()
    for h in raw:
        if not h or h in bad_tokens:
            continue
        if len(h) < 30:
            continue
        if h in seen:
            continue
        seen.add(h)
        headlines.append(h)
        if len(headlines) >= max_headlines:
            break
    return headlines


def _score_with_vader(headlines):
    if not _USE_VADER:
        return None
    scores = []
    for h in headlines:
        try:
            v = cast(Any, _VADER).polarity_scores(h)
            scores.append(v.get('compound', 0.0))
        except Exception:
            scores.append(0.0)
    return scores


def _score_with_textblob(headlines):
    if not _USE_TEXTBLOB:
        return None
    scores = []
    for h in headlines:
        try:
            scores.append(float(cast(Any, _TEXTBLOB)(h).sentiment.polarity))
        except Exception:
            scores.append(0.0)
    return scores


def get_sentiment(ticker, backend=None):
    """Compute sentiment using configured backend.

    Parameters:
    - ticker: symbol or query used for news search
    - backend: optional override: 'auto', 'vader', or 'textblob'
    """
    url = f"https://news.google.com/search?q={ticker}"
    try:
        res = requests.get(url, timeout=6)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
    except Exception:
        return {"Sentiment": None, "Headlines": [], "error": "Failed to fetch headlines."}

    headlines = _clean_headlines_from_soup(soup, max_headlines=8)

    # decide backend: override if provided
    eff = (backend or SENTIMENT_BACKEND or 'auto').lower()

    scores = None
    if eff in ('auto', 'vader') and _USE_VADER:
        scores = _score_with_vader(headlines)
    if (scores is None or not scores) and eff in ('auto', 'textblob') and _USE_TEXTBLOB:
        scores = _score_with_textblob(headlines)

    if scores is None:
        return {"Sentiment": None, "Headlines": headlines[:5], "error": "No sentiment backend available; install vaderSentiment or textblob."}

    scores = [float(s) for s in scores]
    avg = round(sum(scores) / len(scores), 2) if scores else 0.0
    return {"Sentiment": avg, "Headlines": headlines[:5]}
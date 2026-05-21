# ==============================
# data/fetch_data.py
# ==============================
import yfinance as yf
import requests
from typing import Optional, Any, cast


def _search_yahoo_symbol(query: str) -> Optional[str]:
    url = 'https://query1.finance.yahoo.com/v1/finance/search'
    try:
        r = requests.get(url, params={'q': query}, timeout=5)
        r.raise_for_status()
        j = r.json()
        quotes = j.get('quotes') or []
        for q in quotes:
            if q.get('quoteType') == 'EQUITY' and q.get('symbol'):
                return q.get('symbol')
    except Exception:
        return None
    return None


def get_stock(ticker: str):
    """Return a yfinance Ticker. If the provided `ticker` yields no recent data,
    attempt to resolve via Yahoo search (company name -> symbol) and return that Ticker.
    The resolved symbol is attached as attribute `.resolved_symbol` on the returned object.
    """
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period='5d')
        if hist is not None and not hist.empty:
            cast(Any, t).resolved_symbol = ticker
            return t
    except Exception:
        pass

    # try uppercased direct variants
    up = ticker.strip().upper()
    if up != ticker:
        t2 = yf.Ticker(up)
        try:
            h2 = t2.history(period='5d')
            if h2 is not None and not h2.empty:
                cast(Any, t2).resolved_symbol = up
                return t2
        except Exception:
            pass

    # try common Indian exchange suffixes when input looks like a company name
    if '.' not in up:
        for suf in ['.NS', '.BO']:
            try:
                cand = f"{up}{suf}"
                t_cand = yf.Ticker(cand)
                h_cand = t_cand.history(period='5d')
                if h_cand is not None and not h_cand.empty:
                    cast(Any, t_cand).resolved_symbol = cand
                    return t_cand
            except Exception:
                continue

    # fallback: search Yahoo for a matching equity symbol
    sym = _search_yahoo_symbol(ticker)
    if sym:
        # prefer exchange-qualified NSE symbol if available (common for Indian tickers)
        t3 = yf.Ticker(sym)
        try:
            # if symbol has no dot (no exchange suffix), try adding .NS
            if '.' not in str(sym):
                alt = f"{sym}.NS"
                t_alt = yf.Ticker(alt)
                h_alt = None
                try:
                    h_alt = t_alt.history(period='5d')
                except Exception:
                    h_alt = None
                if h_alt is not None and not h_alt.empty:
                    t3 = t_alt
                    sym = alt
        except Exception:
            pass
        try:
            h3 = t3.history(period='5d')
            # return ticker even if history empty; caller should check
            cast(Any, t3).resolved_symbol = sym
            return t3
        except Exception:
            cast(Any, t3).resolved_symbol = sym
            return t3

    # final fallback: return original Ticker with no resolved_symbol
    cast(Any, t).resolved_symbol = ticker
    return t
# ==============================
# analysis/fundamentals.py
# ==============================
import numpy as np


def _find_row(df, candidates):
    if df is None or df.empty:
        return None
    idx = list(df.index)
    low_idx = [str(i).lower() for i in idx]
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(low_idx):
            if name == c:
                return idx[i]
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(low_idx):
            if c in name:
                return idx[i]
    # generic fallbacks
    for i, name in enumerate(low_idx):
        if 'equity' in name and ('stock' in name or 'share' in name or 'holders' in name):
            return idx[i]
    for i, name in enumerate(low_idx):
        if 'debt' in name:
            return idx[i]
    for i, name in enumerate(low_idx):
        if 'ebit' in name or 'operating income' in name:
            return idx[i]
    for i, name in enumerate(low_idx):
        if 'net income' in name or 'net income common' in name or 'net income including' in name:
            return idx[i]
    return None


def _get_first_value(series):
    try:
        vals = series.dropna()
        if vals.empty:
            return None
        return float(vals.iloc[0])
    except Exception:
        try:
            return float(series.iloc[0])
        except Exception:
            return None


def _get_row_value(df, label):
    """Safely get a numeric value from DataFrame `df` at index `label`.
    Returns the first non-null float or None."""
    if df is None or label is None:
        return None
    try:
        if label in df.index:
            return _get_first_value(df.loc[label])
    except Exception:
        # fall through to fuzzy match
        pass

    # fuzzy match by lowercased containment
    try:
        for idx in df.index:
            try:
                if str(label).lower() in str(idx).lower():
                    return _get_first_value(df.loc[idx])
            except Exception:
                continue
    except Exception:
        return None
    return None


def get_financial_ratios(stock):
    bs = getattr(stock, 'balance_sheet', None)
    fin = getattr(stock, 'financials', None)

    equity_labels = ["Total Stockholder Equity", "Stockholders Equity", "Total Equity", "Shareholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]
    debt_labels = ["Total Debt", "Net Debt", "Long Term Debt", "Current Debt", "Current Debt And Capital Lease Obligation"]
    ebit_labels = ["EBIT", "Ebit", "Operating Income", "Total Operating Income As Reported"]
    net_income_labels = ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests", "Net Income From Continuing Operation Net Minority Interest"]

    try:
        eq_label = _find_row(bs, equity_labels)
        debt_label = _find_row(bs, debt_labels)
        ebit_label = _find_row(fin, ebit_labels)
        ni_label = _find_row(fin, net_income_labels)

        equity = _get_row_value(bs, eq_label)
        debt = _get_row_value(bs, debt_label) or 0.0
        ebit = _get_row_value(fin, ebit_label)
        net_income = _get_row_value(fin, ni_label)

        roe = None
        roce = None
        if equity and net_income is not None and equity != 0:
            roe = (net_income / equity) * 100
        if ebit is not None and (equity is not None):
            denom = (equity + (debt if debt is not None else 0.0))
            if denom != 0:
                roce = (ebit / denom) * 100

        return {"ROE": round(roe, 2) if roe is not None else None, "ROCE": round(roce, 2) if roce is not None else None}
    except Exception:
        return {"ROE": None, "ROCE": None}
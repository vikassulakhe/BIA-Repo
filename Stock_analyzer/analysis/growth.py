# ==============================
# analysis/growth.py
# ==============================
import numpy as np


def _find_row_by_keywords(df, keywords):
    """Find an index label in df whose name matches any of the keywords (case-insensitive) or contains one.

    Returns the matched label or None.
    """
    if df is None or df.empty:
        return None
    idx = list(df.index)
    low_idx = [str(i).lower() for i in idx]

    # exact match preference
    for kw in keywords:
        kwl = kw.lower()
        for i, name in enumerate(low_idx):
            if name == kwl:
                return idx[i]

    # contains match
    for kw in keywords:
        kwl = kw.lower()
        for i, name in enumerate(low_idx):
            if kwl in name:
                return idx[i]

    # generic fallbacks: for revenue look for 'revenue', for profit look for 'net'+'income' or 'profit'
    for i, name in enumerate(low_idx):
        if 'revenue' in name:
            return idx[i]
    for i, name in enumerate(low_idx):
        if ('net' in name and ('income' in name or 'profit' in name)) or ('profit' in name and 'net' in name):
            return idx[i]

    return None


def get_quarterly_growth(stock):
    """Return simple average quarterly growth rates for revenue and profit.

    Handles missing/alternate row labels gracefully and returns NaN when data unavailable.
    """
    q = getattr(stock, 'quarterly_financials', None)
    if q is None or q.empty:
        return {
            "Revenue Growth %": np.nan,
            "Profit Growth %": np.nan,
        }

    # possible labels
    revenue_labels = ["Total Revenue", "Revenue", "Net Revenue", "Total revenue"]
    profit_labels = ["Net Income", "Net Profit", "Profit", "Net income"]

    rev_label = _find_row_by_keywords(q, revenue_labels)
    prof_label = _find_row_by_keywords(q, profit_labels)

    rev_growth = np.nan
    prof_growth = np.nan

    try:
        if rev_label is not None:
            revenue = q.loc[rev_label].astype(float)
            rev_growth = revenue.pct_change().mean() * 100
    except Exception:
        rev_growth = np.nan

    try:
        if prof_label is not None:
            profit = q.loc[prof_label].astype(float)
            prof_growth = profit.pct_change().mean() * 100
    except Exception:
        prof_growth = np.nan

    return {
        "Revenue Growth %": round(float(rev_growth) if not np.isnan(rev_growth) else np.nan, 2),
        "Profit Growth %": round(float(prof_growth) if not np.isnan(prof_growth) else np.nan, 2),
    }

# ==============================
# analysis/technicals.py
# ==============================
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD


def get_technicals(stock):
    df = stock.history(period="6mo")

    # defensive: return NaNs if insufficient history
    if df is None or df.empty or 'Close' not in df.columns or len(df['Close']) < 5:
        return {"RSI": np.nan, "MACD": np.nan}

    try:
        df["RSI"] = RSIIndicator(df["Close"]).rsi()
    except Exception:
        df["RSI"] = np.nan

    try:
        macd = MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
    except Exception:
        df["MACD"] = np.nan
        df["Signal"] = np.nan

    rsi_val = df["RSI"].dropna()
    macd_val = df["MACD"].dropna()

    rsi_out = float(rsi_val.iloc[-1]) if not rsi_val.empty else np.nan
    macd_out = float(macd_val.iloc[-1]) if not macd_val.empty else np.nan

    return {
        "RSI": round(rsi_out, 2) if not np.isnan(rsi_out) else np.nan,
        "MACD": round(macd_out, 2) if not np.isnan(macd_out) else np.nan,
    }
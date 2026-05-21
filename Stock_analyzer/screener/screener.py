# ==============================
# screener/screener.py
# ==============================
def screen_stocks(tickers, get_financial_ratios, get_technicals, get_stock):
    selected = []
    for t in tickers:
        stock = get_stock(t)
        ratios = get_financial_ratios(stock)
        tech = get_technicals(stock)

        if ratios["ROE"] and ratios["ROE"] > 15 and tech["RSI"] < 70:
            selected.append(t)

    return selected

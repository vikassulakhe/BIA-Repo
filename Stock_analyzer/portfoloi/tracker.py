# ==============================
# portfolio/tracker.py
# ==============================
def track_portfolio(portfolio, get_stock):
    result = []

    for ticker, buy_price in portfolio.items():
        stock = get_stock(ticker)
        current = stock.history(period="1d")["Close"].iloc[-1]

        change = ((current - buy_price)/buy_price)*100

        action = "HOLD"
        if change > 20:
            action = "BOOK PROFIT"
        elif change < -10:
            action = "STOP LOSS"

        result.append({
            "ticker": ticker,
            "return %": round(change,2),
            "action": action
        })

    return result
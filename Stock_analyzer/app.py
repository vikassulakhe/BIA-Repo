# ==============================
# app.py (Streamlit)
# ==============================
import streamlit as st
from data.fetch_data import get_stock
from analysis.fundamentals import get_financial_ratios
from analysis.growth import get_quarterly_growth
from analysis.technicals import get_technicals
from analysis.sentiment import get_sentiment, SENTIMENT_BACKEND
from ai.advisor import ai_advice
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

st.title("📊 AI Stock Analyzer")

ticker = st.text_input("Enter Stock (e.g. TCS.NS)")

if ticker:
    stock = get_stock(ticker)

    ratios = get_financial_ratios(stock)
    growth = get_quarterly_growth(stock)
    tech = get_technicals(stock)
    news = get_sentiment(ticker)

    # Friendly display for ratios and growth
    def _clean_val(v):
        try:
            if v is None:
                return 'N/A'
            if isinstance(v, float) and (pd.isna(v)):
                return 'N/A'
            return v
        except Exception:
            return 'N/A'

    st.write("**Fundamental Ratios**")
    ratios_df = pd.DataFrame(list(ratios.items()), columns=['Metric', 'Value']) if ratios else pd.DataFrame()
    ratios_df['Value'] = ratios_df['Value'].apply(_clean_val)
    st.table(ratios_df)

    st.write("**Recent Growth (quarterly avg)**")
    growth_df = pd.DataFrame(list(growth.items()), columns=['Metric', 'Value']) if growth else pd.DataFrame()
    growth_df['Value'] = growth_df['Value'].apply(_clean_val)
    st.table(growth_df)

    # Current price metric (last close) with delta vs previous close, currency, market time, and sparkline
    try:
        short_hist = stock.history(period='5d')
        if short_hist is None or short_hist.empty:
            raise ValueError('no short history')
        last_close = float(short_hist['Close'].iloc[-1])
        prev_close = float(short_hist['Close'].iloc[-2]) if len(short_hist) >= 2 else None

        # attempt to fetch currency from ticker info
        currency = None
        try:
            info = stock.info if hasattr(stock, 'info') else {}
            currency = info.get('currency')
        except Exception:
            currency = None

        # market time from last index
        try:
            last_dt = short_hist.index[-1]
            # format nicely
            market_time = last_dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            market_time = None

        col_left, col_right = st.columns([1, 1])
        with col_left:
            label = 'Current Price'
            if currency:
                label = f"{label} ({currency})"
            if prev_close is not None:
                delta = last_close - prev_close
                pct = (delta / prev_close) * 100 if prev_close != 0 else 0.0
                st.metric(label, f"{last_close:.2f}", delta=f"{pct:.2f}%")
            else:
                st.metric(label, f"{last_close:.2f}")
            if market_time:
                st.caption(f"Market time: {market_time}")
        with col_right:
            try:
                # small sparkline of recent closes
                st.line_chart(short_hist['Close'].tail(20))
            except Exception:
                pass
    except Exception:
        st.info('Current price not available')

    st.write("**Sentiment**")
    # Show metric with hover tooltip (using HTML title attribute)
    tooltip = "Mean polarity of recent news headlines about the ticker. Positive >0.05 bullish; Neutral -0.05..0.05; Negative < -0.05. Small sample — check headlines."
    col1, col2 = st.columns([1, 0.1])
    with col1:
        st.metric('News Sentiment (avg)', _clean_val(news.get('Sentiment')))
    with col2:
        st.markdown(f"<span title=\"{tooltip}\">ℹ️</span>", unsafe_allow_html=True)
    for h in news.get('Headlines', []):
        st.write('- ', h)

    with st.expander('About News Sentiment'):
        st.write('**Metric:** News Sentiment (avg) — the mean polarity score computed from recent news headlines about the ticker (approx -1 to +1).')
        st.write('**Scale & interpretation:**')
        st.write('- **Positive (> 0.05):** overall positive headlines — bullish tone.')
        st.write('- **Neutral (-0.05 to 0.05):** mixed or balanced coverage — no clear directional signal.')
        st.write('- **Negative (< -0.05):** overall negative headlines — bearish tone.')
        st.write('- **Magnitude:** larger absolute values imply stronger conviction; values near 0 indicate neutrality or low signal.')
        st.write('**Caveats:** small sample, headline noise, and timing; combine with fundamentals and technicals.')
        st.write(f'**Backend:** {SENTIMENT_BACKEND}')

    # Price and technical charts
    hist = stock.history(period='1y')
    if hist is None or hist.empty:
        st.info('No historical price data available for plotting.')
    else:
        st.write('**Price (1y)**')
        st.line_chart(hist['Close'])

        # compute RSI and MACD series for plotting
        try:
            rsi = RSIIndicator(hist['Close']).rsi()
            macd = MACD(hist['Close'])
            macd_series = macd.macd()
            signal = macd.macd_signal()
            tech_df = pd.DataFrame({'RSI': rsi, 'MACD': macd_series, 'Signal': signal})
            st.write('**Technical Indicators**')
            st.line_chart(tech_df.dropna())
        except Exception:
            st.info('Could not compute technical indicators for plotting.')

    if st.button("Get AI Recommendation"):
         result = ai_advice({**ratios, **growth, **tech, **news})
         st.write(result)

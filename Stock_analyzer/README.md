# Stock_analyzer

Short README describing the `Stock_analyzer` app, architecture, and flow.

**Purpose**
- Interactive per-ticker analysis and AI-assisted recommendation UI implemented with Streamlit.
- Accepts free-text company names or ticker symbols, auto-resolves symbols, fetches price and fundamentals, computes technicals and growth, scores news sentiment, and produces an AI recommendation.

**Quick start**
- Create and activate the project's venv (you already have `venv3.13`).
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run the Streamlit app:

```bash
source venv3.13/bin/activate
streamlit run Stock_analyzer/app.py
```

**Environment variables**
- `SENTIMENT_BACKEND` — overrides sentiment backend. Values: `textblob` (default), `vader`, or `auto`.
- OpenAI configuration (if using AI advisor): set `OPENAI_API_KEY` and related vars as required by your setup.

**Architecture (high level)**

- Input: free-text company name or ticker entered in UI
- Resolver: `data/fetch_data.py` maps name → Yahoo symbol (variants, .NS/.BO heuristics, Yahoo search)
- Data fetch: uses `yfinance` to fetch price history and financial tables
- Analysis modules:
  - `analysis/fundamentals.py` → ROE / ROCE (robust label matching)
  - `analysis/growth.py` → quarterly growth metrics
  - `analysis/technicals.py` → RSI, MACD and other indicators
  - `analysis/sentiment.py` → news headlines scraping + sentiment scoring (backend configurable)
- AI advisor: `ai/advisor.py` (builds prompt and calls OpenAI; cached results logged)
- UI: `app.py` (Streamlit) — shows ratios, growth, sentiment, price chart, technicals, and AI recommendation

**Mermaid flowchart**

```mermaid
flowchart TD
  A[User input: Name or Ticker] --> B[Resolver]
  B --> C[Resolved Symbol]
  C --> D[Data Fetch (yfinance)]
  D --> E1[Fundamentals]
  D --> E2[Growth]
  D --> E3[Technicals]
  D --> E4[Sentiment (news)]
  E1 --> F[AI Advisor]
  E2 --> F
  E3 --> F
  E4 --> F
  D --> UI[Streamlit UI]
  F --> UI
  UI --> User[User sees analysis & recommendation]
```

**Notes & implementation details**
- The resolver attaches a dynamic attribute `resolved_symbol` to returned `yfinance.Ticker` objects; the code uses safe casts to avoid type-checker warnings.
- Sentiment backend default is `textblob`; install `vaderSentiment` if you want VADER support. Set `SENTIMENT_BACKEND` to `vader` to use it.
- FinBERT was considered as a finance-tuned model but is optional due to model download/time and HF token requirements.
- AI calls are cached and audited to `ai_cache.json` / `ai_outputs.csv` to limit API usage.

**Troubleshooting**
- If a company name resolves but returns empty history, the data provider (Yahoo) may have no recent price or the symbol is different for that exchange — try adding `.NS`/`.BO` suffixes or the explicit ticker.
- If Streamlit warns about missing `watchdog`, install it for better file-watching performance: `pip install watchdog`.

**Next steps / enhancements**
- Add provider fallbacks (Alpha Vantage / Finnhub) for missing Yahoo data.
- Persist user preferences for sentiment backend in Streamlit session state.
- Add unit tests for label-matching and resolver heuristics.

---

File pointers:
- App: [Stock_analyzer/app.py](app.py)
- Resolver: [Stock_analyzer/data/fetch_data.py](data/fetch_data.py)
- Analysis modules: [Stock_analyzer/analysis](analysis/)
- AI advisor: [Stock_analyzer/ai/advisor.py](ai/advisor.py)

Feel free to ask me to expand any section or generate diagrams in other formats.

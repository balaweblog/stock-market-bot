# 📈 Stock Market AI Trading & Advisory Platform

An enterprise-grade, multi-agent AI quantitative research and advisory platform for Indian (`NSE`/`BSE`) and US equities, derivatives (Nifty Index Options), commodities (Gold/Silver), and mutual funds.

Built with Python, dynamic LLM fallback chains (Groq, Gemini, Mistral, Local Qwen2.5), deterministic market data engines (`yfinance`, AMFI, NSE Bhavcopy), and automated quantitative risk management.

---

## 📑 Table of Contents
1. [Overview & High-Level Application Flow](#-overview--high-level-application-flow)
2. [Detailed Application Flow by Advisor Pipeline](#-detailed-application-flow-by-advisor-pipeline)
3. [Architecture Diagrams](#-architecture-diagrams)
4. [Functional Understanding & Quantitative Mechanics](#-functional-understanding--quantitative-mechanics)
   - [A. Composite Quality Scoring & Conviction Engine](#a-composite-quality-scoring--conviction-engine)
   - [B. Defined-Risk Multi-Horizon Options Strategy](#b-defined-risk-multi-horizon-options-strategy)
   - [C. Market Regime & Trend Gating](#c-market-regime--trend-gating)
   - [D. Quota-Adaptive Multi-LLM Fallback Chain](#d-quota-adaptive-multi-llm-fallback-chain)
   - [E. Track Record Freeze & State Machine](#e-track-record-freeze--state-machine)
5. [Package Directory Layout](#-package-directory-layout)
6. [Complete Environment Variables Reference](#-complete-environment-variables-reference)
7. [Local Execution & Test Commands](#-local-execution--test-commands)

---

## 🌐 Overview & High-Level Application Flow

The platform operates as **5 autonomous advisory pipelines**, each engineered to generate institutional-grade market research, entry/exit signals, and risk-adjusted position sizing:

```
+-----------------------------------------------------------------------------------+
|                            AUTONOMOUS ADVISORY ENGINES                            |
+-------------------+--------------------+--------------------+--------------------+--------------------+
| 1. Stock          | 2. Nifty Stock     | 3. Swing Trade     | 4. Nifty Options   | 5. Mutual Fund     |
|    Predictor      |    News Advisor    |    Advisor         |    Strategy        |    Portfolio       |
+-------------------+--------------------+--------------------+--------------------+--------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                        STEP 1: DETERMINISTIC DATA FETCH                           |
|      (yfinance Quote/Hist, AMFI NAVs, NSE Option Chain/Bhavcopy, Commodity Spot)  |
+-----------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                   STEP 2: PRE-SCREENING & QUANTITATIVE RULES                      |
|      (YoY Revenue/Profit Growth >=15%, RSI <=70, D/E <=100%, 20-week SMA Gate)   |
+-----------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                 STEP 3: QUOTA-ADAPTIVE MULTI-LLM FALLBACK CHAIN                   |
| (Groq Llama-3.3-70B/Compound -> Gemini 2.5 Flash -> Mistral Web -> Local Qwen2.5)|
+-----------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                    STEP 4: RISK MANAGEMENT & POSITION SIZING                      |
|        (ATR Volatility Bands, Margin Caps, Expected Move & Sector Caps)           |
+-----------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                    STEP 5: HTML REPORT & EMAIL DELIVERY (SMTP)                    |
|      (Rich Dark Mode HTML, Compliance Disclaimers & Persistent State Update)      |
+-----------------------------------------------------------------------------------+
```

---

## 🔬 Detailed Application Flow by Advisor Pipeline

### 1. Stock Predictor & Equity Portfolio Review (`controllers/stock_controller.py`)
- **Data Ingestion**: Downloads 1-year price history via `yfinance`, fundamentals (P/E, Debt/Equity, ROE, Profit Margins), and upcoming corporate events (earnings, dividends).
- **Commodity Tracking**: Calculates retail gold/silver buy levels based on 52-week ranges and monthly SIP budget caps.
- **LLM Reasoning**: Passes raw fundamentals, news, and technicals to the LLM to produce structured qualitative summaries and risk factors.
- **Scoring & Decision**: Computes composite fundamental/technical score ($0-100$), classifies conviction (`STRONG BUY`, `BUY / HOLD`, `HOLD`, `SELL`), and calculates exact entry zones and ATR stop losses.
- **Track Record Update**: Freezes new recommendations into `run_history.json` and tracks outcome hit-rates across subsequent runs.

### 2. Nifty Watchlist News & Macro Advisor (`controllers/nifty_stock_controller.py`)
- **4-Stage Synthesis**:
  - **Stage 1 (Market/Macro)**: Synthesizes global indices, RBI policy, crude oil, and USD/INR movement.
  - **Stage 2 (Stock Batching)**: Analyzes Nifty 50 watchlist stocks in parallel chunks with live search grounding.
  - **Stage 3 (Sector Trends)**: Summarizes sector rotation (IT, Banking, Pharma, Auto, Energy).
  - **Stage 4 (Executive Synthesis)**: Blends macro, stock, and sector insights into a cohesive executive digest.

### 3. Multi-Stage Swing Trade Pipeline (`controllers/swing_controller.py`)
- **Universe Screen**: Screens a predefined universe of liquid Indian stocks across high-growth sectors.
- **Deterministic Gate**: Pre-filters candidates on Python level before LLM invocation:
  $$\text{YoY Revenue Growth} \ge 15\%, \quad \text{YoY Profit Growth} \ge 15\%, \quad \text{RSI} \le 70, \quad \frac{\text{Debt}}{\text{Equity}} \le 1.0$$
- **Regime Gate**: Verifies Nifty 50 is trading above its 20-week SMA before allowing any bullish breakout setup.
- **LLM Setup Identification**: Identifies pattern setups (VCP, Cup & Handle, Bull Flag) and exact trigger levels.
- **Quality Score Gating**: Calculates composite Quality Score ($0-100$). Rejects setups below $60/100$ threshold or returns explicit "No qualifying trade found".

### 4. Nifty Options Strategy Advisor (`controllers/option_controller.py`)
- **Live Feed Fetch**: Fetches live Nifty spot price, India VIX, FII/DII flow data, and NSE Option Chain / Bhavcopy.
- **Expected Move Calculation**: Calculates 1-SD Expected Move for 3 distinct horizons:
  $$\text{EM} = \text{Spot} \times \text{IV} \times \sqrt{\frac{\text{DTE}}{365}}$$
- **Defined-Risk Constraint**: Mandates 100% defined-risk structures only (`Bull Call Spread`, `Bear Call Spread`, `Bull Put Spread`, `Bear Put Spread`, `Iron Condor`, `Iron Butterfly`). Naked options and ratio spreads are strictly rejected.
- **Multi-Horizon Strategy Normalization**: Renders primary and alternative trade setups for **Weekly**, **Next Week**, and **Monthly** expiries.

### 5. Mutual Fund Portfolio Review Pipeline (`controllers/mutual_fund_controller.py`)
- **AMFI Integration**: Fetches official scheme NAVs and 1Y/3Y/5Y rolling returns directly from AMFI (Association of Mutual Funds in India).
- **Portfolio Tracking**: Evaluates equity, hybrid, and debt mutual funds against market benchmark indices.
- **LLM Portfolio Review**: Synthesizes fund manager shifts, portfolio turnover, sector overweights, and actionable `Hold` / `Rebalance` advice.

---

## 🏗️ Architecture Diagrams

### System Architecture Diagram
```
                     +-----------------------------------+
                     |      USER / GITHUB ACTIONS        |
                     +-----------------+-----------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
     +---------------------------+           +---------------------------+
     |   controllers/ Package    |           |      tests/ Package       |
     | (Execution Entrypoints)   |           | (24 Unit Tests - 100% OK) |
     +-------------+-------------+           +---------------------------+
                   |
     +-------------+---------------------------------------+
     |             |                   |                   |
     v             v                   v                   v
+----------+ +-----------+     +---------------+    +--------------+
| services | |    llm    |     |    models     |    |    utils     |
| Data     | | Fallback  |     | Financial &   |    | Config, Log  |
| Fetchers | | Engine    |     | Quant Logic   |    | & Compliance |
+----------+ +-----------+     +---------------+    +--------------+
```

### LLM Quota-Adaptive Fallback Sequence
```
+---------------+      Quota OK?     +-------------------+
|  Groq Llama3  |------------------->|  Execute & Return |
+-------+-------+                    +-------------------+
        | Exceeded / Error
        v
+---------------+      Quota OK?     +-------------------+
|  Gemini Flash |------------------->|  Execute & Return |
+-------+-------+                    +-------------------+
        | Exceeded / Error
        v
+---------------+      Quota OK?     +-------------------+
|  Mistral Web  |------------------->|  Execute & Return |
+-------+-------+                    +-------------------+
        | Exceeded / Error
        v
+---------------+                    +-------------------+
|  Local Qwen   |------------------->|  Execute & Return |
+---------------+                    +-------------------+
```

---

## 🧠 Functional Understanding & Quantitative Mechanics

### A. Composite Quality Scoring & Conviction Engine
The system evaluates trade candidates using a weighted composite score ($0-100$):

$$\text{Score} = w_{\text{EV}} \cdot \text{EV} + w_{\text{RR}} \cdot \text{RR} + w_{\text{POP}} \cdot \text{POP} + w_{\text{Conf}} \cdot \text{Conf} + w_{\text{Liq}} \cdot \text{Liq} + w_{\text{OI}} \cdot \text{OI}$$

- **Expected Value (30%)**: Normalized probability-weighted net payout.
- **Reward : Risk (20%)**: Target gain divided by maximum stop-loss risk.
- **Probability of Profit (15%)**: Delta/IV-derived probability of expiring in-the-money.
- **Confidence (15%)**: Technical pattern clarity & alignment across multiple timeframes.
- **Liquidity & OI Alignment (20%)**: Option open interest build-up and bid-ask spread tightness.

### B. Defined-Risk Multi-Horizon Options Strategy
For Nifty option spreads, maximum loss and required margin are capped deterministically:
- **Bull Put Spread Max Loss**:
  $$\text{Max Loss} = (\text{Strike}_{\text{Short}} - \text{Strike}_{\text{Long}} - \text{Net Credit}) \times \text{Lot Size}$$
- **Max Capital Cap**: Per-horizon allocation is capped at $5\%$ of total capital, and aggregate across all horizons is capped at $15\%$.

### C. Market Regime & Trend Gating
To eliminate buying breakouts in a crashing market, the system computes the 20-week SMA of Nifty 50:
$$\text{Regime} = \begin{cases} \text{Bullish (Gated Open)}, & \text{Nifty Close} > 20\text{-week SMA and } \frac{d}{dt}(SMA20) > 0 \\ \text{Caution / Filtered}, & \text{Otherwise} \end{cases}$$

### D. Quota-Adaptive Multi-LLM Fallback Chain
The `llm/llm_backend.py` module persists rate-limit counters to `quota_cache.json`. When Groq returns HTTP 429 (Rate Limit Exceeded), it immediately toggles execution to Gemini Flash without failing the job.

### E. Track Record Freeze & State Machine
When a stock receives a `STRONG BUY` or `BUY / HOLD` recommendation, its entry price, target price, and stop loss are **frozen** in `run_history.json`. Subsequent daily runs compare live prices against frozen boundaries to track real win-rate percentage.

---

## 📁 Package Directory Layout

```
stockcode/
├── controllers/              # Pipeline Controllers
│   ├── stock_controller.py
│   ├── nifty_stock_controller.py
│   ├── swing_controller.py
│   ├── option_controller.py
│   └── mutual_fund_controller.py
├── services/                 # Data Fetchers
│   ├── stock_fetcher.py
│   ├── commodity_tracker.py
│   └── news_engine.py
├── llm/                      # AI Engine & Sentiment
│   ├── llm_backend.py
│   └── sentiment_score.py
├── models/                   # Financial & Quant Models
│   ├── fundamentals.py
│   ├── advanced_fundamentals.py
│   ├── recommendation_logic.py
│   ├── position_sizing.py
│   ├── support_resistance.py
│   ├── market_context.py
│   ├── scorer.py
│   ├── swing_trade_scoring.py
│   ├── swing_trade_risk.py
│   ├── swing_trade_regime.py
│   ├── swing_trade_universe.py
│   ├── swing_trade_outcomes.py
│   ├── swing_trade_backtest.py
│   └── track_record.py
├── utils/                    # Utilities & Config
│   ├── config.py
│   ├── constants.py
│   ├── logger.py
│   ├── compliance.py
│   └── analyze_rejection_history.py
├── tests/                    # Unit Tests
│   └── test_*.py
├── .gitignore
├── README.md
├── requirements.txt
└── run_history.json
```

---

## 🔑 Complete Environment Variables Reference

### 1. API Keys & AI Provider Secrets
| Variable Name | Type | Required? | Default | Description |
|---------------|------|-----------|---------|-------------|
| `GROQ_API_KEY` | String | Highly Recommended | Unset | Groq API Key for fast synthesis & autonomous search (`groq/compound`). |
| `GOOGLE_API_KEY` | String | Recommended | Unset | Google Gemini API Key for Gemini Flash + Google Search grounding fallback. |
| `TAVILY_API_KEY` | String | Optional | Unset | Tavily Web Search API Key for web context gathering. |
| `MISTRAL_API_KEY` | String | Optional | Unset | Mistral AI API Key for Mistral `web_search` agent fallback. |
| `NEWS_API_KEY` | String | Optional | Unset | NewsAPI Key for financial headline search (falls back to Google News RSS). |

### 2. Email Delivery Configuration
| Variable Name | Type | Required? | Default | Description |
|---------------|------|-----------|---------|-------------|
| `EMAIL_FROM` | String | For Emailing | Unset | Gmail sender address (e.g. `sender@gmail.com`). |
| `EMAIL_PASSWORD` | String | For Emailing | Unset | Gmail App Password (16 characters). |
| `EMAIL_TO` | String | For Emailing | Unset | Comma-separated list of primary recipient emails. |
| `EMAIL_CC` | String | Optional | Unset | Comma-separated list of CC recipient emails. |

### 3. Global Execution & Data Controls
| Variable Name | Type | Default | Description |
|---------------|------|---------|-------------|
| `DRY_RUN` | Boolean | `false` | When `true`, writes HTML report files locally instead of sending emails. |
| `REQUIRE_LIVE_DATA` | Boolean | `true` | When `true`, aborts run if live market feeds or search cannot be verified. |
| `RUN_HISTORY_PATH` | String | `run_history.json` | Path to persistent track record JSON state file. |

### 4. Options Strategy Configuration (`controllers/option_controller.py`)
| Variable Name | Type | Default | Description |
|---------------|------|---------|-------------|
| `OPTIONS_TOTAL_CAPITAL_INR` | Float | `1000000` | Total portfolio capital in INR (₹1,000,000 default). |
| `NIFTY_LOT_SIZE` | Integer | `75` | Nifty 50 option contract lot size. |
| `OPTIONS_PER_HORIZON_CAP_PCT` | Float | `5.0` | Maximum risk capital per horizon expiry ($5\%$). |
| `OPTIONS_AGGREGATE_CAP_PCT` | Float | `15.0` | Maximum aggregate risk capital across all horizons ($15\%$). |
| `OPTIONS_MAX_LOTS_PER_HORIZON` | Integer | `5` | Maximum number of option lots per strategy. |
| `OPTIONS_MIN_REWARD_RISK_RATIO` | Float | `0.5` | Minimum acceptable Reward-to-Risk ratio. |
| `OPTIONS_MIN_CREDIT_WIDTH_PCT` | Float | `15.0` | Minimum net credit as % of strike width ($15\%$). |
| `OPTIONS_CONSIDER_QUALITY_THRESHOLD` | Float | `75.0` | Quality Score threshold (0-100) to mark trade as "Consider". |
| `OPTIONS_REJECT_IC_SHORT_INSIDE_EM` | Boolean | `true` | Rejects Iron Condors whose short strikes fall inside expected move. |
| `OPTIONS_EM_DIVERGENCE_THRESHOLD_PCT` | Float | `25.0` | Max allowed divergence % between Black-Scholes and VIX expected move. |
| `OPTIONS_MAX_LEG_SPREAD_PCT` | Float | `15.0` | Max allowed Bid-Ask spread % on individual legs. |
| `OPTIONS_RISK_FREE_RATE` | Float | `0.065` | Annualized risk-free interest rate ($6.5\%$). |
| `OPTIONS_DIVIDEND_YIELD` | Float | `0.012` | Nifty 50 dividend yield ($1.2\%$). |
| `OPTIONS_MARGIN_MULTIPLIER` | Float | `1.2` | Defined-risk margin requirement buffer multiplier ($120\%$). |

### 5. Swing Trade Strategy Knobs (`controllers/swing_controller.py`)
| Variable Name | Type | Default | Description |
|---------------|------|---------|-------------|
| `USE_DETERMINISTIC_SCREEN` | Boolean | `true` | Enables zero-token Python/`yfinance` growth screening before LLM Stage 2. |
| `MIN_GROWTH_YOY_PCT` | Float | `15.0` | Minimum YoY revenue or profit growth % required. |
| `MIN_RISK_REWARD` | Float | `1.5` | Minimum risk-to-reward ratio for trade entries. |
| `MAX_RSI_OVERBOUGHT` | Float | `70.0` | Maximum weekly RSI limit to prevent buying overbought peaks. |
| `MAX_DEBT_TO_EQUITY_PCT` | Float | `100.0` | Maximum Debt-to-Equity % limit ($1.0$ ratio). |
| `MIN_ROE_PCT` | Float | `10.0` | Minimum Return on Equity % required. |
| `REQUIRE_UPTREND_FILTER` | Boolean | `true` | Enforces price above 50-week SMA uptrend requirement. |
| `REQUIRE_QUALIFYING_STOCK` | Boolean | `true` | When `true`, reports "No qualifying trade found" rather than returning weak picks. |
| `REQUIRE_PROFESSIONAL_QUALITY_GATE` | Boolean | `true` | Enforces composite quality score minimum gate ($\ge 60.0$). |
| `REGIME_SOFTEN_GROWTH_BAR` | Boolean | `true` | Temporarily relaxes growth bar from 20% to 15% in strong bull regimes. |
| `AUTO_ADJUST_THRESHOLDS` | Boolean | `false` | Enables adaptive threshold relaxation based on historical near-miss logs. |
| `MIN_RUNS_BEFORE_ADJUST` | Integer | `4` | Minimum distinct runs required before threshold auto-adjustment triggers. |

---

## 🚀 Local Execution & Test Commands

### 1. Execute Stock Predictor & Portfolio Review
```bash
python -m controllers.stock_controller
```

### 2. Execute Multi-Horizon Nifty Option Strategy
```bash
python -m controllers.option_controller
```

### 3. Execute Swing Trade Advisor
```bash
python -m controllers.swing_controller
```

### 4. Execute Mutual Fund Portfolio Advisor
```bash
python -m controllers.mutual_fund_controller
```

### 5. Execute Nifty Stock Market News Advisor
```bash
python -m controllers.nifty_stock_controller
```

### 6. Run Unit Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 7. Run Rejection History Analytics
```bash
python utils/analyze_rejection_history.py
```

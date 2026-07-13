import os
import yfinance as yf
import pandas as pd
import ta
import smtplib
import traceback
from email.mime.text import MIMEText
from datetime import datetime
from zoneinfo import ZoneInfo
try:
    from transformers import pipeline
except ImportError:
    pipeline = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from config import *
from stock_fetcher import fetch_fundamentals
from fundamentals import score_fundamentals
from advanced_fundamentals import fetch_advanced_fundamentals, score_advanced_fundamentals
from market_context import build_market_context, get_resilient_session
from news_engine import get_news
from sentiment_score import score_headlines
from scorer import final_score, decision
from position_sizing import apply_risk_management
from recommendation_logic import choose_stock_entry
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger import log
from commodity_tracker import CommodityTracker

import threading

model_lock = threading.Lock()

llm_pipeline = None
use_gemini_flash = False

def init_llm_generator():
    """
    Initializes the AI model.
    Prioritizes Gemini 1.5 Flash (Free Cloud Tier) if GOOGLE_API_KEY is present.
    Otherwise, falls back to the ultra-lightweight and highly smart Qwen2.5-1.5B local model.
    """
    global llm_pipeline, use_gemini_flash
    
    # 1. Check for Gemini API key
    api_key = os.getenv("GOOGLE_API_KEY") or globals().get("GOOGLE_API_KEY")
    if api_key and genai is not None:
        try:
            log.info("Google API key detected. Initializing Gemini 1.5 Flash (Free Cloud Tier)...")
            genai.configure(api_key=api_key)
            use_gemini_flash = True
            log.info("Gemini 1.5 Flash initialized successfully.")
            return "gemini"
        except Exception as exc:
            log.warning(f"Failed to initialize Gemini, falling back to local model: {exc}")
            use_gemini_flash = False

    # 2. Fallback to local model
    if pipeline is None:
        log.warning("The 'transformers' library is not installed. LLM reasoning will be disabled.")
        return None
        
    if llm_pipeline is None:
        try:
            import torch
            device = -1
            torch_dtype = torch.float32
            
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                torch_dtype = torch.float16
                log.info("Apple Silicon GPU (MPS) detected. Enabling hardware acceleration.")
            elif torch.cuda.is_available():
                device = 0
                torch_dtype = torch.float16
                log.info("Nvidia GPU (CUDA) detected. Enabling hardware acceleration.")
            else:
                log.info("No compatible GPU detected. Running model on CPU.")

            log.info("Initializing high-quality local AI model (Qwen2.5-1.5B-Instruct)...")
            # Using Qwen2.5-1.5B-Instruct: extremely smart, fast on MPS/CPU, vastly superior reasoning
            llm_pipeline = pipeline(
                "text-generation", 
                model="Qwen/Qwen2.5-1.5B-Instruct",
                device=device,
                torch_dtype=torch_dtype
            )
            log.info("Local AI model initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize local AI model: {e}")
            llm_pipeline = None
            
    return "local" if llm_pipeline is not None else None


def generate_gemini_reasoning(prompt):
    """
    Generates text using the Gemini 1.5 Flash model.
    """
    if genai is None:
        return None
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        log.error(f"Gemini generation failed: {e}")
        return None


def generate_hf_reasoning(prompt, stock_name=""):
    """
    Generates text using the locally run Hugging Face model.
    Serializes execution using a thread lock to prevent CPU/GPU thrashing.
    """
    if llm_pipeline is None:
        return None

    try:
        # Format the prompt for Qwen2.5 Chat template (uses ChatML template)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = llm_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Inference must be serialized to prevent resource thrashing
        with model_lock:
            log.info(f"Generating AI trade plan for {stock_name}...")
            # Allow a longer, more detailed generation for richer trade analysis
            outputs = llm_pipeline(formatted_prompt, max_new_tokens=400, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            log.info(f"AI trade plan successfully generated for {stock_name}.")
        
        generated_text = outputs[0]['generated_text']
        
        # Clean up the output to get only the assistant's response (Qwen2.5 style)
        response = generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        
        return response
    except Exception as e:
        log.error(f"Hugging Face model generation failed for {stock_name}: {e}")
        return None
def fetch_data(symbol):
    df = yf.download(
        symbol,
        period="300d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        session=get_resilient_session()
    )

    if df.empty:
        raise Exception(f"No data for {symbol}")

    df.reset_index(inplace=True)

    # flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        cols = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    else:
        cols = list(df.columns)

    normalized = []
    for col in cols:
        name = str(col).lower().strip()
        name = name.replace(" ", "").replace("-", "").replace("_", "")
        normalized.append(name)

    df.columns = normalized

    if "close" not in df.columns:
        source_close = None
        for candidate in df.columns:
            if "close" in candidate:
                source_close = candidate
                break
        if source_close is not None:
            df["close"] = df[source_close]

    if "close" not in df.columns:
        raise Exception(
            f"Missing required 'close' column after normalization for {symbol}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    return df


# -----------------------------
# Indicator calculations
# -----------------------------
def calculate_indicators(df):
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    rsi_indicator = ta.momentum.RSIIndicator(
        close=df["close"],
        window=14
    )
    df["rsi"] = rsi_indicator.rsi()

    df["vol_avg"] = df["volume"].rolling(20).mean()

    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    ).adx()

    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    ).average_true_range()

    df["high20"] = df["high"].rolling(20).max()
    df["low20"] = df["low"].rolling(20).min()

    return df


# -----------------------------
# Scoring engine
# -----------------------------
def calculate_score(df):
    latest = df.iloc[-1]
    score = 0
    reason = []

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema100 = latest["ema100"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]
    macd_hist = latest["macd_hist"]
    adx = latest["adx"]

    if price > ema200:
        score += 20
        reason.append("Above EMA200")
    if price > ema100:
        score += 15
        reason.append("Above EMA100")
    if price > ema50:
        score += 10
        reason.append("Above EMA50")
    if price > ema20:
        score += 10
        reason.append("Above EMA20")

    if ema20 > ema50 > ema100 > ema200:
        score += 20
        reason.append("Strong trend structure")
    elif ema50 > ema100 > ema200:
        score += 10
        reason.append("Uptrend confirmed")
    elif ema20 > ema50 and ema50 > ema200:
        score += 5
        reason.append("Bullish alignment")

    if rsi <= 60:
        score += 20
        if rsi < 40:
            reason.append("RSI low (swing entry zone)")
        else:
            reason.append("RSI healthy")
    elif rsi <= 70:
        score += 10
        reason.append("RSI moderate")
    else:
        score -= 5
        reason.append("RSI high")

    if macd > macd_signal:
        score += 10
        reason.append("MACD bullish")
        if macd_hist > 0:
            score += 5
            reason.append("MACD momentum rising")
    else:
        reason.append("MACD bearish")

    if adx >= 25:
        score += 10
        reason.append("ADX strong trend")
    elif adx >= 20:
        score += 5
        reason.append("ADX trend developing")
    else:
        reason.append("ADX weak")

    if latest["volume"] > latest["vol_avg"]:
        score += 10
        reason.append("Volume strong")

    if latest["high20"] and price >= latest["high20"] * 0.98:
        score += 5
        reason.append("Near 20-day breakout")
    elif latest["low20"] and price < latest["low20"] * 1.02 and price > ema20:
        score += 3
        reason.append("Healthy pullback")

    return score, reason


    # -----------------------------
    # Signal

def get_signal(score):
    if score >= 80:
        return "GREEN -> BUY / ADD"
    elif score >= 50:
        return "YELLOW -> HOLD"
    else:
        return "RED -> SELL / REDUCE"


def calculate_combined_score(technical, fundamentals, sentiment, adv_fundamentals, market_context):
    # Use the existing final_score weighting and fold advanced fundamentals into total score.
    combined_fund = fundamentals + (adv_fundamentals * 0.4)
    total = final_score(technical, combined_fund, sentiment)

    trend_text = str(market_context.get("trend", "")).lower()
    if "up" in trend_text or "positive" in trend_text:
        total += 2
    elif "down" in trend_text or "negative" in trend_text:
        total -= 2

    return round(max(0, min(100, total)), 2)


def get_recommended_entry(signal, total_score, latest, market_context, entry_context):
    return choose_stock_entry(signal, total_score, latest, market_context, entry_context)


def generate_llm_reasoning(stock_name, ticker, latest, tech_score, fund_score, sentiment_score, sentiment_label, tech_signal, signal, headlines, risk_data, entry_context=None):
    # Fallback summary generation
    def _get_fallback_reason():
        headline_summary = headlines[:2]
        headline_text = "; ".join(headline_summary) if headline_summary else "No recent headlines."
        return (
            f"{stock_name} is currently in a {signal.lower()} posture. "
            f"Technical momentum is {tech_signal.lower()} with a score of {tech_score}, "
            f"fundamentals are {fund_score}, and sentiment is {sentiment_label.lower()} ({round(sentiment_score, 2)}). "
            f"Recent headlines: {headline_text}."
        )

    entry_context = entry_context or {}
    price_vs_ema20 = entry_context.get("price_vs_ema20_pct", "n/a")
    price_vs_ema50 = entry_context.get("price_vs_ema50_pct", "n/a")
    volume_vs_avg = entry_context.get("volume_vs_avg_pct", "n/a")
    risk_reward = entry_context.get("risk_reward_ratio", "n/a")
    current_price = entry_context.get("current_price", round(latest['close'], 2))

    # High-quality prompt for a professional swing-trading analysis
    prompt = (
        f"Act as a veteran swing trader managing real capital. Deliver a high-conviction trade plan for {stock_name} ({ticker}) focused on the next 1-10 trading sessions. "
        f"Be direct, professional, and actionable, and treat this as a real decision-making brief rather than a generic summary.\n\n"
        f"**Trade Context:**\n"
        f"- Overall Signal: {signal}\n"
        f"- Current Market Price: {current_price}\n"
        f"- Technical Score: {tech_score}\n"
        f"- Key Price Levels: EMA20: {round(latest['ema20'], 2)}, EMA50: {round(latest['ema50'], 2)}, EMA200: {round(latest['ema200'], 2)}\n"
        f"- Price vs EMA20 / EMA50: {price_vs_ema20} / {price_vs_ema50}\n"
        f"- Volume vs 20D Avg: {volume_vs_avg}\n"
        f"- Risk/Reward Ratio: {risk_reward}\n"
        f"- Momentum: RSI at {round(latest['rsi'], 2)}; ADX at {round(latest['adx'], 2)}\n"
        f"- Sentiment: {sentiment_label} ({round(sentiment_score, 2)})\n\n"
        f"**Pre-calculated Risk Levels:**\n"
        f"- Patient Entry: {risk_data['buy_levels']['patient_entry']} (best for a pullback or discount entry)\n"
        f"- Optimal Entry: {risk_data['buy_levels']['optimal_entry']} (best for a retest or confirmation entry)\n"
        f"- Aggressive Entry: {risk_data['buy_levels']['aggressive_entry']} (only for breakout momentum or a strong trend continuation)\n"
        f"- Stop-Loss: {risk_data['stop_loss']}\n"
        f"- Profit Target: {risk_data['target']}\n\n"
        f"**Instructions for a Pro Swing Trader (Detailed Analysis Requested):**\n"
        f"1. **Trade Thesis:** One clear sentence stating the setup, directional bias, and expected move.\n"
        f"2. **Entry Plan:** Recommend Patient, Optimal, or Aggressive style and give exact price levels for each, with rationale.\n"
        f"3. **Positioning & Sizing:** Recommend position sizing guidance (scale-in points, % of portfolio, full size vs partial) and explain rationale.\n"
        f"4. **Risk Management:** State stop-loss, invalidation point, and calculate risk per share and % risk vs suggested position size.\n"
        f"5. **Exit Plan & Targets:** Provide at least two profit target levels and what market events or indicator changes would invalidate them.\n"
        f"6. **Scenario Analysis:** Provide 2-3 short scenarios (Bull case, Base case, Bear case) with approximate probabilities and numeric price ranges.\n"
        f"7. **Confirmation Signals & Timing:** Name specific near-term triggers (e.g., retest of EMA20, breakout volume) and an expected time horizon (1-3 sessions, 3-10 sessions).\n"
        f"8. **Supporting Evidence:** Cite 3 concrete data points from the provided inputs (e.g., RSI value, EMA alignment, top headline) that justify the recommendation.\n"
        f"9. **Format:** Output with clear labeled sections and short bullet points. Use numeric values where possible and avoid generic filler."
    )

    generator = init_llm_generator()
    if generator is None:
        return _get_fallback_reason()

    # Use the Hugging Face model for reasoning
    hf_text = generate_hf_reasoning(prompt, stock_name=stock_name)
    if hf_text:
        return hf_text

    # Fallback if the AI model fails for any reason
    return _get_fallback_reason()


def truncate_text(text, limit=350):
    if not text:
        return text
    if len(text) <= limit:
        return text
    # Prefer to cut at sentence boundary
    cut = text[:limit]
    last_period = cut.rfind('. ')
    if last_period != -1 and last_period > limit - 100:
        return cut[:last_period + 1] + '...'
    return cut.rstrip() + '...'


def get_date_with_suffix(d):
    day = d.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    return d.strftime(f"%#d{suffix} %B %Y" if os.name == 'nt' else f"%-d{suffix} %B %Y")


# -----------------------------
# Email
# -----------------------------
def send_email(report_html, mode):
    if not all([EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO]):
        print(
            "Email credentials not found. "
            "Please set EMAIL_FROM, EMAIL_PASSWORD, and EMAIL_TO environment variables."
        )
        return

    to_recipients = parse_email_list(EMAIL_TO)
    cc_recipients = parse_email_list(EMAIL_CC)

    if not to_recipients:
        print("No valid TO recipients found. Please set EMAIL_TO with a comma-separated list of emails.")
        return

    subject = " BlueOcean Stock Report"
    if mode == "real_time":
        subject = f"REAL-TIME: {subject}"
    elif mode == "eod":
        subject = f"DAILY: {subject}"

    # Append the formatted date and current time in IST
    now_utc = datetime.now(ZoneInfo("UTC"))
    now_ist = now_utc.astimezone(ZoneInfo("Asia/Kolkata"))
    formatted_date = get_date_with_suffix(now_ist)
    formatted_time = now_ist.strftime("%I:%M %p")
    subject += f" - {formatted_date} {formatted_time} IST"

    msg = MIMEText(report_html, "html")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, all_recipients, msg.as_string())
        server.quit()
    except smtplib.SMTPAuthenticationError:
        print(
            "SMTP Authentication Error: The username or password you entered is not correct. "
            "Please check your credentials and App Password if using Gmail."
        )
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")
        traceback.print_exc()


def process_stock(stock_name, ticker, use_llm=True, detailed_llm=False):
    try:
        # fetch data and compute scores
        df = fetch_data(ticker)
        df = calculate_indicators(df)

        # consolidated technical score calculation
        tech_score, reasons = calculate_score(df)
        tech_signal = get_signal(tech_score)

        fund_raw = fetch_fundamentals(ticker)
        fund_score = score_fundamentals(fund_raw)
        upcoming_events = fund_raw.get("upcomingEvents", {})

        adv_raw = fetch_advanced_fundamentals(ticker)
        adv_fund_score = score_advanced_fundamentals(adv_raw)

        headlines = get_news(stock_name)
        sentiment = score_headlines(headlines)
        sentiment_score = sentiment.get("score", 50.0)
        sentiment_label = sentiment.get("label", "Neutral")

        try:
            market_context = build_market_context(ticker)
        except Exception as exc:
            print(f"Market context failed for {ticker}: {exc}")
            market_context = {
                "trend": "unknown",
                "return_20d": 0.0,
                "return_50d": 0.0,
                "sector": "unknown",
                "industry": "unknown",
            }
        total_score = calculate_combined_score(
            tech_score,
            fund_score,
            sentiment_score,
            adv_fund_score,
            market_context,
        )

        signal = decision(total_score)

        latest = df.iloc[-1]
        prev_close = df.iloc[-2]["close"] if len(df) >= 2 else None
        prev_close_change_pct = None
        if prev_close is not None and prev_close != 0:
            prev_close_change_pct = round(((latest["close"] - prev_close) / prev_close) * 100, 2)
        prev_close_change_color = "#16a34a" if prev_close_change_pct is not None and prev_close_change_pct >= 0 else "#dc2626"
        prev_close_change_text = f"{prev_close_change_pct:+.2f}%" if prev_close_change_pct is not None else "n/a"
        news_text = ", ".join(headlines[:3])
        
        # Get risk management data
        risk_data = apply_risk_management(signal, total_score, cash=100000, price=latest["close"])

        entry_context = {
            "current_price": round(latest["close"], 2),
            "price_vs_ema20_pct": round(((latest["close"] - latest["ema20"]) / latest["ema20"]) * 100, 2) if pd.notna(latest["ema20"]) and latest["ema20"] else None,
            "price_vs_ema50_pct": round(((latest["close"] - latest["ema50"]) / latest["ema50"]) * 100, 2) if pd.notna(latest["ema50"]) and latest["ema50"] else None,
            "volume_vs_avg_pct": round(((latest["volume"] - latest["vol_avg"]) / latest["vol_avg"]) * 100, 2) if pd.notna(latest["vol_avg"]) and latest["vol_avg"] else None,
            "risk_reward_ratio": round((risk_data["target"] - latest["close"]) / max(latest["close"] - risk_data["stop_loss"], 1e-6), 2) if latest["close"] > risk_data["stop_loss"] else None,
        }

        recommended_entry = get_recommended_entry(signal, total_score, latest, market_context, entry_context)
        risk_data["recommended_entry"] = recommended_entry
        risk_data["recommended_entry_label"] = recommended_entry.replace("_", " ").title()
        risk_data["recommended_buy_level"] = risk_data["buy_levels"][recommended_entry]

        llm_reason = generate_llm_reasoning(
            stock_name,
            ticker,
            latest,
            tech_score,
            fund_score,
            sentiment_score,
            sentiment_label,
            tech_signal,
            signal,
            headlines,
            risk_data,
            entry_context,
        )

        # If user requested concise output, truncate the LLM reasoning server-side
        llm_display = llm_reason
        if llm_reason and not detailed_llm:
            llm_display = truncate_text(llm_reason, limit=350)

        if "sell" in signal.lower():
            priority = 3
        elif "hold" in signal.lower() or "buy / hold" in signal.lower():
            priority = 2
        else:
            priority = 1

        # card-style HTML for each stock (email-safe)
        row_html = f"""
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 0;border-radius:12px;background:#ffffff;border:1px solid #e5e7eb;">
            <tr>
                <td style="padding:14px;">
                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                        <tr>
                            <td style="vertical-align:top;">
                                <h3 style="margin:0;font-size:16px;color:#0f172a;line-height:1.2;">{stock_name} <span style="font-size:13px;color:#64748b;">{ticker}</span></h3>
                                <div style="margin:6px 0 0;font-size:13px;color:#334155;line-height:1.4;max-height:140px;overflow:hidden;">{llm_display}</div>
                            </td>
                            <td style="width:110px;text-align:right;vertical-align:top;">
                                <div style="display:inline-block;padding:6px 10px;border-radius:999px;font-weight:700;color:#fff;background:{'#dc2626' if 'sell' in signal.lower() else '#f59e0b' if 'hold' in signal.lower() else '#047857'};">{signal}</div>
                                <div style="margin-top:8px;font-size:13px;color:#334155;">Score: <strong>{total_score}</strong></div>
                            </td>
                        </tr>
                    </table>
                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;margin-top:10px;font-size:13px;color:#475569;">
                        <tr>
                            <td style="padding:6px 0;width:50%;"><strong>Current Price</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['close'],2)}</div></td>
                            <td style="padding:6px 0;width:50%;"><strong>EMA20 / EMA50</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['ema20'],2)} / {round(latest['ema50'],2)}</div></td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0;"><strong>EMA100 / EMA200</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['ema100'],2)} / {round(latest['ema200'],2)}</div></td>
                            <td style="padding:6px 0;"><strong>RSI / ADX</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['rsi'],2)} / {round(latest['adx'],2)}</div></td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0;"><strong>Prev Close</strong><div style="color:#0f172a;margin-top:4px;">{prev_close if prev_close is not None else 'n/a'}</div></td>
                            <td style="padding:6px 0;"><strong>Change vs Prev</strong><div style="color:{prev_close_change_color};margin-top:4px;">{prev_close_change_text}</div></td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0;"><strong>Entry Edge</strong><div style="color:#0f172a;margin-top:4px;">{entry_context['price_vs_ema20_pct']:+.1f}% vs EMA20 / {entry_context['price_vs_ema50_pct']:+.1f}% vs EMA50</div></td>
                            <td style="padding:6px 0;"><strong>Volume / RR</strong><div style="color:#0f172a;margin-top:4px;">{entry_context['volume_vs_avg_pct']:+.1f}% vol / {entry_context['risk_reward_ratio']}:1 RR</div></td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0;"><strong>Tech / Fund</strong><div style="color:#0f172a;margin-top:4px;">{tech_score} / {fund_score}</div></td>
                            <td style="padding:6px 0;"><strong>AdvFund / Sentiment</strong><div style="color:#0f172a;margin-top:4px;">{adv_fund_score} / {sentiment_score} ({sentiment_label})</div></td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0;"><strong>Target / Stop</strong><div style="color:#0f172a;margin-top:4px;">{risk_data['target']} / {risk_data['stop_loss']}</div></td>
                            <td style="padding:6px 0;"><strong>Trend</strong><div style="color:#0f172a;margin-top:4px;">{market_context['trend']}</div></td>
                        </tr>
                        <tr>
                            <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;">
                                <div style="font-size:13px;color:#475569;"><strong>Buy Levels:</strong></div>
                                <div style="font-size:12px;color:#0f172a;margin-top:4px;">
                                    <span style="color:#047857;font-weight:700;">Recommended {risk_data['recommended_entry_label']}: <strong>{risk_data['recommended_buy_level']}</strong></span>
                                    <div style="margin-top:4px;color:#64748b;">
                                        Patient: <strong>{risk_data['buy_levels']['patient_entry']}</strong> &bull;
                                        Optimal: <strong>{risk_data['buy_levels']['optimal_entry']}</strong> &bull;
                                        Aggressive: <strong>{risk_data['buy_levels']['aggressive_entry']}</strong>
                                    </div>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;">
                                <div style="font-size:13px;color:#475569;"><strong>Upcoming Events</strong></div>
                                <div style="margin-top:6px;padding:8px 10px;border-radius:8px;background:#fef3c7;border:1px solid #f59e0b;">
                                    <div style="font-size:13px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:0.04em;">{upcoming_events.get('next_upcoming_event_label', 'Upcoming Event')}</div>
                                    <div style="margin-top:3px;font-size:15px;font-weight:800;color:#b45309;">{upcoming_events.get('next_upcoming_event_date', 'Not available')}</div>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;font-size:13px;color:#475569;"><strong>News:</strong> {news_text or 'No recent headlines.'}</td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        """
        print(f"{stock_name} ({ticker}) -> {signal} | Score: {total_score}")
        return (priority, total_score, stock_name, row_html)
    except Exception as e:
        error_text = f"Error processing {ticker}: {str(e)}"
        print(error_text)
        traceback.print_exc()
        err_html = f"""
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 20px;border-radius:8px;background:#fff7f7;border:1px solid #f5c2c7;">
            <tr>
                <td style="padding:12px;color:#721c24;font-size:13px;"><strong>Error:</strong> {error_text}</td>
            </tr>
        </table>
        """
        return (4, 0, ticker, err_html)


def get_section_html(title, count, items):
    """
    Generates the HTML for a section of the report.
    """
    if not items:
        return ""

    header = f'<tr><td style="padding:12px 20px 0;"><h2 style="margin:0;font-size:15px;">{title} ({count})</h2></td></tr>'
    rows_html = "".join([f'<tr><td style="padding:0 20px;">{html}</td></tr>' for _, _, html in items])
    return header + rows_html


# -----------------------------
# Main
# -----------------------------
def main(mode, use_llm, detailed_llm=False):
    log.info(f"Starting stock analysis run. Mode: {mode}, LLM Enabled: {use_llm}, Detailed LLM: {detailed_llm}")
    report_html = """
<html>
  <body style="margin:0;padding:0;background:#f4f6f8;font-family:Arial,sans-serif;color:#111827;">
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#f4f6f8;width:100%;min-width:100%;">
      <tr>
        <td align="center" style="padding:16px;">
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="max-width:680px;min-width:320px;background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;">
            <tr>
              <td style="padding:18px 20px 12px;">
                <h1 style="margin:0;font-size:22px;line-height:1.2;color:#111827;"> Stock Report</h1>
                <p style="margin:10px 0 0;font-size:14px;line-height:1.6;color:#4b5563;">A clean mobile-friendly stock update with color-coded signal cards and compact metrics optimized for Gmail and Outlook.</p>
              </td>
            </tr>
            <tr>
              <td style="padding:0 20px 18px;border-top:1px solid #e5e7eb;">
                <p style="margin:0;font-size:12px;color:#6b7280;line-height:1.5;">Each stock is shown in its own card, with the most important metrics and a short insight summary. Scroll vertically on mobile for the best view.</p>
              </td>
            </tr>
"""

    rows = []
    if use_llm:
        init_llm_generator()

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {
            executor.submit(process_stock, name, ticker, use_llm, detailed_llm): (name, ticker)
            for name, ticker in STOCKS.items()
        }
        for future in as_completed(future_to_stock):
            stock_name, ticker = future_to_stock[future]
            try:
                result = future.result()
                if result:
                    rows.append(result)
            except Exception as exc:
                log.error(f"Error processing {stock_name} ({ticker}) in executor: {exc}")
                err_html = f"""
                <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 20px;border-radius:8px;background:#fff7f7;border:1px solid #f5c2c7;">
                    <tr>
                        <td style="padding:12px;color:#721c24;font-size:13px;"><strong>Error processing {ticker}:</strong> {exc}</td>
                    </tr>
                </table>
                """
                rows.append((4, 0, ticker, err_html))

    # This block is now correctly dedented and will run only once
    groups = {"Buy": [], "Hold": [], "Sell": [], "Errors": []}
    for pr, score, name, html in rows:
        if pr == 1:
            groups["Buy"].append((score, name, html))
        elif pr == 2:
            groups["Hold"].append((score, name, html))
        elif pr == 3:
            groups["Sell"].append((score, name, html))
        else:
            groups["Errors"].append((score, name, html))

    for key in groups:
        groups[key].sort(key=lambda item: item[0], reverse=True)

    buy_count = len(groups["Buy"])
    hold_count = len(groups["Hold"])
    sell_count = len(groups["Sell"])
    err_count = len(groups["Errors"])

    # Format date for the report header
    now_utc = datetime.now(ZoneInfo("UTC"))
    now_ist = now_utc.astimezone(ZoneInfo("Asia/Kolkata"))
    formatted_date_ist = now_ist.strftime('%A, %d %B %Y, %I:%M %p %Z')

    summary_html = f"""
        <tr>
          <td style="padding:16px 20px;border-top:1px solid #e5e7eb;">
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
              <tr>
                <td>
                  <h2 style="margin:0;font-size:16px;">Portfolio Snapshot</h2>
                </td>
                <td style="text-align:right;font-size:13px;color:#475569;">{formatted_date_ist}</td>
              </tr>
            </table>
            <p style="margin:8px 0 0;font-size:13px;color:#4b5563;">Total symbols: <strong>{len(rows)}</strong>  &bull;  Buy: <strong>{buy_count}</strong>  &bull;  Hold: <strong>{hold_count}</strong>  &bull;  Sell: <strong>{sell_count}</strong></p>
          </td>
        </tr>
    """

    section_html = get_section_html("Buy", buy_count, groups["Buy"])
    section_html += get_section_html("Hold", hold_count, groups["Hold"])
    section_html += get_section_html("Sell", sell_count, groups["Sell"])
    section_html += get_section_html("Errors", err_count, groups["Errors"])

    report_html += summary_html + section_html
    report_html += """
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
    """

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("report.html", "w") as f:
            f.write(report_html)
        log.info("Report saved to report.html (DRY_RUN enabled)")
    else:
        try:
         tracker = CommodityTracker()
         commodity_html = tracker.generate_html()
         report_html += commodity_html
        except Exception as e:
         log.error(f"Commodity tracker failed: {e}")
         traceback.print_exc()

        # Continue even if commodity API fails
        send_email(report_html, mode)
        log.info("Email report sent successfully.")
    log.info("Stock analysis run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock analysis.")
    parser.add_argument(
        "--mode",
        type=str,
        default="eod",
        choices=["real_time", "eod"],
        help="Specify the mode: real_time or eod"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable AI-powered reasoning to speed up execution."
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include full detailed LLM analysis in the report (can increase size)."
    )
    args = parser.parse_args()
    main(args.mode, use_llm=not args.no_llm, detailed_llm=args.detailed)
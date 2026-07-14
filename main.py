import os
import math
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


def get_conviction_rating(score):
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0

    numeric_score = max(0.0, min(100.0, numeric_score))

    if numeric_score >= 90:
        filled_icons = 5
        label = "Elite"
        fill_color = "#15803d"
    elif numeric_score >= 75:
        filled_icons = 4
        label = "High"
        fill_color = "#047857"
    elif numeric_score >= 60:
        filled_icons = 3
        label = "Strong"
        fill_color = "#0f766e"
    elif numeric_score >= 45:
        filled_icons = 2
        label = "Mixed"
        fill_color = "#d97706"
    elif numeric_score >= 30:
        filled_icons = 1
        label = "Low"
        fill_color = "#dc2626"
    else:
        filled_icons = 0
        label = "Very Low"
        fill_color = "#b91c1c"

    icons = []
    icons_text = []
    for index in range(5):
        active = index < filled_icons
        icon_background = fill_color if active else "#f8fafc"
        icon_border = fill_color if active else "#dbe3ea"
        icon_color = "#ffffff" if active else "#cbd5e1"
        glyph = "★" if active else "☆"
        icons_text.append(glyph)
        icons.append(
            f'<span style="display:inline-block;width:26px;height:26px;'
            f'line-height:26px;text-align:center;border-radius:999px;'
            f'margin-left:2px;margin-right:2px;font-size:15px;font-weight:900;'
            f'background:{icon_background};border:1px solid {icon_border};'
            f'color:{icon_color};box-shadow:0 1px 2px rgba(15,23,42,0.08);">'
            f'{glyph}</span>'
        )

    return {
        "label": label,
        "fill_color": fill_color,
        "icons_html": "".join(icons),
        "icons_text": "".join(icons_text),
    }


def _safe_float(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(numeric_value):
        return None

    return numeric_value


def _risk_level_meta(score):
    if score <= 1.5:
        return {
            "label": "Low",
            "emoji": "🟢",
            "color": "#047857",
            "background": "#dcfce7",
            "border": "#86efac",
        }
    if score <= 2.25:
        return {
            "label": "Medium",
            "emoji": "🟡",
            "color": "#a16207",
            "background": "#fef3c7",
            "border": "#fcd34d",
        }
    return {
        "label": "High",
        "emoji": "🔴",
        "color": "#b91c1c",
        "background": "#fee2e2",
        "border": "#fca5a5",
    }


def _risk_component_score(value, low_threshold, medium_threshold, reverse=False):
    if value is None:
        return 2

    if reverse:
        if value >= low_threshold:
            return 1
        if value >= medium_threshold:
            return 2
        return 3

    if value <= low_threshold:
        return 1
    if value <= medium_threshold:
        return 2
    return 3


def calculate_risk_meter(df, latest, beta=None):
    close = _safe_float(latest.get("close"))
    atr = _safe_float(latest.get("atr"))
    adx = _safe_float(latest.get("adx"))
    beta_value = _safe_float(beta)

    atr_pct = round((atr / close) * 100, 2) if atr is not None and close not in (None, 0) else None

    volatility_pct = None
    recent_returns = df["close"].pct_change().dropna().tail(20)
    if len(recent_returns) >= 2:
        recent_volatility = recent_returns.std()
        if pd.notna(recent_volatility):
            volatility_pct = round(float(recent_volatility) * math.sqrt(252) * 100, 2)

    atr_score = _risk_component_score(atr_pct, 2.0, 4.0)
    volatility_score = _risk_component_score(volatility_pct, 20.0, 35.0)
    beta_score = _risk_component_score(beta_value, 0.9, 1.2)
    adx_score = _risk_component_score(adx, 25.0, 18.0, reverse=True)

    overall_score = round((atr_score + volatility_score + beta_score + adx_score) / 4, 2)
    overall_meta = _risk_level_meta(overall_score)

    factor_rows = [
        {
            "label": "ATR",
            "value": f"{atr_pct:.2f}%" if atr_pct is not None else "n/a",
            "meta": _risk_level_meta(atr_score),
        },
        {
            "label": "Volatility",
            "value": f"{volatility_pct:.2f}%" if volatility_pct is not None else "n/a",
            "meta": _risk_level_meta(volatility_score),
        },
        {
            "label": "Beta",
            "value": f"{beta_value:.2f}" if beta_value is not None else "n/a",
            "meta": _risk_level_meta(beta_score),
        },
        {
            "label": "ADX",
            "value": f"{adx:.2f}" if adx is not None else "n/a",
            "meta": _risk_level_meta(adx_score),
        },
    ]

    return {
        "label": overall_meta["label"],
        "emoji": overall_meta["emoji"],
        "color": overall_meta["color"],
        "background": overall_meta["background"],
        "border": overall_meta["border"],
        "score": overall_score,
        "factors": factor_rows,
    }


def build_risk_meter_html(risk_meter):
    legend_cells = []
    for option in ("Low", "Medium", "High"):
        meta = _risk_level_meta(1 if option == "Low" else 2 if option == "Medium" else 3)
        active = option == risk_meter["label"]
        background = meta["background"] if active else "#f8fafc"
        border     = meta["border"]     if active else "#e2e8f0"
        color      = meta["color"]      if active else "#94a3b8"
        opacity    = "1"                if active else "0.5"
        legend_cells.append(
            f'<td style="padding:0 4px 0 0;vertical-align:top;width:33.33%;">'
            f'<div style="padding:7px 6px;border-radius:8px;text-align:center;'
            f'background:{background};border:1px solid {border};color:{color};'
            f'font-size:12px;font-weight:800;opacity:{opacity};white-space:nowrap;">'
            f'{meta["emoji"]} {option}</div></td>'
        )
    # Remove trailing right-padding on last cell
    legend_cells[-1] = legend_cells[-1].replace('padding:0 4px 0 0;', 'padding:0;')

    def _factor_cell(factor, right=False):
        pad = "padding:8px 0 0 10px;" if right else "padding:8px 10px 0 0;"
        return (
            f'<td style="{pad}width:50%;vertical-align:top;">'
            f'<div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.03em;">'
            f'{factor["label"]}</div>'
            f'<div style="margin-top:3px;color:#0f172a;font-size:13px;font-weight:700;">'
            f'{factor["value"]}</div>'
            f'<div style="margin-top:2px;font-size:11px;color:{factor["meta"]["color"]};font-weight:700;">'
            f'{factor["meta"]["emoji"]} {factor["meta"]["label"]}</div></td>'
        )

    factors = risk_meter["factors"]

    return f"""
                            <tr>
                                <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;">
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                                        <tr>
                                            <td style="vertical-align:middle;">
                                                <div style="font-size:13px;color:#475569;font-weight:700;">Risk Meter</div>
                                                <div style="margin-top:2px;font-size:11px;color:#94a3b8;">ATR &bull; Volatility &bull; Beta &bull; ADX</div>
                                            </td>
                                            <td style="text-align:right;vertical-align:middle;">
                                                <span style="display:inline-block;padding:5px 12px;border-radius:999px;background:{risk_meter['background']};border:1px solid {risk_meter['border']};color:{risk_meter['color']};font-size:12px;font-weight:800;white-space:nowrap;">{risk_meter['emoji']} {risk_meter['label']} Risk</span>
                                            </td>
                                        </tr>
                                    </table>
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:8px;border-collapse:collapse;">
                                        <tr>
                                            {''.join(legend_cells)}
                                        </tr>
                                    </table>
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:4px;border-collapse:collapse;">
                                        <tr>
                                            {_factor_cell(factors[0])}
                                            {_factor_cell(factors[1], right=True)}
                                        </tr>
                                        <tr>
                                            {_factor_cell(factors[2])}
                                            {_factor_cell(factors[3], right=True)}
                                        </tr>
                                    </table>
                                </td>
                            </tr>
    """


def calculate_52_week_range(df, latest):
    current_price = _safe_float(latest.get("close"))
    recent_history = df.tail(252)

    high_52w = _safe_float(recent_history["high"].max()) if "high" in recent_history else None
    low_52w = _safe_float(recent_history["low"].min()) if "low" in recent_history else None

    below_high_pct = None
    if current_price is not None and high_52w not in (None, 0):
        below_high_pct = round(((current_price - high_52w) / high_52w) * 100, 2)

    above_low_pct = None
    if current_price is not None and low_52w not in (None, 0):
        above_low_pct = round(((current_price - low_52w) / low_52w) * 100, 2)

    if below_high_pct is None:
        high_distance_text = "n/a"
        high_distance_color = "#64748b"
    elif below_high_pct <= 0:
        high_distance_text = f"↓ {abs(below_high_pct):.1f}% below high"
        high_distance_color = "#dc2626" if below_high_pct <= -15 else "#d97706"
    else:
        high_distance_text = f"↑ {below_high_pct:.1f}% above high"
        high_distance_color = "#047857"

    if above_low_pct is None:
        low_distance_text = "n/a"
        low_distance_color = "#64748b"
    elif above_low_pct >= 0:
        low_distance_text = f"↑ {above_low_pct:.1f}% above low"
        low_distance_color = "#047857" if above_low_pct >= 20 else "#d97706"
    else:
        low_distance_text = f"↓ {abs(above_low_pct):.1f}% below low"
        low_distance_color = "#dc2626"

    return {
        "current_price": current_price,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "high_distance_text": high_distance_text,
        "high_distance_color": high_distance_color,
        "low_distance_text": low_distance_text,
        "low_distance_color": low_distance_color,
    }


def _format_rupee(value):
    if value is None:
        return "n/a"
    return f"₹{value:.2f}"


def build_52_week_range_html(range_data):
    return f"""
                            <tr>
                                <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;">
                                    <!-- Section heading row -->
                                    <div style="font-size:13px;color:#475569;font-weight:700;">Distance from 52-Week High / Low</div>
                                    <!-- High | Current | Low three-column price display -->
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:10px;border-collapse:collapse;">
                                        <tr>
                                            <!-- 52W High -->
                                            <td style="width:33.33%;vertical-align:top;padding-right:6px;">
                                                <div style="padding:10px;border-radius:10px;background:#f0fdf4;border:1px solid #bbf7d0;text-align:center;">
                                                    <div style="font-size:10px;color:#16a34a;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">52W High</div>
                                                    <div style="margin-top:5px;color:#0f172a;font-size:14px;font-weight:800;">{_format_rupee(range_data['high_52w'])}</div>
                                                </div>
                                            </td>
                                            <!-- Current Price -->
                                            <td style="width:33.33%;vertical-align:top;padding:0 3px;">
                                                <div style="padding:10px;border-radius:10px;background:#f8fafc;border:1px solid #e2e8f0;text-align:center;">
                                                    <div style="font-size:10px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">Current</div>
                                                    <div style="margin-top:5px;color:#0f172a;font-size:14px;font-weight:800;">{_format_rupee(range_data['current_price'])}</div>
                                                </div>
                                            </td>
                                            <!-- 52W Low -->
                                            <td style="width:33.33%;vertical-align:top;padding-left:6px;">
                                                <div style="padding:10px;border-radius:10px;background:#fef2f2;border:1px solid #fecaca;text-align:center;">
                                                    <div style="font-size:10px;color:#dc2626;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">52W Low</div>
                                                    <div style="margin-top:5px;color:#0f172a;font-size:14px;font-weight:800;">{_format_rupee(range_data['low_52w'])}</div>
                                                </div>
                                            </td>
                                        </tr>
                                    </table>
                                    <!-- Distance badges -->
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:8px;border-collapse:collapse;">
                                        <tr>
                                            <td style="width:50%;padding-right:4px;vertical-align:top;">
                                                <div style="padding:7px 10px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;">
                                                    <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.03em;">vs 52W High</div>
                                                    <div style="margin-top:4px;font-size:13px;font-weight:800;color:{range_data['high_distance_color']};">{range_data['high_distance_text']}</div>
                                                </div>
                                            </td>
                                            <td style="width:50%;padding-left:4px;vertical-align:top;">
                                                <div style="padding:7px 10px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;">
                                                    <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.03em;">vs 52W Low</div>
                                                    <div style="margin-top:4px;font-size:13px;font-weight:800;color:{range_data['low_distance_color']};">{range_data['low_distance_text']}</div>
                                                </div>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
    """


def _format_ratio(value, decimals=1):
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return "n/a"
    return f"{numeric_value:.{decimals}f}"


def _format_percent(value, decimals=1):
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return "n/a"
    if abs(numeric_value) <= 1:
        numeric_value *= 100
    return f"{numeric_value:.{decimals}f}%"


def _fund_color(label, raw_value):
    """Return (value_color, hint_text, hint_color) based on metric health."""
    v = _safe_float(raw_value)
    if v is None:
        return "#64748b", "No data", "#94a3b8"

    if label == "PE":
        if v < 15:
            return "#047857", "Undervalued", "#047857"
        if v < 25:
            return "#0f172a", "Fair", "#64748b"
        if v < 40:
            return "#d97706", "Stretched", "#d97706"
        return "#dc2626", "Expensive", "#dc2626"

    if label == "PB":
        if v < 1:
            return "#047857", "Below book", "#047857"
        if v < 3:
            return "#0f172a", "Reasonable", "#64748b"
        if v < 6:
            return "#d97706", "Premium", "#d97706"
        return "#dc2626", "Very high", "#dc2626"

    if label == "Dividend Yield":
        # raw_value comes in as a fraction (e.g. 0.023) or percent
        pct = v * 100 if v <= 1 else v
        if pct >= 3:
            return "#047857", "High yield", "#047857"
        if pct >= 1:
            return "#0f172a", "Moderate", "#64748b"
        if pct > 0:
            return "#d97706", "Low yield", "#d97706"
        return "#94a3b8", "No dividend", "#94a3b8"

    if label == "ROE":
        pct = v * 100 if v <= 1 else v
        if pct >= 20:
            return "#047857", "Strong", "#047857"
        if pct >= 12:
            return "#0f172a", "Decent", "#64748b"
        if pct >= 0:
            return "#d97706", "Weak", "#d97706"
        return "#dc2626", "Negative", "#dc2626"

    if label == "Debt / Equity":
        if v < 30:
            return "#047857", "Low debt", "#047857"
        if v < 100:
            return "#0f172a", "Manageable", "#64748b"
        if v < 200:
            return "#d97706", "High debt", "#d97706"
        return "#dc2626", "Very high", "#dc2626"

    return "#0f172a", "", "#64748b"


def build_fundamentals_html(fundamentals, fund_score):
    # Score pill color based on score value
    if fund_score >= 70:
        score_bg, score_border, score_color = "#f0fdf4", "#bbf7d0", "#15803d"
    elif fund_score >= 40:
        score_bg, score_border, score_color = "#fffbeb", "#fde68a", "#b45309"
    else:
        score_bg, score_border, score_color = "#fef2f2", "#fecaca", "#b91c1c"

    def _tile(label, raw_value, formatted_value, right=False):
        val_color, hint, hint_color = _fund_color(label, raw_value)
        pad = "padding:0 0 6px 6px;" if right else "padding:0 6px 6px 0;"
        hint_html = (
            f'<div style="margin-top:3px;font-size:10px;font-weight:700;color:{hint_color};">{hint}</div>'
            if hint else ""
        )
        return (
            f'<td style="{pad}width:50%;vertical-align:top;">'
            f'<div style="padding:9px 10px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;">'
            f'<div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">{label}</div>'
            f'<div style="margin-top:5px;font-size:15px;font-weight:800;color:{val_color};">{formatted_value}</div>'
            f'{hint_html}'
            f'</div></td>'
        )

    pe_raw   = fundamentals.get("pe")
    pb_raw   = fundamentals.get("pb")
    div_raw  = fundamentals.get("dividendYield")
    roe_raw  = fundamentals.get("roe")
    debt_raw = fundamentals.get("debtToEquity")

    pe_fmt   = _format_ratio(pe_raw, 1)
    pb_fmt   = _format_ratio(pb_raw, 1)
    div_fmt  = _format_percent(div_raw, 1)
    roe_fmt  = _format_percent(roe_raw, 1)
    debt_fmt = _format_ratio(debt_raw, 1)

    return f"""
                            <tr>
                                <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;">
                                    <!-- Header: title + score pill -->
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                                        <tr>
                                            <td style="vertical-align:middle;">
                                                <div style="font-size:13px;color:#475569;font-weight:700;">Fundamentals</div>
                                            </td>
                                            <td style="text-align:right;vertical-align:middle;">
                                                <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{score_bg};border:1px solid {score_border};color:{score_color};font-size:12px;font-weight:800;white-space:nowrap;">Score {fund_score}</span>
                                            </td>
                                        </tr>
                                    </table>
                                    <!-- Metric tiles grid -->
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:8px;border-collapse:collapse;">
                                        <tr>
                                            {_tile("PE",   pe_raw,   pe_fmt)}
                                            {_tile("PB",   pb_raw,   pb_fmt,  right=True)}
                                        </tr>
                                        <tr>
                                            {_tile("ROE",            roe_raw,  roe_fmt)}
                                            {_tile("Dividend Yield", div_raw,  div_fmt, right=True)}
                                        </tr>
                                        <tr>
                                            {_tile("Debt / Equity",  debt_raw, debt_fmt)}
                                            <td style="width:50%;padding:0 0 6px 6px;vertical-align:top;"></td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
    """


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
        risk_meter = calculate_risk_meter(df, latest, fund_raw.get("beta"))
        range_52w = calculate_52_week_range(df, latest)

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
        conviction_rating = get_conviction_rating(total_score)

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

        events = upcoming_events
        
        if events.get("has_event"):
        
            details = []

            if events["dividend_record_date"] != "NA":
                details.append(
                    f"<div><strong>Dividend Record:</strong> {events['dividend_record_date']}</div>"
                )

            if events["results_announcement_date"] != "NA":
                details.append(
                    f"<div><strong>Results Announcement:</strong> {events['results_announcement_date']}</div>"
                )

            events_html = f"""
            <div style="margin-top:6px;padding:10px 12px;
                        border-radius:8px;
                        background:#FEF3C7;
                        border-left:4px solid #F59E0B;">

                <div style="font-size:12px;
                            color:#92400E;
                            font-weight:bold;
                            text-transform:uppercase;">
                    Next Upcoming Event
                </div>

                <div style="font-size:16px;
                            color:#92400E;
                            font-weight:bold;
                            margin-top:3px;">
                    {events['next_upcoming_event_label']}
                </div>

                <div style="font-size:14px;
                            color:#B45309;
                            margin-top:3px;">
                    {events['next_upcoming_event_date']}
                </div>

                <hr style="margin:8px 0;border:none;border-top:1px solid #FCD34D;">

                {''.join(details)}

            </div>
            """

        else:

            events_html = """
            <div style="margin-top:6px;
                        padding:10px;
                        border-radius:8px;
                        background:#F8FAFC;
                        border:1px solid #CBD5E1;
                        color:#475569;
                        font-size:13px;">
                No dividend or earnings announcements are scheduled in the next 60 days.
            </div>
            """
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
                                <td style="width:150px;text-align:right;vertical-align:top;">
                                    <div style="display:inline-block;padding:6px 10px;border-radius:999px;font-weight:700;color:#fff;background:{'#dc2626' if 'sell' in signal.lower() else '#f59e0b' if 'hold' in signal.lower() else '#047857'};">{signal}</div>
                                    <div style="margin-top:10px;font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:#64748b;">Conviction</div>
                                    <div style="margin-top:6px;white-space:nowrap;">{conviction_rating['icons_html']}</div>
                                    <div style="margin-top:6px;font-size:12px;font-weight:700;color:{conviction_rating['fill_color']};">{conviction_rating['label']}</div>
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
                                <td style="padding:6px 0;"><strong>Technical Score</strong><div style="color:#0f172a;margin-top:4px;">{tech_score}</div></td>
                                <td style="padding:6px 0;"><strong>Sentiment</strong><div style="color:#0f172a;margin-top:4px;">{sentiment_score} ({sentiment_label})</div></td>
                            </tr>
                            <tr>
                                <td style="padding:6px 0;"><strong>Target / Stop</strong><div style="color:#0f172a;margin-top:4px;">{risk_data['target']} / {risk_data['stop_loss']}</div></td>
                                <td style="padding:6px 0;"><strong>Trend</strong><div style="color:#0f172a;margin-top:4px;">{market_context['trend']}</div></td>
                            </tr>
                            {build_fundamentals_html(fund_raw, fund_score)}
                            {build_52_week_range_html(range_52w)}
                            {build_risk_meter_html(risk_meter)}
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
                            <div style="font-size:13px;color:#475569;">
                            <strong>Upcoming Events</strong>
                        </div>

                        {events_html}
                                   
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
        summary_entry = {
            "stock_name": stock_name,
            "signal": signal,
            "current_price": round(latest["close"], 2),
            "recommended_buy_level": risk_data.get("recommended_buy_level"),
            "ema20": latest.get("ema20"),
            "upcoming_events": events,
        }

        print(f"{stock_name} ({ticker}) -> {signal} | Conviction: {conviction_rating['label']} {conviction_rating['icons_text']} | Risk: {risk_meter['label']}")
        return (priority, total_score, stock_name, row_html, summary_entry) 
    
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
    return (4, 0, ticker, err_html, {
        "stock_name": ticker,
        "signal": "ERROR",
        "current_price": None,
        "recommended_buy_level": None,
        "ema20": None,
        "upcoming_events": {},
    })


def _format_short_date(value):
    if not value or str(value).upper() == "NA":
        return None

    text = str(value).strip()
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%d %b")
        except ValueError:
            continue
    return text


def _relative_day_label(value):
    if not value or str(value).upper() == "NA":
        return None

    text = str(value).strip()
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            event_date = datetime.strptime(text, fmt).date()
            break
        except ValueError:
            event_date = None

    if event_date is None:
        return None

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    delta_days = (event_date - today).days
    if delta_days == 0:
        return "today"
    if delta_days == 1:
        return "in 1 day"
    if delta_days > 1:
        return f"in {delta_days} days"
    if delta_days == -1:
        return "1 day ago"
    return f"{abs(delta_days)} days ago"


def build_quick_summary(rows):
    """
    Builds compact, human-readable summary bullets for the top of the report.
    """
    if not rows:
        return ""

    items = []
    for row in rows:
        stock_name = row.get("stock_name") or ""
        signal = str(row.get("signal") or "").upper()
        current_price = row.get("current_price")
        recommended_buy_level = row.get("recommended_buy_level")
        ema20 = row.get("ema20")
        upcoming_events = row.get("upcoming_events") or {}

        if "BUY" in signal and current_price is not None and recommended_buy_level is not None and current_price < recommended_buy_level:
            items.append(f"✅ Buy: {stock_name} below ₹{recommended_buy_level}")
            continue

        results_date = upcoming_events.get("results_announcement_date")
        if results_date and str(results_date).upper() != "NA":
            short_date = _format_short_date(results_date)
            relative_label = _relative_day_label(results_date)
            if relative_label == "today":
                items.append(f"✅ Watch: {stock_name} results today")
            else:
                items.append(f"✅ Watch: {stock_name} results on {short_date or results_date}")
            continue

        dividend_date = upcoming_events.get("dividend_record_date")
        if dividend_date and str(dividend_date).upper() != "NA":
            relative_label = _relative_day_label(dividend_date)
            if relative_label == "today":
                items.append(f"📅 {stock_name} dividend record today")
            else:
                items.append(f"📅 {stock_name} dividend record {relative_label or 'soon'}")
            continue

        if current_price is not None and ema20 is not None and current_price < ema20:
            items.append(f"⚠ {stock_name} below EMA20—avoid adding")

    return "\n".join(items)


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
    summary_rows = []
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
                    summary_rows.append(result[4])
            except Exception as exc:
                log.error(f"Error processing {stock_name} ({ticker}) in executor: {exc}")
                err_html = f"""
                <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 20px;border-radius:8px;background:#fff7f7;border:1px solid #f5c2c7;">
                    <tr>
                        <td style="padding:12px;color:#721c24;font-size:13px;"><strong>Error processing {ticker}:</strong> {exc}</td>
                    </tr>
                </table>
                """
                rows.append((4, 0, ticker, err_html, {
                    "stock_name": stock_name,
                    "signal": "ERROR",
                    "current_price": None,
                    "recommended_buy_level": None,
                    "ema20": None,
                    "upcoming_events": {},
                }))

    # This block is now correctly dedented and will run only once
    groups = {"Buy": [], "Hold": [], "Sell": [], "Errors": []}
    for pr, score, name, html, _ in rows:
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

    quick_summary_text = build_quick_summary(summary_rows)

    # Fetch commodity data early so we can include it in the quick summary
    # and reuse the same data object when rendering the full section below.
    commodity_data = None
    commodity_fetch_error = None
    try:
        tracker = CommodityTracker()
        commodity_data = tracker.get_commodity_data()
    except Exception as e:
        commodity_fetch_error = e
        log.error(f"Commodity tracker failed during data fetch: {e}")
        traceback.print_exc()

    # Build commodity summary bullets
    commodity_bullets = []
    if commodity_data:
        for metal, key in [("Gold (22K)", "gold"), ("Silver", "silver")]:
            metal_data = commodity_data[key]
            change = metal_data.get("change", 0)
            current = metal_data.get("current", 0)
            history = metal_data.get("history", [])

            change_sign = "+" if change > 0 else ""
            direction = "↑" if change > 0 else ("↓" if change < 0 else "→")

            # Check if a buy signal is triggered
            buy_triggered = False
            if len(history) >= 3:
                recent = [r["change"] for r in history[-3:]]
                latest_c, prev_c, older_c = recent[-1], recent[-2], recent[-3]
                score = 0
                if latest_c <= -1.5: score += 4
                if latest_c <= -2.5: score += 4
                if prev_c <= -1.0:   score += 2
                if older_c <= -1.0:  score += 1
                if latest_c < prev_c: score += 2
                if latest_c < 0:     score += 1
                buy_triggered = score >= 8

            if buy_triggered:
                commodity_bullets.append(
                    f"✅ Buy Signal: {metal} at &#8377;{current:.2f} ({change_sign}{change}%)"
                )
            elif change <= -1.5:
                commodity_bullets.append(
                    f"📉 {metal} {direction} {change_sign}{change}% — watching for entry"
                )
            elif change >= 1.5:
                commodity_bullets.append(
                    f"📈 {metal} {direction} {change_sign}{change}% — momentum up"
                )
            else:
                commodity_bullets.append(
                    f"⬛ {metal} {direction} {change_sign}{change}% — stable"
                )

    # Merge stock bullets + commodity bullets into one quick summary block
    all_bullets = (quick_summary_text.splitlines() if quick_summary_text else []) + commodity_bullets

    quick_summary_html = ""
    if all_bullets:
        stock_bullet_html = "".join(
            f'<div style="margin:4px 0 0;font-size:13px;color:#0f172a;">{item}</div>'
            for item in (quick_summary_text.splitlines() if quick_summary_text else [])
        )
        commodity_bullet_html = ""
        if commodity_bullets:
            commodity_bullet_html = (
                '<div style="margin-top:8px;padding-top:8px;border-top:1px solid #dbeafe;">'
                '<div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;'
                'letter-spacing:0.04em;margin-bottom:4px;">Commodities</div>'
                + "".join(
                    f'<div style="margin:4px 0 0;font-size:13px;color:#0f172a;">{item}</div>'
                    for item in commodity_bullets
                )
                + '</div>'
            )

        quick_summary_html = f"""
            <tr>
              <td style="padding:0 20px 12px;">
                <div style="border:1px solid #dbeafe;border-left:4px solid #2563eb;border-radius:10px;background:#f8fbff;padding:10px 12px;">
                  <div style="font-size:12px;font-weight:700;color:#2563eb;text-transform:uppercase;letter-spacing:0.04em;">Quick Summary</div>
                  {stock_bullet_html}
                  {commodity_bullet_html}
                </div>
              </td>
            </tr>
        """

    section_html = get_section_html("Buy", buy_count, groups["Buy"])
    section_html += get_section_html("Hold", hold_count, groups["Hold"])
    section_html += get_section_html("Sell", sell_count, groups["Sell"])
    section_html += get_section_html("Errors", err_count, groups["Errors"])

    # Commodity section (Gold & Silver) — built BEFORE stock sections so it appears
    # near the top of the email and is never clipped by Gmail's 102 KB limit.
    commodity_row_html = ""
    if commodity_data is not None:
        try:
            gold_levels   = tracker.derive_buy_levels(commodity_data["gold"]["current"],   commodity_data["gold"]["history"])
            silver_levels = tracker.derive_buy_levels(commodity_data["silver"]["current"], commodity_data["silver"]["history"])
            gold_plan   = tracker.build_trade_plan(commodity_data["gold"]["current"],   commodity_data["gold"]["history"],   gold_levels)
            silver_plan = tracker.build_trade_plan(commodity_data["silver"]["current"], commodity_data["silver"]["history"], silver_levels)

            gold_card = tracker._commodity_card_html(
                name="Gold (22K)", ticker_label="XAU/INR",
                current_price=commodity_data["gold"]["current"],
                change=commodity_data["gold"]["change"],
                history=commodity_data["gold"]["history"],
                levels=gold_levels, plan=gold_plan,
            )
            silver_card = tracker._commodity_card_html(
                name="Silver", ticker_label="XAG/INR",
                current_price=commodity_data["silver"]["current"],
                change=commodity_data["silver"]["change"],
                history=commodity_data["silver"]["history"],
                levels=silver_levels, plan=silver_plan,
            )
            commodity_section_html = f"""
                <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
                    <tr>
                        <td style="padding:12px 0 0;">
                            <h2 style="margin:0;font-size:15px;color:#111827;">Commodities (2)</h2>
                        </td>
                    </tr>
                </table>
                {gold_card}
                {silver_card}"""
            commodity_row_html = f"""
                <tr>
                  <td style="padding:0 20px 20px;">
                    {commodity_section_html}
                  </td>
                </tr>
            """
        except Exception as e:
            log.error(f"Commodity card render failed: {e}")
            traceback.print_exc()
            commodity_row_html = """
                <tr>
                  <td style="padding:0 20px 20px;">
                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 0;border-radius:12px;background:#fff7f7;border:1px solid #f5c2c7;">
                      <tr>
                        <td style="padding:14px;color:#721c24;font-size:13px;">
                          <strong>Commodities Unavailable:</strong> Could not render Gold &amp; Silver cards.
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
            """
    else:
        commodity_row_html = """
            <tr>
              <td style="padding:0 20px 20px;">
                <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 0;border-radius:12px;background:#fff7f7;border:1px solid #f5c2c7;">
                  <tr>
                    <td style="padding:14px;color:#721c24;font-size:13px;">
                      <strong>Commodities Unavailable:</strong> Could not fetch Gold &amp; Silver prices at this time.
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
        """

    # Commodities first, then stock sections — ensures commodities are never clipped
    report_html += quick_summary_html + summary_html + commodity_row_html + section_html

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

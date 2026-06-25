import os
import yfinance as yf
import pandas as pd
import ta
import smtplib
import traceback
from email.mime.text import MIMEText
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from config import *
from stock_fetcher import fetch_fundamentals
from fundamentals import score_fundamentals
from advanced_fundamentals import fetch_advanced_fundamentals, score_advanced_fundamentals
from market_context import build_market_context
from news_engine import get_news
from sentiment_score import score_headlines
from scorer import technical_score, final_score, decision
from position_sizing import apply_risk_management

# -----------------------------
# Fetch historical data
# -----------------------------
def fetch_data(symbol):
    df = yf.download(
        symbol,
        period="300d",
        interval="1d",
        auto_adjust=True,
        progress=False
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


def init_llm_generator():
    if genai is None:
        return None

    api_key = os.getenv("GOOGLE_API_KEY") or globals().get("GOOGLE_API_KEY")
    if not api_key:
        print("Google API key not found for Gemini.")
        return None

    try:
        genai.configure(api_key=api_key)
        return "gemini"
    except Exception as exc:
        print(f"Failed to initialize Gemini: {exc}")
        return None


def generate_gemini_reasoning(prompt):
    if genai is None:
        return None

    try:
        if hasattr(genai, "GenerativeModel") and hasattr(genai.GenerativeModel, "from_pretrained"):
            model = genai.GenerativeModel.from_pretrained("gemini-1.5")
            response = model.generate(prompt=prompt, max_output_tokens=220)
            if hasattr(response, "text"):
                return response.text.strip()
            if isinstance(response, dict):
                return response.get("output", {}).get("text", "").strip()

        if hasattr(genai, "generate"):
            response = genai.generate(model="gemini-1.5", prompt=prompt, max_output_tokens=220)
            if isinstance(response, dict):
                if "candidates" in response and response["candidates"]:
                    return response["candidates"][0].get("output", "").strip()
                if "output" in response and isinstance(response["output"], dict):
                    return response["output"].get("text", "").strip()
                return str(response.get("output", "")).strip()

        return None
    except Exception as exc:
        print(f"Gemini generation failed: {exc}")
        return None


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


def generate_llm_reasoning(stock_name, ticker, latest, tech_score, fund_score, sentiment_score, sentiment_label, tech_signal, signal, headlines):
    headlines_text = "; ".join(headlines[:3]) if headlines else "No recent headlines."
    prompt = (
        f"Write a concise, insightful investment note for {stock_name} ({ticker}). "
        f"Use the following data to explain the current posture, what is driving it, and what investors should watch next:\n"
        f"- Close: {round(latest['close'], 2)}\n"
        f"- EMA20: {round(latest['ema20'], 2)}\n"
        f"- EMA50: {round(latest['ema50'], 2)}\n"
        f"- EMA100: {round(latest['ema100'], 2)}\n"
        f"- EMA200: {round(latest['ema200'], 2)}\n"
        f"- RSI: {round(latest['rsi'], 2)}\n"
        f"- ADX: {round(latest['adx'], 2)}\n"
        f"- Technical score: {tech_score} ({tech_signal})\n"
        f"- Fundamental score: {fund_score}\n"
        f"- Sentiment: {sentiment_label} ({round(sentiment_score, 2)})\n"
        f"- Overall signal: {signal}\n"
        f"- Headlines: {headlines_text}\n\n"
        f"Output a short, clear rationale that highlights momentum, valuation/fundamental strength, sentiment context, and a key risk or catalyst. "
        f"Do not simply repeat the signal or scores. Focus on why the current posture makes sense and what could change it."
    )

    generator = init_llm_generator()
    if generator is None:
        headline_summary = headlines[:2]
        headline_text = "; ".join(headline_summary) if headline_summary else "No recent headlines."
        return (
            f"{stock_name} is currently in a {signal.lower()} posture. "
            f"Technical momentum is {tech_signal.lower()} with a score of {tech_score}, "
            f"fundamentals are {fund_score}, and sentiment is {sentiment_label.lower()} ({round(sentiment_score, 2)}). "
            f"Recent headlines: {headline_text}."
        )

    if generator == "gemini" and genai is not None:
        gemini_text = generate_gemini_reasoning(prompt)
        if gemini_text:
            return gemini_text

    headline_summary = headlines[:2]
    headline_text = "; ".join(headline_summary) if headline_summary else "No recent headlines."
    return (
        f"{stock_name} is currently in a {signal.lower()} posture. "
        f"Technical momentum is {tech_signal.lower()} with a score of {tech_score}, "
        f"fundamentals are {fund_score}, and sentiment is {sentiment_label.lower()} ({round(sentiment_score, 2)}). "
        f"Recent headlines: {headline_text}."
    )


# -----------------------------
# Email
# -----------------------------
def send_email(report_html):
    msg = MIMEText(report_html, "html")
    msg["Subject"] = "BlueOcean - Technical Stock Report"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_FROM, EMAIL_PASSWORD)
    server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    server.quit()


# -----------------------------
# Main
# -----------------------------
def main():
    html_report = """
<html>
  <body style="margin:0;padding:0;background:#f4f6f8;font-family:Arial,sans-serif;color:#111827;">
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#f4f6f8;width:100%;min-width:100%;">
      <tr>
        <td align="center" style="padding:16px;">
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="max-width:680px;min-width:320px;background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;">
            <tr>
              <td style="padding:18px 20px 12px;">
                <h1 style="margin:0;font-size:22px;line-height:1.2;color:#111827;">BlueOcean Stock Report</h1>
                <p style="margin:10px 0 0;font-size:14px;line-height:1.6;color:#4b5563;">A clean mobile-friendly stock update with color-coded signal cards and compact metrics optimized for Gmail and Outlook.</p>
              </td>
            </tr>
            <tr>
              <td style="padding:0 20px 18px;border-top:1px solid #e5e7eb;">
                <p style="margin:0;font-size:12px;color:#6b7280;line-height:1.5;">Each stock is shown in its own card, with the most important metrics and a short insight summary. Scroll vertically on mobile for the best view.</p>
              </td>
            </tr>
"""

    print("Running unified stock analysis...\n")
    rows = []
    for stock_name, ticker in STOCKS.items():
        try:
            # fetch data and compute scores
            df = fetch_data(ticker)
            df = calculate_indicators(df)

            tech_score = technical_score(df)

            fund_raw = fetch_fundamentals(ticker)
            fund_score = score_fundamentals(fund_raw)

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

            # keep technical reasons from the original trend engine
            score, reasons = calculate_score(df)
            tech_signal = get_signal(score)
            signal = decision(total_score)

            latest = df.iloc[-1]
            reason_text = ", ".join(reasons)
            news_text = ", ".join(headlines[:3])
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
            )

            risk_data = apply_risk_management(signal, total_score, cash=100000, price=latest["close"])

            if "sell" in signal.lower():
                row_color = "#fef2f2"
                text_color = "#991b1b"
                priority = 3
            elif "hold" in signal.lower() or "buy / hold" in signal.lower():
                row_color = "#fffbeb"
                text_color = "#78350f"
                priority = 2
            else:
                row_color = "#ecfdf5"
                text_color = "#0f766e"
                priority = 1

            # card-style HTML for each stock (email-safe)
            row_html = f"""
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 20px;border-radius:12px;background:#ffffff;border:1px solid #e5e7eb;">
                <tr>
                    <td style="padding:14px;">
                        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;">
                            <tr>
                                <td style="vertical-align:top;">
                                    <h3 style="margin:0;font-size:16px;color:#0f172a;line-height:1.2;">{stock_name} <span style="font-size:13px;color:#64748b;">{ticker}</span></h3>
                                    <p style="margin:6px 0 0;font-size:13px;color:#334155;line-height:1.4;">{llm_reason}</p>
                                </td>
                                <td style="width:110px;text-align:right;vertical-align:top;">
                                    <div style="display:inline-block;padding:6px 10px;border-radius:999px;font-weight:700;color:#fff;background:{'#dc2626' if 'sell' in signal.lower() else '#f59e0b' if 'hold' in signal.lower() else '#047857'};">{signal}</div>
                                    <div style="margin-top:8px;font-size:13px;color:#334155;">Score: <strong>{total_score}</strong></div>
                                </td>
                            </tr>
                        </table>
                        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border-collapse:collapse;margin-top:10px;font-size:13px;color:#475569;">
                            <tr>
                                <td style="padding:6px 0;width:50%;"><strong>Close</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['close'],2)}</div></td>
                                <td style="padding:6px 0;width:50%;"><strong>EMA20 / EMA50</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['ema20'],2)} / {round(latest['ema50'],2)}</div></td>
                            </tr>
                            <tr>
                                <td style="padding:6px 0;"><strong>EMA100 / EMA200</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['ema100'],2)} / {round(latest['ema200'],2)}</div></td>
                                <td style="padding:6px 0;"><strong>RSI / ADX</strong><div style="color:#0f172a;margin-top:4px;">{round(latest['rsi'],2)} / {round(latest['adx'],2)}</div></td>
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
                                <td colspan="2" style="padding-top:10px;border-top:1px solid #eef2f7;font-size:13px;color:#475569;"><strong>News:</strong> {news_text or 'No recent headlines.'}</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
            """
            rows.append((priority, total_score, stock_name, row_html))
            print(f"{stock_name} ({ticker}) -> {signal} | Score: {score}")
        except Exception as e:
            error_text = f"Error processing {ticker}: {str(e)}"
            print(error_text)
            import traceback
            traceback.print_exc()
            err_html = f"""
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:12px 20px;border-radius:8px;background:#fff7f7;border:1px solid #f5c2c7;">
                <tr>
                    <td style="padding:12px;color:#721c24;font-size:13px;"><strong>Error:</strong> {error_text}</td>
                </tr>
            </table>
            """
            rows.append((4, 0, ticker, err_html))

    # group rows by priority for clearer sections
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

    # sort each group by score (desc)
    for key in groups:
        groups[key].sort(key=lambda item: item[0], reverse=True)

    buy_count = len(groups["Buy"])
    hold_count = len(groups["Hold"])
    sell_count = len(groups["Sell"])
    err_count = len(groups["Errors"])

    # build summary and section HTML
    from datetime import datetime
    summary_html = f"""
        <tr>
          <td style="padding:16px 20px;border-top:1px solid #e5e7eb;">
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
              <tr>
                <td>
                  <h2 style="margin:0;font-size:16px;">Portfolio Snapshot</h2>
                </td>
                <td style="text-align:right;font-size:13px;color:#475569;">{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</td>
              </tr>
            </table>
            <p style="margin:8px 0 0;font-size:13px;color:#4b5563;">Total symbols: <strong>{len(rows)}</strong>  &bull;  Buy: <strong>{buy_count}</strong>  &bull;  Hold: <strong>{hold_count}</strong>  &bull;  Sell: <strong>{sell_count}</strong></p>
          </td>
        </tr>
    """

    section_html = ""
    if groups["Buy"]:
        section_html += f'<tr><td style="padding:12px 20px 0;"><h2 style="margin:0;font-size:15px;color:#059669;">Buy ({buy_count})</h2></td></tr>'
        for _, _, html_row in groups["Buy"]:
            section_html += f"<tr><td>{html_row}</td></tr>"
    if groups["Hold"]:
        section_html += f'<tr><td style="padding:12px 20px 0;"><h2 style="margin:0;font-size:15px;color:#a16207;">Hold ({hold_count})</h2></td></tr>'
        for _, _, html_row in groups["Hold"]:
            section_html += f"<tr><td>{html_row}</td></tr>"
    if groups["Sell"]:
        section_html += f'<tr><td style="padding:12px 20px 0;"><h2 style="margin:0;font-size:15px;color:#b91c1c;">Sell ({sell_count})</h2></td></tr>'
        for _, _, html_row in groups["Sell"]:
            section_html += f"<tr><td>{html_row}</td></tr>"
    if groups["Errors"]:
        section_html += f'<tr><td style="padding:12px 20px 0;"><h2 style="margin:0;font-size:15px;color:#b91c1c;">Errors ({err_count})</h2></td></tr>'
        for _, _, html_row in groups["Errors"]:
            section_html += f"<tr><td>{html_row}</td></tr>"

    # assemble final email
    report_html += summary_html + section_html
    report_html += """
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
    """

    # send or save report
    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("report.html", "w") as f:
            f.write(report_html)
        print("\\nReport saved to report.html (DRY_RUN enabled)")
    else:
        send_email(report_html)
        print("\\nEmail report sent successfully.")
                        groups["Errors"].append((score, name, html))

        # summary header
        total = len(rows)
        buy_count = len(groups["Buy"]) ; hold_count = len(groups["Hold"]) ; sell_count = len(groups["Sell"]) ; err_count = len(groups["Errors"])
        from datetime import datetime
        summary_html = f"""
                        <tr>
                            <td style=\"padding:12px 20px;\">
                                <table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" role=\"presentation\" style=\"border-collapse:collapse;background:#f8fafc;border-radius:10px;padding:12px;\">
                                    <tr>
                                        <td style=\"font-size:14px;color:#0f172a;font-weight:700;\">Portfolio Snapshot</td>
                                        <td style=\"text-align:right;font-size:13px;color:#475569;\">{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</td>
                                    </tr>
                                    <tr>
                                        <td colspan=\"2\" style=\"padding-top:8px;font-size:13px;color:#334155;\">Total symbols: <strong>{total}</strong> &nbsp;•&nbsp; Buy: <strong>{buy_count}</strong> &nbsp;•&nbsp; Hold: <strong>{hold_count}</strong> &nbsp;•&nbsp; Sell: <strong>{sell_count}</strong></td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
        """
        html_report += summary_html

        # sections
        section_order = [("Buy", "#ecfdf5"), ("Hold", "#fffbeb"), ("Sell", "#fff1f2"), ("Errors", "#fff7f7")]
        for section, bg in section_order:
                items = groups.get(section, [])
                if not items:
                        continue
                html_report += f"""\n            <tr><td style=\"padding:12px 20px;\"><h2 style=\"margin:0;font-size:16px;color:#0f172a;\">{section} ({len(items)})</h2></td></tr>\n        """
                # add each card (sorted by score desc)
                for score, name, card_html in sorted(items, key=lambda x: -x[0]):
                        # wrap card in a table row container for consistent spacing
                        html_report += f"""\n            <tr><td style=\"padding:0;\">{card_html}</td></tr>\n            """

    html_report += """
            <tr>
              <td style="padding:0 20px 18px;">
                <p style="margin:0;font-size:12px;color:#6b7280;line-height:1.5;">Generated by BlueOcean — review before acting on any trade.</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

    send_email(html_report)


if __name__ == "__main__":
    main()
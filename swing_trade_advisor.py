"""
swing_trade_advisor.py

Standalone companion to main.py. Runs a single, open-ended "find me a
high-conviction 3-5 month swing trade" prompt against whichever free LLM
backend main.py already knows how to set up (Groq free tier -> Gemini
2.5 Flash free tier -> local Qwen2.5-1.5B fallback), then emails the
result to the same recipients configured for the stock report
(EMAIL_TO / EMAIL_CC in config.py / the workflow yaml's env vars).

This intentionally reuses main.py's LLM-selection and email-credential
plumbing instead of duplicating it, so both scripts stay in sync with
whatever provider is configured.

LIVE DATA: when Groq is the active backend (GROQ_API_KEY set, the default
first choice), this uses Groq's free "groq/compound" system, which can
autonomously run live web searches before answering -- so it isn't limited
to training-data knowledge the way a plain chat completion is. The email
lists the actual URLs it searched under "Sources checked" so you can spot-
check its claims.

CAVEAT: this is still not a verified real-time trade signal. Web search
results can be a few hours stale, incomplete, or misread by the model, and
if GROQ_API_KEY isn't set the run falls back to Gemini or a local model,
neither of which has any live internet access at all (pure training-data
reasoning in that case). Either way, treat every price level, %, and
"recent" news item as a starting hypothesis to verify against a live
quote/news source yourself -- not investment advice.
"""

import os
import re
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import smtplib
from email.mime.text import MIMEText

import main  # reuses LLM init, email config/credentials, and helpers

# -----------------------------
# Prompt
# -----------------------------
def build_prompt():
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    return f"""Core Objective: Analyze like PRO and use latest data, Identify a high-conviction swing trade opportunity suitable for a 3 to 5-month holding period, based on a robust blend of professional-grade fundamental, technical, and sentiment analysis, utilizing the most current market data. Give only confident stocks only and make sure you consider global data, Indian market news any risks associated with etc

Stock Selection Strategy (Choose One):

Analyze and recommend a stock based on one of the following high-probability strategies for the 3-5 month horizon:

Momentum Breakout: Identify a large-cap or high-quality mid-cap stock that has recently broken out of a multi-month consolidation pattern (e.g., Cup and Handle, Multi-Year Base, or Symmetrical Triangle on a Weekly Chart) with significantly higher-than-average volume. The breakout must signal the start of a new, sustained intermediate-term uptrend.
Event-Driven: Select a stock positioned to gain from a major near-term corporate catalyst (e.g., successful regulatory approval, large contract win, major demerger/spinoff, or an M&A arbitrage play). The expected price impact must be clearly quantifiable within the 3-5 month window.
Technical Swing Trade: Pinpoint a stock in a confirmed, strong secular uptrend (e.g., above the 50-WMA and 200-WMA) that has recently experienced a healthy pullback to a key support level (e.g., 20-WMA, Fibonacci Retracement of 38.2% or 50%, or horizontal support) and shows a clear reversal candlestick pattern.
Fundamental Short-Term Bet: Choose a stock with a compelling valuation that has recently reported exceptionally strong quarterly earnings (both YoY and QoQ growth, with significant beat on analyst consensus) and provided highly positive future guidance. This is a play on the market repricing the stock to reflect the new fundamental reality.

Mandatory Analysis Parameters:
- Fundamental Quality: Excellent Health -- Low Debt-to-Equity (or strong asset quality for financials), High and improving ROCE/ROE. Check for promoter/institutional buying in the last quarter.
- Quarterly Performance: Highly Positive -- Latest Quarterly Net Profit and Revenue growth must exceed the 20% YoY threshold, with positive margin expansion.
- Technical Setup (3-5M View): Strong & Sustainable -- Price must be trading above its 20-week and 50-week Simple Moving Averages (SMAs). Use RSI on the weekly chart (must be trending up but below 70 for entry) and MACD for a bullish crossover confirmation.
- Market/News Sentiment: Overwhelmingly Positive -- Search for recent positive news catalysts (analyst upgrades, sector tailwinds, large orders) that support the intermediate-term view. Check institutional activity (FII/DII) for confidence.
- Risk Management: Pro-Grade Risk/Reward -- The setup must provide a minimum Risk/Reward Ratio of 1:2.5 based on the proposed Stop-Loss (SL) and Target (T1/T2) levels.
- Data Recency: Use stock prices, news, and financial data as of {today_str}.

Provide the analysis for TWO stocks -- one based on the highest conviction strategy from the list above, and the second from a different strategy if possible.

For EACH stock, give: Name of the Stock; How Much Can Invest (% of total trading capital, e.g. 5-10%); Date of Entry (Targeted); Date of Exit (Expected, 3-5 months from entry); Momentum Name (Strategy Type used); Favourable % (Upside Target for 3-5 months); Risk % (Stop-Loss); Expected Profit % (T1); Expected Profit % (T2, optional); Any Top Buyers (Recent FII/DII, if known); Any Recommendations from Brokers (e.g. 'Buy' with target X from a named brokerage, if known).

OUTPUT FORMAT -- respond with ONLY the HTML below, fully filled in, and nothing else (no markdown, no code fences, no commentary before or after):

<table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;border-collapse:collapse;">
<tr style="background:#f8fafc;"><td style="padding:8px 10px;font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;">Parameter</td><td style="padding:8px 10px;font-size:11px;font-weight:700;color:#2563eb;text-transform:uppercase;">Stock A</td><td style="padding:8px 10px;font-size:11px;font-weight:700;color:#2563eb;text-transform:uppercase;">Stock B</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Name of the Stock</td><td style="padding:6px 10px;font-size:13px;font-weight:700;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:13px;font-weight:700;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Allocation (% of capital)</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Entry Date (Targeted)</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Exit Date (Expected)</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Strategy Type</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Upside Target %</td><td style="padding:6px 10px;font-size:12px;font-weight:700;color:#047857;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;font-weight:700;color:#047857;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Stop-Loss %</td><td style="padding:6px 10px;font-size:12px;font-weight:700;color:#dc2626;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;font-weight:700;color:#dc2626;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Target 1 (T1) %</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Target 2 (T2) %</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Recent Top Buyers (FII/DII)</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
<tr><td style="padding:6px 10px;font-size:12px;color:#475569;border-top:1px solid #f1f5f9;">Broker Recommendations</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td><td style="padding:6px 10px;font-size:12px;color:#0f172a;border-top:1px solid #f1f5f9;">FILL</td></tr>
</table>
<div style="margin-top:10px;font-size:12px;color:#475569;line-height:1.6;"><strong>Why these two:</strong> FILL two to three sentences per stock covering the fundamental + technical + sentiment rationale and the key risk to watch.</div>
"""


# -----------------------------
# LLM call (larger token budget than main.py's per-stock reasoning calls,
# since this is one long-form response rather than a short per-stock blurb)
# -----------------------------
def generate_analysis(prompt):
    backend = main.init_llm_generator()
    main.log.info(f"Swing trade advisor using LLM backend: {backend}")

    if backend == "groq":
        # groq/compound is Groq's free, tool-using system: same API key/client
        # as the plain llama model, but it can autonomously trigger a live web
        # search (via Tavily under the hood) when the query needs current
        # info -- which is exactly what this "use latest data" prompt needs.
        # It's slower and uses more of the free-tier quota than a plain chat
        # completion (it may make several web searches per request under the
        # hood), but for a once-a-run email that's a good trade.
        #
        # groq/compound orchestrates via meta-llama/llama-4-scout, whose free
        # tier is capped at 30,000 tokens/minute org-wide. A 429 here is
        # usually just that minute's shared quota being briefly exhausted --
        # not a real outage -- so retry a couple of times with the wait time
        # Groq's own error message reports before giving up on it.
        for attempt in range(3):
            try:
                response = main.groq_client.chat.completions.create(
                    model="groq/compound",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=2200,
                )
                text = response.choices[0].message.content.strip()
                sources = _extract_groq_sources(response)
                return text, sources, True  # True = had live web search available
            except Exception as e:
                main.log.error(
                    f"Groq (compound) swing-trade generation failed "
                    f"(attempt {attempt + 1}/3): {e}"
                )
                if _is_request_too_large(e):
                    # A 413 means this exact payload can't fit regardless of
                    # timing -- retrying it unchanged is guaranteed to fail
                    # again (as seen: attempt 1 got 413, then wasted 2 more
                    # attempts and 10+ seconds turning it into 429s instead).
                    # Stop immediately and let the Gemini fallback below
                    # handle this run rather than burning more of the shared
                    # per-minute token budget on retries that can't succeed.
                    main.log.error(
                        "Request too large for groq/compound -- skipping "
                        "further retries of this payload."
                    )
                    break
                if attempt < 2:
                    wait_s = _parse_groq_retry_seconds(e) or 10
                    main.log.info(f"Retrying groq/compound in {wait_s:.1f}s...")
                    time.sleep(wait_s)

        # groq/compound is still rate-limited/unavailable after retries. Try
        # a real live-search alternative (Gemini + Google Search grounding)
        # before falling back to non-live generation, since that's a
        # genuinely separate free-tier quota from Groq's.
        grounded = _try_gemini_grounded(prompt)
        if grounded is not None:
            return grounded

        # Last resort: plain (non-search) Groq model, so the run still
        # produces *something* rather than giving up entirely.
        try:
            response = main.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=3000,
            )
            return response.choices[0].message.content.strip(), [], False
        except Exception as e2:
            main.log.error(f"Groq fallback (no search) generation also failed: {e2}")

    if backend == "gemini" or main.gemini_client is not None:
        grounded = _try_gemini_grounded(prompt)
        if grounded is not None:
            return grounded
        try:
            response = main.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text.strip(), [], False
        except Exception as e:
            main.log.error(f"Gemini swing-trade generation failed: {e}")

    if backend == "local" and main.llm_pipeline is not None:
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = main.llm_pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            with main.model_lock:
                outputs = main.llm_pipeline(
                    formatted_prompt,
                    max_new_tokens=1200,
                    do_sample=True,
                    temperature=0.4,
                    top_k=50,
                    top_p=0.95,
                )
            generated_text = outputs[0]["generated_text"]
            text = generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            return text, [], False
        except Exception as e:
            main.log.error(f"Local swing-trade generation failed: {e}")

    return None, [], False


def _is_request_too_large(exc):
    """True for Groq's 413 'Request Entity Too Large' -- a payload-size
    failure, not a rate limit, so unlike a 429 it can't be fixed by waiting
    and retrying the same request."""
    msg = str(exc)
    return "413" in msg or "request_too_large" in msg or "Request Entity Too Large" in msg


def _parse_groq_retry_seconds(exc):
    """Groq's 429 body includes a 'Please try again in 7.342s' hint -- pull
    that out so retries wait the right amount instead of guessing."""
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.5  # small buffer
        except ValueError:
            return None
    return None


def _try_gemini_grounded(prompt):
    """
    Genuine live-search fallback: Gemini's free tier supports real Google
    Search grounding (separate free-tier quota from Groq entirely), so this
    is a real second live-data path, not just another training-data model.
    Returns (text, sources, True) on success, or None if Gemini isn't
    configured or the call fails -- callers should fall through to their
    next option in that case.
    """
    if main.gemini_client is None:
        return None
    try:
        from google.genai import types
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        response = main.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(tools=[grounding_tool]),
        )
        sources = []
        try:
            for candidate in response.candidates:
                gm = getattr(candidate, "grounding_metadata", None)
                for chunk in (getattr(gm, "grounding_chunks", None) or []):
                    web = getattr(chunk, "web", None)
                    if web and web.uri and (web.title, web.uri) not in sources:
                        sources.append((web.title or web.uri, web.uri))
        except Exception as e:
            main.log.warning(f"Could not extract Gemini grounding sources: {e}")
        # Only report this as "live" if grounding actually fired -- Gemini
        # may decide a query doesn't need a search and skip it.
        used_live = bool(sources)
        return response.text.strip(), sources, used_live
    except Exception as e:
        main.log.error(f"Gemini grounded (live search) generation failed: {e}")
        return None


def _extract_groq_sources(response):
    """
    Pulls (title, url) pairs out of groq/compound's executed_tools field so
    the email can show what was actually searched -- lets the reader spot-
    check the model's claims instead of taking them on faith. Defensive
    about attribute-vs-dict access since SDK response objects vary.
    """
    def _get(obj, key):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    sources = []
    try:
        message = response.choices[0].message
        for tool in (_get(message, "executed_tools") or []):
            search_results = _get(tool, "search_results")
            for r in (_get(search_results, "results") or []):
                url = _get(r, "url")
                title = _get(r, "title") or url
                if url and (title, url) not in sources:
                    sources.append((title, url))
    except Exception as e:
        main.log.warning(f"Could not extract Groq search sources: {e}")
    return sources


def _strip_code_fences(text):
    """Models occasionally wrap the requested HTML in ```html ... ``` even
    when told not to -- strip that off so it renders as HTML, not literal text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


# -----------------------------
# Email (reuses main.py's credentials/config, custom subject + wrapper
# since this isn't the daily stock report)
# -----------------------------
def _build_sources_html(sources):
    if not sources:
        return ""
    items = "".join(
        f'<div style="margin:4px 0 0;font-size:11px;">'
        f'<a href="{url}" style="color:#2563eb;text-decoration:none;">{title}</a></div>'
        for title, url in sources[:12]
    )
    return f"""
        <div style="margin-top:10px;padding-top:10px;border-top:1px dashed #e5e7eb;">
          <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;">Sources checked (live web search)</div>
          {items}
        </div>
    """


def build_email_html(analysis_html, today_str, sources, used_live_search):
    if used_live_search:
        disclaimer = (
            "Generated using Groq's compound model, which can run live web searches -- see "
            "\"Sources checked\" above for what it actually looked at. Search results can still be "
            "incomplete, out of date by a few hours, or misread by the model, so confirm the key "
            "prices/dates/news against a live source before acting. Not investment advice."
        )
    else:
        disclaimer = (
            "Generated by an LLM with no live market/internet access for this run -- prices, dates, "
            "\"recent\" news and broker calls above are model output and are NOT verified against a "
            "live feed. Confirm every figure against a real-time quote/news source before acting. "
            "Not investment advice."
        )

    sources_html = _build_sources_html(sources)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily Swing Trade Idea</title>
<style>
  body {{ margin:0; padding:0; background:#f4f6f8; }}
  table {{ border-collapse:collapse !important; }}
  @media screen and (max-width:600px) {{
    .email-container {{ width:100% !important; max-width:100% !important; border-radius:0 !important; }}
    .email-padding {{ padding-left:14px !important; padding-right:14px !important; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background:#f4f6f8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#111827;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#f4f6f8;width:100%;">
    <tr>
      <td align="center" style="padding:16px;" class="email-padding">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation" class="email-container" style="max-width:680px;min-width:280px;background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;">
          <tr>
            <td style="background:linear-gradient(135deg,#2563eb,#1d4ed8);padding:4px;font-size:0;line-height:0;">&nbsp;</td>
          </tr>
          <tr>
            <td style="padding:18px 20px 6px;" class="email-padding">
              <h1 style="margin:0;font-size:20px;color:#111827;">📈 Daily Swing Trade Idea (3-5 Month Horizon)</h1>
              <p style="margin:8px 0 0;font-size:13px;color:#6b7280;">Generated {today_str} · {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p IST")}{' · live web search used' if used_live_search else ''}</p>
            </td>
          </tr>
          <tr>
            <td style="padding:0 20px 16px;" class="email-padding">
              {analysis_html}
              {sources_html}
            </td>
          </tr>
          <tr>
            <td style="padding:12px 20px 18px;border-top:1px solid #e5e7eb;" class="email-padding">
              <p style="margin:0;font-size:11px;color:#9ca3af;line-height:1.5;">{disclaimer}</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_swing_trade_email(html_body):
    if not all([main.EMAIL_FROM, main.EMAIL_PASSWORD, main.EMAIL_TO]):
        main.log.error(
            "Email credentials not found. Please set EMAIL_FROM, EMAIL_PASSWORD, "
            "and EMAIL_TO (the same env vars main.py uses)."
        )
        return False

    to_recipients = main.parse_email_list(main.EMAIL_TO)
    cc_recipients = main.parse_email_list(getattr(main, "EMAIL_CC", "") or "")

    if not to_recipients:
        main.log.error("No valid TO recipients found in EMAIL_TO.")
        return False

    now_ist = datetime.now(ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Kolkata"))
    time_str = now_ist.strftime("%I:%M %p IST")
    subject = f"📈 Daily Swing Trade Idea - {main.get_date_with_suffix(now_ist)} · {time_str}"

    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = main.EMAIL_FROM
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    all_recipients = to_recipients + cc_recipients

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(main.EMAIL_FROM, main.EMAIL_PASSWORD)
        server.sendmail(main.EMAIL_FROM, all_recipients, msg.as_string())
        server.quit()
        main.log.info("Swing trade email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        main.log.error(
            "SMTP Authentication Error: check EMAIL_FROM/EMAIL_PASSWORD "
            "(use a Gmail App Password, not the account password)."
        )
    except Exception as e:
        main.log.error(f"Failed to send swing trade email: {e}")
        traceback.print_exc()
    return False


def run():
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    prompt = build_prompt()

    analysis, sources, used_live_search = generate_analysis(prompt)
    if not analysis:
        main.log.error(
            "No LLM backend produced output (no GROQ_API_KEY/GOOGLE_API_KEY set "
            "and local model unavailable/failed). Aborting without sending an email."
        )
        sys.exit(1)

    if not used_live_search and os.getenv("REQUIRE_LIVE_DATA", "true").lower() == "true":
        # Only groq/compound actually hits the live web. Every other path
        # (Groq's plain llama fallback, Gemini, or the local model) is pure
        # training-data reasoning with no real-time prices/news -- exactly
        # the stale output this run is meant to avoid. Refuse to email it
        # rather than silently sending non-verified figures. Set
        # REQUIRE_LIVE_DATA=false to explicitly allow a stale-data fallback
        # email instead of aborting.
        main.log.error(
            "Live web search was not used for this run (Groq's live-search "
            "model was unavailable or the backend fell back to Gemini/local), "
            "so the output would only reflect stale training-data prices/news. "
            "Aborting without sending an email. Set REQUIRE_LIVE_DATA=false to "
            "override and allow a clearly-labeled stale-data email instead."
        )
        sys.exit(1)

    analysis_html = _strip_code_fences(analysis)
    email_html = build_email_html(analysis_html, today_str, sources, used_live_search)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        with open("swing_trade_report.html", "w") as f:
            f.write(email_html)
        main.log.info("DRY_RUN enabled -- wrote swing_trade_report.html instead of emailing.")
        return

    send_swing_trade_email(email_html)


if __name__ == "__main__":
    run()

# -----------------------------
# Real-time grounding status
# -----------------------------
# Groq path: wired in above via model="groq/compound" (free tier, same API
# key). It autonomously runs live web searches (Tavily-backed) when the
# query needs current info, and executed_tools[].search_results gives back
# the actual URLs used -- surfaced in the email under "Sources checked".
# This is the primary path since main.init_llm_generator() tries Groq first.
#
# Gemini / local fallback paths: still plain, non-grounded chat completions
# with no live internet access. If Groq isn't configured (no GROQ_API_KEY)
# and the run falls back to Gemini, real-time grounding could be added via
# Gemini's Google Search grounding tool -- but the exact tool schema differs
# across google-genai SDK versions, so it's deliberately not wired in here;
# verify current syntax against Gemini API docs before enabling it.

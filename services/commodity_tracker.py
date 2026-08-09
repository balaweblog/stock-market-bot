import requests
import statistics
from datetime import datetime
import yfinance as yf

from models.recommendation_logic import derive_commodity_buy_levels, resolve_recommended_commodity_entry

try:
    from models.market_context import get_resilient_session
except ImportError:
    def get_resilient_session():
        return None


class CommodityTracker:
    BASE_URL = "https://api.gold-api.com"

    # Adjust to Indian retail market price
    GOLD_MARKUP = 1.15
    SILVER_MARKUP = 1.36
    GOLD_KARAT = 22

    def __init__(self, api_key=None):
        # Cache so a single tracker run makes one FX call instead of three
        # (fetch_current_prices + fetch_history for gold + fetch_history for
        # silver each called usd_to_inr() independently). Beyond the wasted
        # requests, each call could return a slightly different live rate;
        # caching keeps "current" and "history" priced off the same USD/INR
        # number for internal consistency.
        self._usd_inr_cache = None

    # ---------------- API ----------------
    def call_api(self, endpoint):
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}, {response.text}")
        return response.json()

    def adjust_gold_purity(self, price):
        return round(price * self.GOLD_KARAT / 24, 2)

    def oz_to_gram(self, price):
        return round(price / 31.1035, 2)

    # ---------------- Date Formatting ----------------
    def get_day_suffix(self, day):
        if 11 <= day <= 13:
            return "th"
        if day % 10 == 1:
            return "st"
        if day % 10 == 2:
            return "nd"
        if day % 10 == 3:
            return "rd"
        return "th"

    def format_date_short(self, date_str):
        dt = datetime.strptime(date_str, "%Y%m%d")
        suffix = self.get_day_suffix(dt.day)
        return f"{dt.day}{suffix} {dt.strftime('%b (%a)')}"

    # ---------------- FX ----------------
    def usd_to_inr(self):
        if self._usd_inr_cache is not None:
            return self._usd_inr_cache
        url = "https://open.er-api.com/v6/latest/USD"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "rates" not in data:
            raise Exception(f"FX API Error: {data}")
        self._usd_inr_cache = data["rates"]["INR"]
        return self._usd_inr_cache

    # ---------------- Prices ----------------
    def fetch_current_prices(self):
        usd_inr = self.usd_to_inr()

        gold = self.call_api("price/XAU")
        silver = self.call_api("price/XAG")

        gold_price = gold["price"] * usd_inr
        silver_price = silver["price"] * usd_inr

        gold_per_gram = self.oz_to_gram(gold_price)
        silver_per_gram = self.oz_to_gram(silver_price)

        gold_per_gram = round(gold_per_gram * self.GOLD_MARKUP, 2)
        silver_per_gram = round(silver_per_gram * self.SILVER_MARKUP, 2)

        gold_per_gram = self.adjust_gold_purity(gold_per_gram)

        return {
            "gold": {"current": gold_per_gram},
            "silver": {"current": silver_per_gram},
        }

    # ---------------- Historical ----------------
    def fetch_history(self, symbol, days=7, anchor_price=None):
        history = []
        usd_inr = self.usd_to_inr()
        ticker = "GC=F" if symbol == "XAU" else "SI=F"

        # Pad the requested window for weekends/holidays -- requesting
        # exactly days+2 calendar days only yields ~days*0.7 trading rows,
        # so a "30-day" history was quietly coming back as ~22 rows.
        df = yf.download(
            ticker,
            period=f"{int(days * 1.6) + 5}d",
            interval="1d",
            progress=False,
            session=get_resilient_session(),
        )

        close_series = df["Close"].squeeze().dropna().tail(days)

        for date, close in close_series.items():
            price_inr = close.item() * usd_inr if hasattr(close, "item") else float(close) * usd_inr
            price_per_gram = self.oz_to_gram(price_inr)

            if symbol == "XAU":
                price_per_gram *= self.GOLD_MARKUP
                price_per_gram = self.adjust_gold_purity(price_per_gram)
            else:
                price_per_gram *= self.SILVER_MARKUP

            price_per_gram = round(price_per_gram, 2)
            history.append({
                "date": self.format_date_short(date.strftime("%Y%m%d")),
                "price": price_per_gram,
            })

        # The history above is priced off Yahoo futures (GC=F/SI=F), while
        # the headline "current" price comes from the gold-api.com spot feed.
        # Futures and spot diverge (contango/backwardation) and the futures
        # "close" can be a stale prior-session print, so without reconciling
        # them the last history point wouldn't match the headline number and
        # buy levels (a % of current price) would sit on a trend line from a
        # different instrument. Anchor the series to the live spot price,
        # preserving the shape/trend of the futures data while keeping it
        # consistent with the number shown as "current price".
        if anchor_price is not None and history and history[-1]["price"]:
            scale = anchor_price / history[-1]["price"]
            for row in history:
                row["price"] = round(row["price"] * scale, 2)

        return history

    def calculate_pct_change(self, history):
        prev = None
        for row in history:
            if prev is None:
                row["change"] = 0.0
            else:
                row["change"] = round(((row["price"] - prev) / prev) * 100, 2)
            prev = row["price"]
        return history

    # ---------------- Sparkline ----------------
    def generate_sparkline(self, prices):
        blocks = "▁▂▃▄▅▆▇█"
        min_p, max_p = min(prices), max(prices)
        if min_p == max_p:
            return "▅" * len(prices)
        spark = ""
        for p in prices:
            idx = int((p - min_p) / (max_p - min_p) * (len(blocks) - 1))
            spark += blocks[idx]
        return spark

    # ---------------- Aggregation ----------------
    def get_commodity_data(self):
        current = self.fetch_current_prices()

        # Fetch a full 30-day window for each metal. The sparkline needs the
        # whole 30 days; the price table only ever shows the most recent 7,
        # so we keep both around rather than fetching twice.
        gold_history_30d = self.calculate_pct_change(
            self.fetch_history("XAU", days=30, anchor_price=current["gold"]["current"])
        )
        silver_history_30d = self.calculate_pct_change(
            self.fetch_history("XAG", days=30, anchor_price=current["silver"]["current"])
        )

        current["gold"]["history"] = gold_history_30d[-7:]
        current["silver"]["history"] = silver_history_30d[-7:]
        current["gold"]["sparkline_history"] = gold_history_30d
        current["silver"]["sparkline_history"] = silver_history_30d
        current["gold"]["change"] = gold_history_30d[-1]["change"]
        current["silver"]["change"] = silver_history_30d[-1]["change"]

        return current

    # Floor on the volatility-derived band width, as a fraction of price.
    # Prevents the range from collapsing to (near-)zero width during an
    # unusually quiet stretch, which would misleadingly read as a
    # high-conviction point estimate rather than a momentum-based guess.
    MIN_TARGET_BAND_PCT = 0.5

    def build_trade_plan(self, current_price, history, levels, sparkline_history=None):
        entry_low = round(min(levels["patient_entry"], levels["optimal_entry"], levels["aggressive_entry"]), 2)
        entry_high = round(max(levels["patient_entry"], levels["optimal_entry"], levels["aggressive_entry"]), 2)

        # Band width comes from realized volatility (stdev of daily % change)
        # over the lookback window, not a fixed 2%/1.5% constant -- a quiet
        # week and a volatile week now produce different-width ranges instead
        # of an identical single point. Prefer the longer sparkline window
        # (up to 30 days) when available so the estimate isn't based on just
        # the 7 rows kept for the on-screen history table; fall back to
        # `history` so this still works wherever only 7 days are passed in.
        vol_source = sparkline_history if sparkline_history else history
        daily_changes = [row["change"] for row in vol_source if "change" in row] if vol_source else []
        daily_vol_pct = statistics.pstdev(daily_changes) if len(daily_changes) >= 2 else 0.0
        band_pct = max(daily_vol_pct, self.MIN_TARGET_BAND_PCT)

        stop_base = min(entry_low, current_price)
        target_base = max(entry_high, current_price)
        stop_loss = round(stop_base * (1 - band_pct / 100), 2)
        target_mid = round(target_base * 1.02, 2)
        target_band = round(target_base * (band_pct / 100), 2)
        target_low = round(target_mid - target_band, 2)
        target_high = round(target_mid + target_band, 2)

        if not history:
            bias = "Neutral"
        else:
            latest_change = history[-1].get("change", 0)
            prev_change = history[-2].get("change", 0) if len(history) > 1 else 0
            if latest_change >= 0 and latest_change >= prev_change:
                bias = "Bullish"
            elif latest_change < 0 and latest_change <= prev_change:
                bias = "Bearish"
            else:
                bias = "Neutral"

        # Risk:reward measured against the near (conservative) edge of the
        # target band, not the midpoint -- keeps this ratio from overstating
        # the reward side now that target is a range rather than a point.
        risk_reward = round((target_low - current_price) / max(current_price - stop_loss, 0.01), 2)

        # Per-tier risk/reward -- stop_loss/target_low above are shared
        # across all three entry prices, which structurally favors whichever
        # tier sits lowest (patient) and penalizes whichever sits highest
        # (aggressive): patient is close to stop_loss (small risk) and far
        # from target_low (big reward); aggressive is the reverse. Computed
        # here so the recommendation can be checked against it below instead
        # of recommending purely on momentum and leaving that mismatch to
        # surface only in the rendered report.
        risk_reward_by_entry = {}
        for tier in ("patient_entry", "optimal_entry", "aggressive_entry"):
            entry_price = levels.get(tier)
            if entry_price is None:
                continue
            risk = entry_price - stop_loss
            reward = target_low - entry_price
            risk_reward_by_entry[tier] = round(reward / risk, 2) if risk > 0 else None

        # Mutates `levels` in place (recommended_entry/label/buy_level) so
        # every caller already holding a reference to this dict -- the
        # commodity card, the Action Plan table -- picks up the corrected
        # recommendation without needing to re-fetch it.
        resolve_recommended_commodity_entry(levels, risk_reward_by_entry)

        return {
            "bias": bias,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop_loss": stop_loss,
            "target": target_mid,
            "target_low": target_low,
            "target_high": target_high,
            "risk_reward": risk_reward,
            "risk_reward_by_entry": risk_reward_by_entry,
        }

    def derive_buy_levels(self, current_price, history):
        return derive_commodity_buy_levels(current_price, history)

    # ---------------- Buy signal badge ----------------
    def buy_signal(self, history, label):
        if len(history) < 3:
            return ""

        recent = [row["change"] for row in history[-3:]]
        latest, prev, older = recent[-1], recent[-2], recent[-3]

        score = 0
        if latest <= -1.5:
            score += 4
        if latest <= -2.5:
            score += 4
        if prev <= -1.0:
            score += 2
        if older <= -1.0:
            score += 1
        if latest < prev:
            score += 2
        if latest < 0:
            score += 1

        if score >= 8:
            return (
                '<span style="display:inline-block;margin-left:6px;padding:4px 10px;'
                'border-radius:999px;background:#dcfce7;color:#166534;'
                f'font-weight:700;font-size:12px;">Buy {label}</span>'
            )
        return ""

    # ---------------- HTML helpers ----------------
    # Palette matches the report's own header/banner theme (navy + gold),
    # so the commodity cards read as one continuous "research note" rather
    # than a visually separate widget bolted onto the stock sections.
    _METAL_THEME = {
        "gold":   {"accent": "#B08D57", "accent_dark": "#8A6B3E", "panel_bg": "#FBF7EF", "panel_border": "#B08D5755"},
        "silver": {"accent": "#7C8798", "accent_dark": "#475569", "panel_bg": "#F8FAFC", "panel_border": "#7C879855"},
    }

    def _metal_theme(self, ticker_label):
        return self._METAL_THEME["gold"] if str(ticker_label).upper().startswith("XAU") else self._METAL_THEME["silver"]

    def _kpi_tile_html(self, label, value_html, value_color="#14213D", font_size=14):
        """One stat tile in the KPI grid -- uppercase label over a bold
        value, boxed and lightly shaded so the six key numbers (price,
        trend, bias, entry zone, stop, target) scan as a unified dashboard
        instead of a stack of loose label/value lines."""
        return f"""
                                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#FAFAF8;border:1px solid #EDEAE2;border-radius:6px;">
                                        <tr>
                                            <td style="padding:10px 12px;">
                                                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;color:#8A8F9C;">{label}</div>
                                                <div style="margin-top:5px;font-size:{font_size}px;font-weight:800;color:{value_color};line-height:1.3;word-break:break-word;">{value_html}</div>
                                            </td>
                                        </tr>
                                    </table>"""

    def _entry_chip_html(self, tier_label, price, highlighted, accent):
        """One Patient/Optimal/Aggressive entry chip. The tier that matches
        the resolved recommendation is filled solid in the metal's accent
        color; the other two stay as plain outlined chips -- so the reader's
        eye lands on the ONE price that's actually being recommended instead
        of three equal-weight numbers."""
        if highlighted:
            style = f"background:{accent};border:1px solid {accent};color:#ffffff;"
            label_style = "color:#ffffff;opacity:0.85;"
        else:
            style = "background:#ffffff;border:1px solid #E5E7EB;color:#475569;"
            label_style = "color:#8A8F9C;"
        return f"""<div style="display:inline-block;margin:8px 8px 0 0;padding:6px 12px;border-radius:6px;{style}">
                                        <div style="font-size:9px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;{label_style}">{tier_label}</div>
                                        <div style="margin-top:2px;font-size:13px;font-weight:800;">&#8377;{price:.2f}</div>
                                    </div>"""

    def _history_rows_html(self, history):
        rows = ""
        for i, row in enumerate(reversed(history)):
            is_up = row["change"] >= 0
            color = "#047857" if is_up else "#dc2626"
            arrow = "&#9650;" if is_up else "&#9660;"
            sign = "+" if row["change"] > 0 else ""
            bg = "#ffffff" if i % 2 == 0 else "#FAFAF8"
            rows += (
                f'<tr style="background:{bg};">'
                f'<td style="padding:8px 10px;font-size:12px;color:#14213D;border-bottom:1px solid #F1F1EC;">{row["date"]}</td>'
                f'<td align="right" style="padding:8px 10px;font-size:12px;font-family:\'SF Mono\',Menlo,Consolas,\'Courier New\',monospace;color:#14213D;border-bottom:1px solid #F1F1EC;">&#8377;{row["price"]:.2f}</td>'
                f'<td align="right" style="padding:8px 10px;font-size:12px;font-weight:700;color:{color};border-bottom:1px solid #F1F1EC;white-space:nowrap;">{arrow} {sign}{row["change"]:.2f}%</td>'
                '</tr>'
            )
        return rows

    def fetch_commodity_macro_snapshot(self):
        """
        One-per-run DXY + 10Y yield snapshot, cached on the instance since
        both gold and silver cards want the same numbers. Gold/silver
        typically move opposite the dollar and are sensitive to rate
        expectations, but the report currently shows neither -- this is the
        commodity-side counterpart to fetch_macro_snapshot() in
        stock_controller.py. Returns None for a leg that fails to fetch
        rather than raising, so the card just omits that note instead of
        breaking the report.
        """
        if getattr(self, "_macro_cache", None) is not None:
            return self._macro_cache

        def _last_two_closes(ticker_symbol):
            try:
                hist = yf.Ticker(ticker_symbol).history(period="5d")
                closes = hist["Close"].dropna()
                if len(closes) < 2:
                    return None
                latest, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
                change_pct = round((latest - prev) / prev * 100, 2) if prev else None
                return {"level": round(latest, 2), "change_pct": change_pct}
            except Exception:
                return None

        self._macro_cache = {
            "dollar_index": _last_two_closes("DX-Y.NYB"),
            "treasury_10y": _last_two_closes("^TNX"),
        }
        return self._macro_cache

    def _macro_note_for_commodity(self):
        macro = self.fetch_commodity_macro_snapshot()
        dxy = macro.get("dollar_index")
        if dxy and dxy.get("change_pct") is not None:
            direction = "up" if dxy["change_pct"] > 0 else "down"
            read = "a headwind" if direction == "up" else "supportive"
            return (
                f"&#128181; DXY {direction} {abs(dxy['change_pct']):.1f}% "
                f"({dxy['level']}) — {read} for gold/silver."
            )
        y10 = macro.get("treasury_10y")
        if y10 and y10.get("change_pct") is not None:
            direction = "up" if y10["change_pct"] > 0 else "down"
            read = "raises the opportunity cost of holding metals" if direction == "up" else "lowers the opportunity cost of holding metals"
            return (
                f"&#128200; 10Y yield {direction} {abs(y10['change_pct']):.1f}% "
                f"({y10['level']}%) — {read}."
            )
        return None

    def _commodity_card_html(self, name, ticker_label, current_price, change, history, levels, plan, sparkline_history=None):
        """Renders one commodity card using the exact same table/card structure as stock cards."""
        buy_signal_html = self.buy_signal(history, name)
        change_color = "#047857" if change >= 0 else "#dc2626"
        change_bg    = "#dcfce7" if change >= 0 else "#fee2e2"
        change_sign  = "+" if change > 0 else ""
        change_arrow = "&#9650;" if change >= 0 else "&#9660;"
        history_rows = self._history_rows_html(history)
        theme = self._metal_theme(ticker_label)
        accent, accent_dark = theme["accent"], theme["accent_dark"]

        bias_color = "#047857" if plan["bias"] == "Bullish" else "#dc2626" if plan["bias"] == "Bearish" else "#64748b"
        macro_note = self._macro_note_for_commodity()
        macro_html = (
            f'<div style="margin-top:12px;padding:9px 12px;border-radius:6px;background:#f8fafc;'
            f'border:1px solid #e2e8f0;font-size:12px;color:#475569;">{macro_note}</div>'
        ) if macro_note else ""

        # Which entry tier the recommendation actually landed on, so the
        # matching chip below can be visually singled out instead of all
        # three reading as equally-weighted options.
        rec_label = str(levels.get("recommended_entry_label", "")).lower()

        # KPI grid -- Current Price / Trend / Bias on row one, Entry Zone /
        # Stop Loss / Target on row two. Same six numbers the old stacked
        # layout showed, now scannable as a dashboard instead of a list.
        kpi_grid_html = f"""
                            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:16px;">
                                <tr>
                                    <td width="50%" style="padding:0 4px 8px 0;">{self._kpi_tile_html("Current Price", f'&#8377;{current_price:.2f}<div style="margin-top:2px;font-size:10px;font-weight:500;color:#8A8F9C;">per gram</div>')}</td>
                                    <td width="50%" style="padding:0 0 8px 4px;">{self._kpi_tile_html("Bias", plan['bias'], value_color=bias_color)}</td>
                                </tr>
                                <tr>
                                    <td width="33.33%" style="padding:0 4px 0 0;">{self._kpi_tile_html("Entry Zone", f"&#8377;{plan['entry_low']:.2f} &ndash; &#8377;{plan['entry_high']:.2f}", font_size=13)}</td>
                                    <td width="33.33%" style="padding:0 4px 0 4px;">{self._kpi_tile_html("Stop Loss", f"&#8377;{plan['stop_loss']:.2f}", value_color="#dc2626")}</td>
                                    <td width="33.34%" style="padding:0 0 0 4px;">{self._kpi_tile_html("Target", f"&#8377;{plan['target_low']:.2f} &ndash; &#8377;{plan['target_high']:.2f}", value_color="#047857", font_size=13)}</td>
                                </tr>
                            </table>"""

        buy_levels_html = f"""
                            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:12px;border-radius:8px;background:{theme['panel_bg']};border:1px solid {theme['panel_border']};">
                                <tr>
                                    <td style="padding:12px 14px 14px;">
                                        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;color:{accent_dark};">Buy Levels</div>
                                        <div style="margin-top:6px;font-size:13px;color:#14213D;">
                                            Recommended <strong>{levels['recommended_entry_label']}</strong>:
                                            <span style="margin-left:2px;font-size:16px;font-weight:800;color:#047857;">&#8377;{levels['recommended_buy_level']:.2f}</span>
                                        </div>
                                        <div>
                                            {self._entry_chip_html("Patient", levels['patient_entry'], "patient" in rec_label, accent)}
                                            {self._entry_chip_html("Optimal", levels['optimal_entry'], "optimal" in rec_label, accent)}
                                            {self._entry_chip_html("Aggressive", levels['aggressive_entry'], "aggressive" in rec_label, accent)}
                                        </div>
                                    </td>
                                </tr>
                            </table>"""

        price_history_html = f"""
                            <div style="margin-top:16px;padding-top:14px;border-top:1px solid #EDEAE2;">
                                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;color:#8A8F9C;">Price History &middot; Last {len(history)} Days</div>
                                <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:8px;border-collapse:collapse;border:1px solid #EDEAE2;border-radius:6px;overflow:hidden;">
                                    <tr style="background:#14213D;">
                                        <th align="left" style="padding:8px 10px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#EFEAE0;">Date</th>
                                        <th align="right" style="padding:8px 10px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#EFEAE0;">Price</th>
                                        <th align="right" style="padding:8px 10px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#EFEAE0;">Change</th>
                                    </tr>
                                    {history_rows}
                                </table>
                            </div>"""

        return f"""
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin:14px 0;border-radius:10px;background:#ffffff;border:1px solid #DAD5CB;overflow:hidden;">
                <tr>
                    <td style="height:4px;line-height:4px;font-size:0;background:{accent};">&nbsp;</td>
                </tr>
                <tr>
                    <td style="padding:18px 20px 20px;">
                        <div>
                            <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{accent};">{ticker_label}</div>
                            <h3 style="margin:4px 0 0;font-family:Georgia,'Times New Roman',serif;font-size:19px;font-weight:400;color:#14213D;">{name}</h3>
                            <div style="margin-top:8px;">
                                <span style="display:inline-block;padding:5px 12px;border-radius:999px;font-weight:700;font-size:13px;background:{change_bg};color:{change_color};">{change_arrow} {change_sign}{change}%</span>{buy_signal_html}
                            </div>
                        </div>
                        {kpi_grid_html}
                        {buy_levels_html}
                        {macro_html}
                        {price_history_html}
                    </td>
                </tr>
            </table>"""

    # ---------------- HTML ----------------
    def generate_html(self):
        data = self.get_commodity_data()
        gold   = data["gold"]
        silver = data["silver"]

        gold_levels   = self.derive_buy_levels(gold["current"],   gold["history"])
        silver_levels = self.derive_buy_levels(silver["current"], silver["history"])
        gold_plan   = self.build_trade_plan(gold["current"],   gold["history"],   gold_levels,
                                             sparkline_history=gold.get("sparkline_history"))
        silver_plan = self.build_trade_plan(silver["current"], silver["history"], silver_levels,
                                             sparkline_history=silver.get("sparkline_history"))

        gold_card = self._commodity_card_html(
            name="Gold (22K)", ticker_label="XAU/INR",
            current_price=gold["current"], change=gold["change"],
            history=gold["history"], levels=gold_levels, plan=gold_plan,
            sparkline_history=gold.get("sparkline_history"),
        )
        silver_card = self._commodity_card_html(
            name="Silver", ticker_label="XAG/INR",
            current_price=silver["current"], change=silver["change"],
            history=silver["history"], levels=silver_levels, plan=silver_plan,
            sparkline_history=silver.get("sparkline_history"),
        )

        return f"""
            {self._section_banner_html(2)}
            {gold_card}
            {silver_card}"""

    def _section_banner_html(self, count):
        """Section banner matching the report's own navy/gold masthead
        styling, so 'Commodities' reads as a section of the same research
        note rather than a visually separate block bolted on afterward."""
        return f"""
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:16px;">
                <tr>
                    <td style="padding:0 0 4px;border-bottom:2px solid #B08D57;">
                        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#B08D57;">Precious Metals &amp; Trade Levels</div>
                        <h2 style="margin:4px 0 8px;font-family:Georgia,'Times New Roman',serif;font-size:18px;font-weight:400;color:#14213D;">Commodities <span style="font-size:13px;color:#8A8F9C;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">({count})</span></h2>
                    </td>
                </tr>
            </table>"""


if __name__ == "__main__":
    tracker = CommodityTracker()
    print(tracker.generate_html())
import requests
from datetime import datetime, timedelta
import yfinance as yf

from recommendation_logic import derive_commodity_buy_levels

class CommodityTracker:
    BASE_URL = "https://api.gold-api.com"

    # Adjust to Indian retail market price
    GOLD_MARKUP = 1.15
    SILVER_MARKUP = 1.36
    GOLD_KARAT = 22

    def __init__(self, api_key=None):
       pass

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

    # ---------------- Prices ----------------
    def fetch_current_prices(self):
        usd_inr = self.usd_to_inr()

        gold = self.call_api("price/XAU")
        silver = self.call_api("price/XAG")

        gold_price = gold["price"] * usd_inr
        silver_price = silver["price"] * usd_inr

        # Convert ounce -> gram
        gold_per_gram = self.oz_to_gram(gold_price)
        silver_per_gram = self.oz_to_gram(silver_price)

        gold_per_gram = round(gold_per_gram * self.GOLD_MARKUP, 2)
        silver_per_gram = round(silver_per_gram * self.SILVER_MARKUP, 2)

        gold_per_gram = self.adjust_gold_purity(gold_per_gram)

        return {
            "gold": {"current": gold_per_gram},
            "silver": {"current": silver_per_gram}
        }
    def usd_to_inr(self):
        url = "https://open.er-api.com/v6/latest/USD"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if "rates" not in data:
            raise Exception(f"FX API Error: {data}")

        return data["rates"]["INR"]
    # ---------------- Historical ----------------
    def fetch_history(self, symbol, days=7):
        history = []

        usd_inr = self.usd_to_inr()

        ticker = "GC=F" if symbol == "XAU" else "SI=F"

        df = yf.download(
            ticker,
            period=f"{days + 2}d",
            interval="1d",
            progress=False
        )

        close_series = df["Close"].squeeze().dropna().tail(days)


        for date, close in close_series.items():
            price_inr = close.item() * usd_inr if hasattr(close, "item") else float(close) * usd_inr
            price_per_gram = self.oz_to_gram(price_inr)

            # Adjust to Indian retail market
            if symbol == "XAU":
                price_per_gram *= self.GOLD_MARKUP  # Gold markup
                price_per_gram = self.adjust_gold_purity(price_per_gram)
            else:
                price_per_gram *= self.SILVER_MARKUP   # Silver markup

            price_per_gram = round(price_per_gram, 2)

            history.append({
                "date": self.format_date_short(date.strftime("%Y%m%d")),
                "price": price_per_gram
            })

        return history

    def calculate_pct_change(self, history):
        prev = None

        for row in history:
            if prev is None:
                row["change"] = 0
            else:
                row["change"] = round(
                    ((row["price"] - prev) / prev) * 100, 2
                )
            prev = row["price"]

        return history

    # ---------------- Sparkline ----------------
    def generate_sparkline(self, prices):
        blocks = "▁▂▃▄▅▆▇█"

        min_p = min(prices)
        max_p = max(prices)

        if min_p == max_p:
            return "▅" * len(prices)

        spark = ""
        for p in prices:
            normalized = (p - min_p) / (max_p - min_p)
            index = int(normalized * (len(blocks) - 1))
            spark += blocks[index]

        return spark

    # ---------------- Aggregation ----------------
    def get_commodity_data(self):
        current = self.fetch_current_prices()

        gold_history = self.calculate_pct_change(
            self.fetch_history("XAU")
        )

        silver_history = self.calculate_pct_change(
            self.fetch_history("XAG")
        )

        current["gold"]["history"] = gold_history
        current["silver"]["history"] = silver_history
        current["gold"]["change"] = gold_history[-1]["change"]
        current["silver"]["change"] = silver_history[-1]["change"]

        return current

    # ---------------- Badge ----------------
    def badge(self, change):
        color = "#16a34a" if change >= 0 else "#dc2626"
        bg = "#dcfce7" if change >= 0 else "#fee2e2"
        sign = "+" if change > 0 else ""

        return f"""
        <span style="
            background:{bg};
            color:{color};
            padding:6px 12px;
            border-radius:999px;
            font-weight:700;
            font-size:13px;">
            {sign}{change}%
        </span>
        """

    def buy_signal(self, history, label):
        if len(history) < 3:
            return ""

        recent_changes = [row["change"] for row in history[-3:]]
        latest_change = recent_changes[-1]
        prev_change = recent_changes[-2]
        older_change = recent_changes[-3]

        score = 0
        if latest_change <= -1.5:
            score += 4
        if latest_change <= -2.5:
            score += 4
        if prev_change <= -1.0:
            score += 2
        if older_change <= -1.0:
            score += 1
        if latest_change < prev_change:
            score += 2
        if latest_change < 0:
            score += 1

        if score >= 8:
            return f"""
            <div style="margin-top:10px;display:inline-block;padding:6px 10px;border-radius:999px;background:#dcfce7;color:#166534;font-weight:700;font-size:12px;">
                Buy {label}
            </div>
            """
        return ""

    def derive_buy_levels(self, current_price, history):
        return derive_commodity_buy_levels(current_price, history)

    # ---------------- HTML ----------------
    def generate_html(self):
        data = self.get_commodity_data()
        gold = data["gold"]
        silver = data["silver"]

        # Build separate history rows for Gold and Silver for clearer, split sections
        gold_rows = ""
        for g in reversed(gold["history"]):
            gold_color = "#16a34a" if g["change"] >= 0 else "#dc2626"
            gold_sign = "+" if g["change"] > 0 else ""
            gold_rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:12px;">{g['date']}</td>
                <td>₹{g['price']}</td>
                <td style="color:{gold_color};font-weight:700;">{gold_sign}{g['change']}%</td>
            </tr>
            """

        silver_rows = ""
        for s in reversed(silver["history"]):
            silver_color = "#16a34a" if s["change"] >= 0 else "#dc2626"
            silver_sign = "+" if s["change"] > 0 else ""
            silver_rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:12px;">{s['date']}</td>
                <td>₹{s['price']}</td>
                <td style="color:{silver_color};font-weight:700;">{silver_sign}{s['change']}%</td>
            </tr>
            """

        gold_prices = [x["price"] for x in gold["history"]]
        silver_prices = [x["price"] for x in silver["history"]]

        gold_levels = self.derive_buy_levels(gold["current"], gold["history"])
        silver_levels = self.derive_buy_levels(silver["current"], silver["history"])

        html = f"""
        <div style="
            margin-top:24px;">
            <div style="
                padding:18px 20px 12px;
                border-bottom:1px solid #e5e7eb;
                background:linear-gradient(135deg,#fff7ed,#fffbeb);">
                <div style="font-size:20px;font-weight:800;color:#111827;">Commodities</div>
                <div style="font-size:13px;color:#64748b;margin-top:6px;line-height:1.5;">Separate Gold and Silver sections for a cleaner, professional layout.</div>
            </div>

            <div style="padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                <!-- Gold Section -->
                <div style="background:#ffffff;border-radius:14px;border:1px solid #e5e7eb;padding:16px;">
                    <div style="font-size:13px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:0.04em;">Gold (22K)</div>
                    <div style="font-size:28px;font-weight:800;color:#111827;margin-top:8px;">₹{gold['current']}</div>
                    <div style="margin-top:10px;">{self.badge(gold['change'])} {self.buy_signal(gold['history'], 'Gold')}</div>
                    <div style="margin-top:12px;padding:10px 12px;border-radius:12px;background:#fffdf7;border:1px solid #f3e8b2;font-size:12px;color:#334155;line-height:1.5;">
                        <div style="font-weight:700;color:#047857;">Recommended {gold_levels['recommended_entry_label']}: ₹{gold_levels['recommended_buy_level']}</div>
                        <div style="margin-top:4px;">Patient: ₹{gold_levels['patient_entry']} • Optimal: ₹{gold_levels['optimal_entry']} • Aggressive: ₹{gold_levels['aggressive_entry']}</div>
                    </div>

                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:12px;border-collapse:collapse;font-size:13px;color:#334155;">
                        <tr style="background:#f8fafc;">
                            <th align="left" style="padding:10px;border-bottom:1px solid #e5e7eb;">Date</th>
                            <th align="left" style="padding:10px;border-bottom:1px solid #e5e7eb;">Price</th>
                            <th align="left" style="padding:10px;border-bottom:1px solid #e5e7eb;">Change</th>
                        </tr>
                        {gold_rows}
                    </table>
                </div>

                <!-- Silver Section -->
                <div style="background:#ffffff;border-radius:14px;border:1px solid #e5e7eb;padding:16px;">
                    <div style="font-size:13px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.04em;">Silver</div>
                    <div style="font-size:28px;font-weight:800;color:#111827;margin-top:8px;">₹{silver['current']}</div>
                    <div style="margin-top:10px;">{self.badge(silver['change'])} {self.buy_signal(silver['history'], 'Silver')}</div>
                    <div style="margin-top:12px;padding:10px 12px;border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0;font-size:12px;color:#334155;line-height:1.5;">
                        <div style="font-weight:700;color:#047857;">Recommended {silver_levels['recommended_entry_label']}: ₹{silver_levels['recommended_buy_level']}</div>
                        <div style="margin-top:4px;">Patient: ₹{silver_levels['patient_entry']} • Optimal: ₹{silver_levels['optimal_entry']} • Aggressive: ₹{silver_levels['aggressive_entry']}</div>
                    </div>

                    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="margin-top:12px;border-collapse:collapse;font-size:13px;color:#334155;">
                        <tr style="background:#f8fafc;">
                            <th align="left" style="padding:10px;border-bottom:1px solid #e5e7eb;">Date</th>
                            <th align="left" style="padding:10px;border-bottom:1px solid #e5e7eb;">Price</th>
                            <th align="left" style="padding:10px;border-bottom:1px solid #e5e7eb;">Change</th>
                        </tr>
                        {silver_rows}
                    </table>
                </div>
            </div>

            <div style="padding:16px 20px 20px;border-top:1px solid #e5e7eb;background:transparent;border-radius:0 0 12px 12px;margin-top:12px;">
                <div style="font-size:13px;color:#64748b;">Summary: Latest live prices and recommended buy levels for Gold and Silver are shown above.</div>
            </div>
        </div>
        """

        return html


if __name__ == "__main__":
    tracker = CommodityTracker()
    print(tracker.generate_html())

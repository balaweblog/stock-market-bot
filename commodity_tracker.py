import requests
from datetime import datetime, timedelta
import yfinance as yf

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

    # ---------------- HTML ----------------
    def generate_html(self):
        data = self.get_commodity_data()
        gold = data["gold"]
        silver = data["silver"]

        rows = ""

        for g, s in zip(reversed(gold["history"]), reversed(silver["history"])):
            

            gold_color = "#16a34a" if g["change"] >= 0 else "#dc2626"
            silver_color = "#16a34a" if s["change"] >= 0 else "#dc2626"

            gold_sign = "+" if g["change"] > 0 else ""
            silver_sign = "+" if s["change"] > 0 else ""

            rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:12px;">{g['date']}</td>
                <td>₹{g['price']}</td>
                <td style="color:{gold_color};font-weight:700;">
                    {gold_sign}{g['change']}%
                </td>
                <td>₹{s['price']}</td>
                <td style="color:{silver_color};font-weight:700;">
                    {silver_sign}{s['change']}%
                </td>
            </tr>
            """

        gold_prices = [x["price"] for x in gold["history"]]
        silver_prices = [x["price"] for x in silver["history"]]

        gold_spark = self.generate_sparkline(gold_prices)
        silver_spark = self.generate_sparkline(silver_prices)

        gold_spark_color = "#16a34a" if gold["change"] >= 0 else "#dc2626"
        silver_spark_color = "#16a34a" if silver["change"] >= 0 else "#dc2626"

        gold_trend = (
            '<span style="color:#16a34a;font-weight:700;">Bullish ↗</span>'
            if gold["change"] >= 0
            else '<span style="color:#dc2626;font-weight:700;">Bearish ↘</span>'
        )

        silver_trend = (
            '<span style="color:#16a34a;font-weight:700;">Bullish ↗</span>'
            if silver["change"] >= 0
            else '<span style="color:#dc2626;font-weight:700;">Bearish ↘</span>'
        )

        html = f"""
        <div style="
            margin-top:24px;
            background:white;
            border-radius:20px;
            border:1px solid #e2e8f0;
            box-shadow:0 8px 24px rgba(0,0,0,0.05);
            overflow:hidden;">

            <div style="
                background:linear-gradient(135deg,#fff7ed,#fffbeb);
                padding:20px;
                border-bottom:1px solid #e2e8f0;">
                <div style="font-size:22px;font-weight:800;color:#0f172a;">
                    Gold & Silver Market
                </div>
                <div style="font-size:13px;color:#64748b;margin-top:6px;">
                    Indian Spot Commodity Dashboard
                </div>
            </div>

            <div style="padding:24px;">
                <table width="100%">
                    <tr>
                        <td width="50%">
                            <div style="
                                background:linear-gradient(135deg,#fff8dc,#fff3b0);
                                padding:20px;
                                border-radius:16px;">
                                <div style="font-size:15px;font-weight:700;">Gold (22K)</div>
                                <div style="font-size:34px;font-weight:800;margin-top:8px;">
                                    ₹{gold['current']}
                                </div>
                                <div style="margin-top:10px;">
                                    {self.badge(gold['change'])}
                                </div>
                            </div>
                        </td>

                        <td width="50%">
                            <div style="
                                background:linear-gradient(135deg,#f8fafc,#e2e8f0);
                                padding:20px;
                                border-radius:16px;">
                                <div style="font-size:15px;font-weight:700;">Silver</div>
                                <div style="font-size:34px;font-weight:800;margin-top:8px;">
                                    ₹{silver['current']}
                                </div>
                                <div style="margin-top:10px;">
                                    {self.badge(silver['change'])}
                                </div>
                            </div>
                        </td>
                    </tr>
                </table>

                <table width="100%" style="
                    margin-top:24px;
                    border-collapse:collapse;
                    font-size:14px;">
                    <tr style="background:#f8fafc;">
                        <th align="left" style="padding:12px;">Date</th>
                        <th align="left">Gold</th>
                        <th align="left">Gold %</th>
                        <th align="left">Silver</th>
                        <th align="left">Silver %</th>
                    </tr>
                    {rows}
                </table>

                <div style="
                    margin-top:20px;
                    padding:20px;
                    background:#f8fafc;
                    border-radius:14px;
                    border:1px solid #e2e8f0;">

                    <div style="font-weight:700;">7-Day Trend</div>

                    <div style="font-size:24px;margin-top:10px;">
                        Gold:
                        <span style="color:{gold_spark_color};font-family:monospace;">
                            {gold_spark}
                        </span>
                    </div>

                    <div style="font-size:24px;margin-top:8px;">
                        Silver:
                        <span style="color:{silver_spark_color};font-family:monospace;">
                            {silver_spark}
                        </span>
                    </div>

                    <div style="margin-top:18px;">
                        <b>AI Insights</b>
                        <ul>
                            <li>Gold trend: {gold_trend}</li>
                            <li>Silver trend: {silver_trend}</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """

        return html


if __name__ == "__main__":
    tracker = CommodityTracker()
    print(tracker.generate_html())
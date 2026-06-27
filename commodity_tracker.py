import requests
from datetime import datetime, timedelta


class CommodityTracker:
    BASE_URL = "https://www.goldapi.io/api"

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "x-access-token": api_key,
            "Content-Type": "application/json"
        }

    def call_api(self, endpoint):
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} {response.text}")

        return response.json()

    def oz_to_gram(self, price):
        return round(price / 31.1035, 2)

    def fetch_current_prices(self):
        gold = self.call_api("XAU/INR")
        silver = self.call_api("XAG/INR")

        gold_price = self.oz_to_gram(gold["price"])
        silver_price = self.oz_to_gram(silver["price"])

        gold_change = round(gold.get("ch", 0), 2)
        silver_change = round(silver.get("ch", 0), 2)

        return {
            "gold": {
                "current": gold_price,
                "change": gold_change
            },
            "silver": {
                "current": silver_price,
                "change": silver_change
            }
        }
    def get_day_suffix(self,day):
        if 11 <= day <= 13:
            return "th"
        return {
            1: "st",
            2: "nd",
            3: "rd"
        }.get(day % 10, "th")
    def format_date_short(self,date_str):
        dt = datetime.strptime(date_str, "%Y%m%d")
        day = dt.day
        suffix = self.get_day_suffix(day)
        return dt.strftime(f"{day}{suffix} %b (%a)")

    def fetch_history(self, symbol, days=7):
        history = []

        for i in range(days, 0, -1):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")

            data = self.call_api(f"{symbol}/INR/{date}")

            price = self.oz_to_gram(data["price"])

            history.append({
                "date": self.format_date_short(date),
                "price": price
            })

        return history

    def calculate_pct_change(self, history):
        previous = None

        for row in history:
            if previous is None:
                row["change"] = 0
            else:
                row["change"] = round(
                    ((row["price"] - previous) / previous) * 100,
                    2
                )
            previous = row["price"]

        return history

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

        return current

    def build_history_rows(self, history):
        rows = ""

        for row in history:
            color = "#047857" if row["change"] >= 0 else "#dc2626"
            sign = "+" if row["change"] >= 0 else ""

            rows += f"""
            <tr>
                <td style="padding:4px 0;">{row['date']}</td>
                <td style="padding:4px 0;">₹{row['price']}</td>
                <td style="padding:4px 0;color:{color};font-weight:600;">
                    {sign}{row['change']}%
                </td>
            </tr>
            """

        return rows

    def generate_html(self):
        data = self.get_commodity_data()

        gold = data["gold"]
        silver = data["silver"]

        gold_color = "#047857" if gold["change"] >= 0 else "#dc2626"
        silver_color = "#047857" if silver["change"] >= 0 else "#dc2626"

        gold_sign = "+" if gold["change"] >= 0 else ""
        silver_sign = "+" if silver["change"] >= 0 else ""

        html = f"""
        <table width="100%" cellpadding="0" cellspacing="0"
               role="presentation"
               style="margin:12px 0;border-radius:12px;
               background:#ffffff;border:1px solid #e5e7eb;">
            <tr>
                <td style="padding:14px;">

                    <table width="100%">
                        <tr>
                            <td>
                                <h3 style="margin:0;font-size:16px;color:#0f172a;">
                                    Gold & Silver Market
                                </h3>
                                <p style="margin:6px 0 0;font-size:13px;color:#334155;">
                                    Indian spot commodity report
                                </p>
                            </td>
                        </tr>
                    </table>

                    <table width="100%" style="margin-top:14px;">
                        <tr>
                            <td width="50%">
                                <div style="background:#f8fafc;
                                    padding:12px;
                                    border-radius:10px;
                                    border:1px solid #e2e8f0;">
                                    <div style="font-size:16px;font-weight:700;">
                                        Gold (24K)
                                    </div>
                                    <div style="font-size:22px;
                                        font-weight:700;margin-top:8px;">
                                        ₹{gold['current']}/g
                                    </div>
                                    <div style="color:{gold_color};
                                        font-weight:700;">
                                        {gold_sign}{gold['change']}%
                                    </div>
                                </div>
                            </td>

                            <td width="50%">
                                <div style="background:#f8fafc;
                                    padding:12px;
                                    border-radius:10px;
                                    border:1px solid #e2e8f0;">
                                    <div style="font-size:16px;font-weight:700;">
                                        Silver
                                    </div>
                                    <div style="font-size:22px;
                                        font-weight:700;margin-top:8px;">
                                        ₹{silver['current']}/g
                                    </div>
                                    <div style="color:{silver_color};
                                        font-weight:700;">
                                        {silver_sign}{silver['change']}%
                                    </div>
                                </div>
                            </td>
                        </tr>
                    </table>

                    <table width="100%" style="margin-top:16px;">
                        <tr>
                            <td width="50%" style="vertical-align:top;padding-right:8px;">
                                <div style="font-weight:700;margin-bottom:6px;">
                                    Gold 7-Day History
                                </div>
                                <table width="100%" style="font-size:12px;">
                                    <tr>
                                        <th align="left">Date</th>
                                        <th align="left">Price</th>
                                        <th align="left">% Change</th>
                                    </tr>
                                    {self.build_history_rows(gold["history"])}
                                </table>
                            </td>

                            <td width="50%" style="vertical-align:top;padding-left:8px;">
                                <div style="font-weight:700;margin-bottom:6px;">
                                    Silver 7-Day History
                                </div>
                                <table width="100%" style="font-size:12px;">
                                    <tr>
                                        <th align="left">Date</th>
                                        <th align="left">Price</th>
                                        <th align="left">% Change</th>
                                    </tr>
                                    {self.build_history_rows(silver["history"])}
                                </table>
                            </td>
                        </tr>
                    </table>

                </td>
            </tr>
        </table>
        """

        return html


if __name__ == "__main__":
    GOLD_API_KEY = "YOUR_GOLD_API_KEY"

    tracker = CommodityTracker(GOLD_API_KEY)

    html = tracker.generate_html()

    print(html)
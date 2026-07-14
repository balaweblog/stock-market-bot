"""
Simulate exactly what main.py does for the commodity section and write to HTML.
This lets us see whether the cards render correctly in a browser.
"""
import traceback
from commodity_tracker import CommodityTracker

report_html = """
<html>
  <body style="margin:0;padding:0;background:#f4f6f8;font-family:Arial,sans-serif;color:#111827;">
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="background:#f4f6f8;width:100%;min-width:100%;">
      <tr>
        <td align="center" style="padding:16px;">
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation" style="max-width:680px;min-width:320px;background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;">
            <tr>
              <td style="padding:18px 20px 12px;">
                <h1 style="margin:0;font-size:22px;">Commodity Test</h1>
              </td>
            </tr>
"""

commodity_data = None
tracker = None
try:
    tracker = CommodityTracker()
    commodity_data = tracker.get_commodity_data()
    print(f"Fetched OK: gold={commodity_data['gold']['current']}, silver={commodity_data['silver']['current']}")
except Exception as e:
    print(f"Fetch failed: {e}")
    traceback.print_exc()

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

        report_html += f"""
            <tr>
              <td style="padding:0 20px 20px;">
                {commodity_section_html}
              </td>
            </tr>
        """
        print("Commodity section appended to report_html OK")
    except Exception as e:
        print(f"Render failed: {e}")
        traceback.print_exc()
else:
    report_html += """
        <tr>
          <td style="padding:14px;color:#721c24;">Commodities unavailable.</td>
        </tr>
    """

report_html += """
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

with open("_commodity_test_output.html", "w") as f:
    f.write(report_html)

print("Written to _commodity_test_output.html")
print(f"Total HTML length: {len(report_html)}")
print(f"'Commodities' appears in HTML: {'Commodities' in report_html}")
print(f"'Gold' appears in HTML: {'Gold' in report_html}")
print(f"'Silver' appears in HTML: {'Silver' in report_html}")

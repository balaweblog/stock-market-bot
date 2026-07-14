import traceback
from commodity_tracker import CommodityTracker

try:
    tracker = CommodityTracker()
    print("Fetching commodity data...")
    data = tracker.get_commodity_data()
    print("Gold current:", data["gold"]["current"])
    print("Silver current:", data["silver"]["current"])
    print("Gold history length:", len(data["gold"]["history"]))
    print("Silver history length:", len(data["silver"]["history"]))

    print("\nDeriving buy levels...")
    gold_levels = tracker.derive_buy_levels(data["gold"]["current"], data["gold"]["history"])
    silver_levels = tracker.derive_buy_levels(data["silver"]["current"], data["silver"]["history"])
    print("Gold levels:", gold_levels)
    print("Silver levels:", silver_levels)

    print("\nBuilding trade plans...")
    gold_plan = tracker.build_trade_plan(data["gold"]["current"], data["gold"]["history"], gold_levels)
    silver_plan = tracker.build_trade_plan(data["silver"]["current"], data["silver"]["history"], silver_levels)
    print("Gold plan:", gold_plan)
    print("Silver plan:", silver_plan)

    print("\nRendering gold card HTML...")
    gold_card = tracker._commodity_card_html(
        name="Gold (22K)", ticker_label="XAU/INR",
        current_price=data["gold"]["current"],
        change=data["gold"]["change"],
        history=data["gold"]["history"],
        levels=gold_levels, plan=gold_plan,
    )
    print("Gold card rendered OK, length:", len(gold_card))

    print("\nRendering silver card HTML...")
    silver_card = tracker._commodity_card_html(
        name="Silver", ticker_label="XAG/INR",
        current_price=data["silver"]["current"],
        change=data["silver"]["change"],
        history=data["silver"]["history"],
        levels=silver_levels, plan=silver_plan,
    )
    print("Silver card rendered OK, length:", len(silver_card))

    print("\nAll steps completed successfully.")

except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()

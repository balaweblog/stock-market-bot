"""
swing_trade_universe.py

Static ticker seed list for swing_trade_advisor.py's deterministic Stage-1
fundamentals screen (USE_DETERMINISTIC_SCREEN=true). Every ticker listed
here gets checked against REAL data via _verify_fundamentals /
_fetch_fundamentals -- this file only decides which names are even looked
at; it never asserts that a name currently passes the growth/debt/ROE
bar (the deterministic screen still does that independently, every run).

WHY THIS FILE WAS THIN BEFORE / WHY IT'S BIGGER NOW:
The whole point of USE_DETERMINISTIC_SCREEN is "check the WHOLE seed list,
not an LLM's sample-of-8-12" (see swing_trade_advisor.py's own docstring
for that flag) -- but that guarantee is only as good as this list's
coverage. A short list quietly defeats the point: it can only ever
recommend from however many names happen to be here, no matter how
exhaustive the verification logic is. This version widens coverage across
large/mid/small-cap within each of the 14 SECTORS already defined in
swing_trade_advisor.py, still India-only (NSE), per your call to expand
depth here rather than breadth into other countries' markets.

MARKET-CAP BUCKETS are informational labels only ("Large-cap" / "Mid-cap"
/ "Small-cap") -- a best-effort classification at the time this list was
written, shown in the report and used only for display/steering context,
never as a filter. Cap boundaries drift over time (a stock can migrate
buckets after a rally or a correction), so treat the label as approximate.

IMPORTANT -- AUDIT BEFORE RELYING ON THIS IN PRODUCTION:
Every ticker below is a real, currently-listed NSE company to the best of
available knowledge, but this list was compiled by hand, not pulled live
from an index-constituent feed, so:
  1. Tickers can go stale (delisting, merger, symbol change, demerger
     spinning off a new listing under a new symbol).
  2. A market-cap label can drift out of date.
  3. This is not exhaustive of the NSE -- it's a working seed list,
     weighted toward liquid, analyst-covered names so fundamentals data
     is actually fetchable via yfinance.
None of this breaks the pipeline silently: any ticker that's delisted, or
whose yfinance data can't be fetched, is caught by _verify_fundamentals's
existing "nodata" handling in swing_trade_advisor.py and rejected the same
way a wrong/hallucinated ticker already was -- it just won't produce a
false positive. But a stale ticker also means a real, currently-qualifying
company sits unseen. Re-validate/refresh this list periodically (e.g.
against BSE SmallCap 250 / BSE MidCap 150 / Nifty 100 constituent lists)
rather than treating it as permanent.

Interface expected by swing_trade_advisor.py:
    tickers_for_sectors(sectors) -> iterable of (name, ticker, sector, bucket)
"""

# sector -> [(name, ticker, market_cap_bucket), ...]
# Bucket is one of: "Large-cap", "Mid-cap", "Small-cap".
UNIVERSE = {

    "IT & Technology": [
        ("Tata Consultancy Services", "TCS.NS", "Large-cap"),
        ("Infosys", "INFY.NS", "Large-cap"),
        ("Wipro", "WIPRO.NS", "Large-cap"),
        ("HCL Technologies", "HCLTECH.NS", "Large-cap"),
        ("Tech Mahindra", "TECHM.NS", "Large-cap"),
        ("LTIMindtree", "LTIM.NS", "Mid-cap"),
        ("Mphasis", "MPHASIS.NS", "Mid-cap"),
        ("Coforge", "COFORGE.NS", "Mid-cap"),
        ("Persistent Systems", "PERSISTENT.NS", "Mid-cap"),
        ("L&T Technology Services", "LTTS.NS", "Mid-cap"),
        ("Zensar Technologies", "ZENSARTECH.NS", "Small-cap"),
        ("Newgen Software", "NEWGEN.NS", "Small-cap"),
        ("Intellect Design Arena", "INTELLECT.NS", "Small-cap"),
        ("Cyient", "CYIENT.NS", "Small-cap"),
        ("Happiest Minds Technologies", "HAPPSTMNDS.NS", "Small-cap"),
        ("Sonata Software", "SONATSOFTW.NS", "Small-cap"),
        ("Mastek", "MASTEK.NS", "Small-cap"),
        ("KPIT Technologies", "KPITTECH.NS", "Small-cap"),
        ("Route Mobile", "ROUTE.NS", "Small-cap"),
        ("Birlasoft", "BSOFT.NS", "Small-cap"),
    ],

    "Pharma & Healthcare": [
        ("Sun Pharmaceutical Industries", "SUNPHARMA.NS", "Large-cap"),
        ("Dr. Reddy's Laboratories", "DRREDDY.NS", "Large-cap"),
        ("Cipla", "CIPLA.NS", "Large-cap"),
        ("Divi's Laboratories", "DIVISLAB.NS", "Large-cap"),
        ("Lupin", "LUPIN.NS", "Mid-cap"),
        ("Torrent Pharmaceuticals", "TORNTPHARM.NS", "Mid-cap"),
        ("Alkem Laboratories", "ALKEM.NS", "Mid-cap"),
        ("Aurobindo Pharma", "AUROPHARMA.NS", "Mid-cap"),
        ("Abbott India", "ABBOTINDIA.NS", "Mid-cap"),
        ("Gland Pharma", "GLAND.NS", "Mid-cap"),
        ("Ipca Laboratories", "IPCALAB.NS", "Mid-cap"),
        ("Laurus Labs", "LAURUSLABS.NS", "Small-cap"),
        ("Granules India", "GRANULES.NS", "Small-cap"),
        ("Natco Pharma", "NATCOPHARM.NS", "Small-cap"),
        ("Ajanta Pharma", "AJANTPHARM.NS", "Small-cap"),
        ("J.B. Chemicals & Pharmaceuticals", "JBCHEPHARM.NS", "Small-cap"),
        ("Suven Pharmaceuticals", "SUVENPHAR.NS", "Small-cap"),
        ("Caplin Point Laboratories", "CAPLIPOINT.NS", "Small-cap"),
        ("Strides Pharma Science", "STAR.NS", "Small-cap"),
        ("Windlas Biotech", "WINDLAS.NS", "Small-cap"),
    ],

    "Banking & NBFC": [
        ("HDFC Bank", "HDFCBANK.NS", "Large-cap"),
        ("ICICI Bank", "ICICIBANK.NS", "Large-cap"),
        ("Kotak Mahindra Bank", "KOTAKBANK.NS", "Large-cap"),
        ("Axis Bank", "AXISBANK.NS", "Large-cap"),
        ("State Bank of India", "SBIN.NS", "Large-cap"),
        ("Bajaj Finance", "BAJFINANCE.NS", "Large-cap"),
        ("Federal Bank", "FEDERALBNK.NS", "Mid-cap"),
        ("IDFC First Bank", "IDFCFIRSTB.NS", "Mid-cap"),
        ("Bank of India", "BANKINDIA.NS", "Mid-cap"),
        ("AU Small Finance Bank", "AUBANK.NS", "Mid-cap"),
        ("Cholamandalam Investment & Finance", "CHOLAFIN.NS", "Mid-cap"),
        ("Sundaram Finance", "SUNDARMFIN.NS", "Mid-cap"),
        ("City Union Bank", "CUB.NS", "Small-cap"),
        ("DCB Bank", "DCBBANK.NS", "Small-cap"),
        ("Karur Vysya Bank", "KARURVYSYA.NS", "Small-cap"),
        ("CreditAccess Grameen", "CREDITACC.NS", "Small-cap"),
        ("Ujjivan Small Finance Bank", "UJJIVANSFB.NS", "Small-cap"),
        ("Equitas Small Finance Bank", "EQUITASBNK.NS", "Small-cap"),
        ("Spandana Sphoorty Financial", "SPANDANA.NS", "Small-cap"),
        ("Poonawalla Fincorp", "POONAWALLA.NS", "Small-cap"),
    ],

    "Capital Goods & Infrastructure": [
        ("Larsen & Toubro", "LT.NS", "Large-cap"),
        ("Siemens", "SIEMENS.NS", "Large-cap"),
        ("ABB India", "ABB.NS", "Large-cap"),
        ("Cummins India", "CUMMINSIND.NS", "Mid-cap"),
        ("Thermax", "THERMAX.NS", "Mid-cap"),
        ("BHEL", "BHEL.NS", "Mid-cap"),
        ("KEC International", "KEC.NS", "Mid-cap"),
        ("Kalpataru Projects International", "KALPATPOWR.NS", "Mid-cap"),
        ("APL Apollo Tubes", "APLAPOLLO.NS", "Mid-cap"),
        ("Triveni Turbine", "TRITURBINE.NS", "Small-cap"),
        ("Elgi Equipments", "ELGIEQUIP.NS", "Small-cap"),
        ("Grindwell Norton", "GRINDWELL.NS", "Small-cap"),
        ("Tube Investments of India", "TIINDIA.NS", "Small-cap"),
        ("H.G. Infra Engineering", "HGINFRA.NS", "Small-cap"),
        ("KEI Industries", "KEI.NS", "Small-cap"),
        ("Kirloskar Oil Engines", "KIRLOSENG.NS", "Small-cap"),
        ("Praj Industries", "PRAJIND.NS", "Small-cap"),
        ("Techno Electric & Engineering", "TECHNOE.NS", "Small-cap"),
    ],

    "Auto & Auto Ancillaries": [
        ("Maruti Suzuki India", "MARUTI.NS", "Large-cap"),
        ("Mahindra & Mahindra", "M&M.NS", "Large-cap"),
        ("Tata Motors", "TATAMOTORS.NS", "Large-cap"),
        ("Bajaj Auto", "BAJAJ-AUTO.NS", "Large-cap"),
        ("Eicher Motors", "EICHERMOT.NS", "Large-cap"),
        ("Samvardhana Motherson International", "MOTHERSON.NS", "Mid-cap"),
        ("Bosch", "BOSCHLTD.NS", "Mid-cap"),
        ("TVS Motor Company", "TVSMOTOR.NS", "Mid-cap"),
        ("Ashok Leyland", "ASHOKLEY.NS", "Mid-cap"),
        ("Balkrishna Industries", "BALKRISIND.NS", "Mid-cap"),
        ("Bharat Forge", "BHARATFORG.NS", "Mid-cap"),
        ("Sandhar Technologies", "SANDHAR.NS", "Small-cap"),
        ("Endurance Technologies", "ENDURANCE.NS", "Small-cap"),
        ("Subros", "SUBROS.NS", "Small-cap"),
        ("Suprajit Engineering", "SUPRAJIT.NS", "Small-cap"),
        ("Fiem Industries", "FIEMIND.NS", "Small-cap"),
        ("Gabriel India", "GABRIEL.NS", "Small-cap"),
        ("UNO Minda", "UNOMINDA.NS", "Small-cap"),
        ("Rico Auto Industries", "RICOAUTO.NS", "Small-cap"),
    ],

    "Chemicals & Fertilizers": [
        ("UPL", "UPL.NS", "Large-cap"),
        ("Pidilite Industries", "PIDILITIND.NS", "Large-cap"),
        ("SRF", "SRF.NS", "Mid-cap"),
        ("Deepak Nitrite", "DEEPAKNTR.NS", "Mid-cap"),
        ("Aarti Industries", "AARTIIND.NS", "Mid-cap"),
        ("Coromandel International", "COROMANDEL.NS", "Mid-cap"),
        ("Gujarat Narmada Valley Fertilizers", "GNFC.NS", "Mid-cap"),
        ("Navin Fluorine International", "NAVINFLUOR.NS", "Small-cap"),
        ("Fine Organic Industries", "FINEORG.NS", "Small-cap"),
        ("Vinati Organics", "VINATIORGA.NS", "Small-cap"),
        ("Chemplast Sanmar", "CHEMPLASTS.NS", "Small-cap"),
        ("Tata Chemicals", "TATACHEM.NS", "Small-cap"),
        ("Galaxy Surfactants", "GALAXYSURF.NS", "Small-cap"),
        ("Clean Science and Technology", "CLEAN.NS", "Small-cap"),
        ("Rossari Biotech", "ROSSARI.NS", "Small-cap"),
        ("Alkyl Amines Chemicals", "ALKYLAMINE.NS", "Small-cap"),
    ],

    "Defence": [
        ("Hindustan Aeronautics", "HAL.NS", "Large-cap"),
        ("Bharat Electronics", "BEL.NS", "Large-cap"),
        ("BEML", "BEML.NS", "Mid-cap"),
        ("Mazagon Dock Shipbuilders", "MAZDOCK.NS", "Mid-cap"),
        ("Cochin Shipyard", "COCHINSHIP.NS", "Mid-cap"),
        ("Bharat Dynamics", "BDL.NS", "Mid-cap"),
        ("MTAR Technologies", "MTARTECH.NS", "Small-cap"),
        ("Data Patterns (India)", "DATAPATTNS.NS", "Small-cap"),
        ("Paras Defence and Space Technologies", "PARAS.NS", "Small-cap"),
        ("Astra Microwave Products", "ASTRAMICRO.NS", "Small-cap"),
        ("Garden Reach Shipbuilders & Engineers", "GRSE.NS", "Small-cap"),
        ("Zen Technologies", "ZENTEC.NS", "Small-cap"),
    ],

    "Consumer & FMCG": [
        ("Hindustan Unilever", "HINDUNILVR.NS", "Large-cap"),
        ("ITC", "ITC.NS", "Large-cap"),
        ("Nestle India", "NESTLEIND.NS", "Large-cap"),
        ("Britannia Industries", "BRITANNIA.NS", "Large-cap"),
        ("Varun Beverages", "VBL.NS", "Large-cap"),
        ("Dabur India", "DABUR.NS", "Mid-cap"),
        ("Marico", "MARICO.NS", "Mid-cap"),
        ("Godrej Consumer Products", "GODREJCP.NS", "Mid-cap"),
        ("Colgate-Palmolive (India)", "COLPAL.NS", "Mid-cap"),
        ("Tata Consumer Products", "TATACONSUM.NS", "Mid-cap"),
        ("Emami", "EMAMILTD.NS", "Small-cap"),
        ("Jyothy Labs", "JYOTHYLAB.NS", "Small-cap"),
        ("Bajaj Consumer Care", "BAJAJCON.NS", "Small-cap"),
        ("Radico Khaitan", "RADICO.NS", "Small-cap"),
        ("CCL Products (India)", "CCL.NS", "Small-cap"),
        ("DFM Foods", "DFMFOODS.NS", "Small-cap"),
    ],

    "Metals & Mining": [
        ("Tata Steel", "TATASTEEL.NS", "Large-cap"),
        ("JSW Steel", "JSWSTEEL.NS", "Large-cap"),
        ("Hindalco Industries", "HINDALCO.NS", "Large-cap"),
        ("Vedanta", "VEDL.NS", "Large-cap"),
        ("Hindustan Zinc", "HINDZINC.NS", "Large-cap"),
        ("Jindal Steel & Power", "JINDALSTEL.NS", "Mid-cap"),
        ("NMDC", "NMDC.NS", "Mid-cap"),
        ("National Aluminium Company", "NATIONALUM.NS", "Mid-cap"),
        ("Steel Authority of India", "SAIL.NS", "Mid-cap"),
        ("Ratnamani Metals & Tubes", "RATNAMANI.NS", "Small-cap"),
        ("Welspun Corp", "WELCORP.NS", "Small-cap"),
        ("Jindal Stainless", "JSL.NS", "Small-cap"),
        ("Hindustan Copper", "HINDCOPPER.NS", "Small-cap"),
        ("MOIL", "MOIL.NS", "Small-cap"),
    ],

    "Realty & Construction": [
        ("DLF", "DLF.NS", "Large-cap"),
        ("Godrej Properties", "GODREJPROP.NS", "Large-cap"),
        ("Oberoi Realty", "OBEROIRLTY.NS", "Mid-cap"),
        ("Phoenix Mills", "PHOENIXLTD.NS", "Mid-cap"),
        ("Prestige Estates Projects", "PRESTIGE.NS", "Mid-cap"),
        ("Brigade Enterprises", "BRIGADE.NS", "Mid-cap"),
        ("Sobha", "SOBHA.NS", "Small-cap"),
        ("Mahindra Lifespace Developers", "MAHLIFE.NS", "Small-cap"),
        ("Sunteck Realty", "SUNTECK.NS", "Small-cap"),
        ("Kolte-Patil Developers", "KOLTEPATIL.NS", "Small-cap"),
        ("Puravankara", "PURVA.NS", "Small-cap"),
    ],

    "Energy & Power": [
        ("NTPC", "NTPC.NS", "Large-cap"),
        ("Power Grid Corporation of India", "POWERGRID.NS", "Large-cap"),
        ("Oil and Natural Gas Corporation", "ONGC.NS", "Large-cap"),
        ("Coal India", "COALINDIA.NS", "Large-cap"),
        ("Adani Green Energy", "ADANIGREEN.NS", "Large-cap"),
        ("Tata Power Company", "TATAPOWER.NS", "Mid-cap"),
        ("Torrent Power", "TORNTPOWER.NS", "Mid-cap"),
        ("JSW Energy", "JSWENERGY.NS", "Mid-cap"),
        ("NHPC", "NHPC.NS", "Mid-cap"),
        ("SJVN", "SJVN.NS", "Mid-cap"),
        ("KPI Green Energy", "KPIGREEN.NS", "Small-cap"),
        ("Websol Energy System", "WEBSOL.NS", "Small-cap"),
        ("Gujarat Industries Power Company", "GIPCL.NS", "Small-cap"),
    ],

    "Textiles & Apparel": [
        ("Page Industries", "PAGEIND.NS", "Mid-cap"),
        ("Trident", "TRIDENT.NS", "Mid-cap"),
        ("Raymond", "RAYMOND.NS", "Mid-cap"),
        ("Welspun Living", "WELSPUNIND.NS", "Mid-cap"),
        ("KPR Mill", "KPRMILL.NS", "Small-cap"),
        ("Gokaldas Exports", "GOKEX.NS", "Small-cap"),
        ("Arvind", "ARVIND.NS", "Small-cap"),
        ("Rupa & Company", "RUPA.NS", "Small-cap"),
        ("Dollar Industries", "DOLLAR.NS", "Small-cap"),
        ("Vardhman Textiles", "VTL.NS", "Small-cap"),
    ],

    "Cement": [
        ("UltraTech Cement", "ULTRACEMCO.NS", "Large-cap"),
        ("Shree Cement", "SHREECEM.NS", "Large-cap"),
        ("ACC", "ACC.NS", "Mid-cap"),
        ("Ambuja Cements", "AMBUJACEM.NS", "Mid-cap"),
        ("Dalmia Bharat", "DALBHARAT.NS", "Mid-cap"),
        ("JK Cement", "JKCEMENT.NS", "Mid-cap"),
        ("The Ramco Cements", "RAMCOCEM.NS", "Small-cap"),
        ("HeidelbergCement India", "HEIDELBERG.NS", "Small-cap"),
        ("Nuvoco Vistas Corporation", "NUVOCO.NS", "Small-cap"),
        ("Star Cement", "STARCEMENT.NS", "Small-cap"),
        ("JK Lakshmi Cement", "JKLAKSHMI.NS", "Small-cap"),
    ],

    "Telecom": [
        ("Bharti Airtel", "BHARTIARTL.NS", "Large-cap"),
        ("Indus Towers", "INDUSTOWER.NS", "Mid-cap"),
        ("Vodafone Idea", "IDEA.NS", "Mid-cap"),
        ("Tata Communications", "TATACOMM.NS", "Mid-cap"),
        ("HFCL", "HFCL.NS", "Small-cap"),
        ("Sterlite Technologies", "STLTECH.NS", "Small-cap"),
        ("Railtel Corporation of India", "RAILTEL.NS", "Small-cap"),
        ("ITI Limited", "ITI.NS", "Small-cap"),
    ],
}


def tickers_for_sectors(sectors):
    """
    Yields (name, ticker, sector, bucket) for every ticker in the requested
    sectors. Unknown sector names are silently skipped (returns nothing for
    them) rather than raising, since callers pass a rotating slice of
    swing_trade_advisor.SECTORS and a typo there shouldn't crash a run --
    it should just screen zero candidates for that slice, same as a sector
    with no genuinely qualifying names this week.
    """
    for sector in sectors:
        for name, ticker, bucket in UNIVERSE.get(sector, []):
            yield name, ticker, sector, bucket


def all_tickers():
    """Flat (name, ticker, sector, bucket) list across every sector -- handy
    for one-off audits (e.g. checking every ticker still resolves via
    yfinance) without going through the sector-rotation interface."""
    out = []
    for sector, rows in UNIVERSE.items():
        for name, ticker, bucket in rows:
            out.append((name, ticker, sector, bucket))
    return out


def ticker_count_by_sector():
    """Diagnostic: how many seed tickers exist per sector, so a thin sector
    is visible rather than discovered by "why does this sector never
    produce a pick"."""
    return {sector: len(rows) for sector, rows in UNIVERSE.items()}


if __name__ == "__main__":
    counts = ticker_count_by_sector()
    total = sum(counts.values())
    print(f"Total seed tickers: {total}")
    for sector, n in sorted(counts.items()):
        print(f"  {sector}: {n}")
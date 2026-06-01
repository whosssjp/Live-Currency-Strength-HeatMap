pip install tradingview_ta pandas numpy matplotlib seaborn requests

from tradingview_ta import TA_Handler, Interval, Exchange

# Example: fetch EURUSD live data
handler = TA_Handler(
    symbol="EURUSD",
    screener="forex",        # must be "forex" for currency pairs
    exchange="FX_IDC",       # TradingView's Interdealer Composite feed
    interval=Interval.INTERVAL_1_MINUTE
)

analysis = handler.get_analysis()
print(analysis.indicators["close"])   # live close price
print(analysis.indicators["RSI"])     # RSI value
print(analysis.summary)               # BUY / SELL / NEUTRAL summary

CURRENCIES = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']

# (symbol, base, quote) — covers all cross combinations
PAIRS = [
    ("EURUSD",  "EUR", "USD"),
    ("GBPUSD",  "GBP", "USD"),
    ("USDJPY",  "USD", "JPY"),
    ("USDCHF",  "USD", "CHF"),
    ("USDCAD",  "USD", "CAD"),
    ("AUDUSD",  "AUD", "USD"),
    ("NZDUSD",  "NZD", "USD"),
    ("EURGBP",  "EUR", "GBP"),
    ("EURJPY",  "EUR", "JPY"),
    ("EURCHF",  "EUR", "CHF"),
    ("EURCAD",  "EUR", "CAD"),
    ("EURAUD",  "EUR", "AUD"),
    ("EURNZD",  "EUR", "NZD"),
    ("GBPJPY",  "GBP", "JPY"),
    ("GBPCHF",  "GBP", "CHF"),
    ("GBPCAD",  "GBP", "CAD"),
    ("GBPAUD",  "GBP", "AUD"),
    ("GBPNZD",  "GBP", "NZD"),
    ("CHFJPY",  "CHF", "JPY"),
    ("CADJPY",  "CAD", "JPY"),
    ("AUDJPY",  "AUD", "JPY"),
    ("NZDJPY",  "NZD", "JPY"),
    ("AUDCAD",  "AUD", "CAD"),
    ("AUDCHF",  "AUD", "CHF"),
    ("AUDNZD",  "AUD", "NZD"),
    ("CADCHF",  "CAD", "CHF"),
    ("NZDCAD",  "NZD", "CAD"),
    ("NZDCHF",  "NZD", "CHF"),
]

from tradingview_ta import TA_Handler, Interval

def fetch_live_rates(pairs: list) -> dict:
    """
    Fetches the latest close price for each forex pair via tradingview_ta.
    Returns: {'EURUSD': 1.08523, 'GBPUSD': 1.26548, ...}
    """
    rates = {}
    for symbol, base, quote in pairs:
        try:
            handler = TA_Handler(
                symbol=symbol,
                screener="forex",
                exchange="FX_IDC",
                interval=Interval.INTERVAL_1_MINUTE
            )
            analysis = handler.get_analysis()
            close = analysis.indicators["close"]
            rates[symbol] = {"price": close, "base": base, "quote": quote}
            print(f"  ✓ {symbol}: {close:.5f}")
        except Exception as e:
            print(f"  ✗ {symbol}: Error — {e}")
    return rates

import numpy as np

def calculate_strength(rates: dict, currencies: list) -> dict:
    """
    For each currency, averages log(price) contributions across all pairs.
    - Base currency: benefits when pair price rises  → +log(price)
    - Quote currency: hurt when pair price rises     → -log(price)
    Then normalises to [-1, +1].
    """
    scores = {c: [] for c in currencies}

    for symbol, info in rates.items():
        base  = info["base"]
        quote = info["quote"]
        log_price = np.log(info["price"])

        scores[base].append(log_price)
        scores[quote].append(-log_price)

    strength = {}
    for currency, values in scores.items():
        strength[currency] = float(np.mean(values)) if values else 0.0

    # Normalise to [-1, +1]
    max_val = max(abs(v) for v in strength.values()) or 1.0
    return {k: v / max_val for k, v in strength.items()}

import pandas as pd

def build_strength_matrix(strength: dict, currencies: list) -> pd.DataFrame:
    """
    matrix[base][quote] = strength[base] - strength[quote]
    Positive (green) = base is stronger; Negative (red) = base is weaker.
    Diagonal = 0.
    """
    matrix = pd.DataFrame(0.0, index=currencies, columns=currencies)
    for base in currencies:
        for quote in currencies:
            if base != quote:
                matrix.loc[base, quote] = round(strength[base] - strength[quote], 2)
    return matrix

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def plot_heatmap(matrix: pd.DataFrame, strength: dict, timestamp: str) -> plt.Figure:
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.4)

    # ── Heatmap ──────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])
    cmap = sns.diverging_palette(10, 133, as_cmap=True)   # red → white → green

    sns.heatmap(
        matrix,
        ax=ax_heat,
        cmap=cmap,
        center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Strength Index", "shrink": 0.8}
    )

    ax_heat.set_title(
        "Live Currency Strength Meter\n(Green = Strong, Red = Weak)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax_heat.tick_params(axis="x", rotation=45)
    ax_heat.tick_params(axis="y", rotation=0)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    plt.figtext(0.5, 0.38, f"Last updated: {timestamp}",
                ha="center", fontsize=9, color="gray")

    # ── Legend panel ─────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")
    legend_text = (
        "• Green cells: The base currency is strong against the quote currency\n"
        "• Red cells:   The base currency is weak against the quote currency\n"
        "• Values near 0: The currencies are relatively balanced"
    )
    ax_leg.text(
        0.02, 0.95, "Interpreting the Results",
        transform=ax_leg.transAxes,
        fontsize=11, fontweight="bold", va="top"
    )
    ax_leg.text(
        0.02, 0.65, legend_text,
        transform=ax_leg.transAxes,
        fontsize=10, va="top", linespacing=2.0
    )

    return fig

# ============================================================
#  Live Currency Strength Heatmap  ·  tradingview_ta edition
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from tradingview_ta import TA_Handler, Interval
from datetime import datetime

# ── Config ───────────────────────────────────────────────────
CURRENCIES = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']

PAIRS = [
    ("EURUSD",  "EUR", "USD"), ("GBPUSD",  "GBP", "USD"),
    ("USDJPY",  "USD", "JPY"), ("USDCHF",  "USD", "CHF"),
    ("USDCAD",  "USD", "CAD"), ("AUDUSD",  "AUD", "USD"),
    ("NZDUSD",  "NZD", "USD"), ("EURGBP",  "EUR", "GBP"),
    ("EURJPY",  "EUR", "JPY"), ("EURCHF",  "EUR", "CHF"),
    ("EURCAD",  "EUR", "CAD"), ("EURAUD",  "EUR", "AUD"),
    ("EURNZD",  "EUR", "NZD"), ("GBPJPY",  "GBP", "JPY"),
    ("GBPCHF",  "GBP", "CHF"), ("GBPCAD",  "GBP", "CAD"),
    ("GBPAUD",  "GBP", "AUD"), ("GBPNZD",  "GBP", "NZD"),
    ("CHFJPY",  "CHF", "JPY"), ("CADJPY",  "CAD", "JPY"),
    ("AUDJPY",  "AUD", "JPY"), ("NZDJPY",  "NZD", "JPY"),
    ("AUDCAD",  "AUD", "CAD"), ("AUDCHF",  "AUD", "CHF"),
    ("AUDNZD",  "AUD", "NZD"), ("CADCHF",  "CAD", "CHF"),
    ("NZDCAD",  "NZD", "CAD"), ("NZDCHF",  "NZD", "CHF"),
]

# ── Functions (paste Steps 4–7 here) ─────────────────────────
# fetch_live_rates(), calculate_strength(),
# build_strength_matrix(), plot_heatmap()
# ... (defined above)

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🌐 Live Currency Strength Meter")
    print("=" * 42)

    print("\n📊 Fetching live currency data...")
    rates = fetch_live_rates(PAIRS)

    if not rates:
        print("❌ No data fetched. Check internet connection.")
        exit()

    print("\n🧮 Calculating currency strength...")
    strength = calculate_strength(rates, CURRENCIES)

    print("\n📈 Strength Rankings:")
    for cur, val in sorted(strength.items(), key=lambda x: -x[1]):
        bar  = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else ""
        print(f"  {cur}: {sign}{val:.4f}  {bar}")

    print("\n🎨 Generating heatmap...")
    matrix    = build_strength_matrix(strength, CURRENCIES)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig       = plot_heatmap(matrix, strength, timestamp)

    filename = f"currency_strength_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"\n✅ Done! Chart saved as '{filename}'")
    print("💡 Green = Strong currency, Red = Weak currency")


import time

REFRESH_SECONDS = 60   # update every minute

while True:
    rates     = fetch_live_rates(PAIRS)
    strength  = calculate_strength(rates, CURRENCIES)
    matrix    = build_strength_matrix(strength, CURRENCIES)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig       = plot_heatmap(matrix, strength, timestamp)
    filename  = f"currency_strength_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {filename} — next refresh in {REFRESH_SECONDS}s...")
    time.sleep(REFRESH_SECONDS)

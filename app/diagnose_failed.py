"""Diagnose why specific signals were wrong.

For each failed bearish signal, fetch the historical candles
and show what the indicators looked like at that moment.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ccxt
from app.indicators import calc_adx, calc_ema

SYMBOL = "BTC/USDT:USDT"

FAILED_SIGNALS = [
    {"id": 786, "tf": "30m", "price": 80456.9, "adx": 24.9, "+di": 14.7, "-di": 23.1, "created_at": 1778629791},
    {"id": 776, "tf": "30m", "price": 80613.6, "adx": 39.3, "+di": 13.1, "-di": 20.6, "created_at": 1778611107},
    {"id": 777, "tf": "1h", "price": 80613.6, "adx": 34.6, "+di": 11.2, "-di": 26.6, "created_at": 1778611107},
    {"id": 767, "tf": "30m", "price": 80510.3, "adx": 41.9, "+di": 10.7, "-di": 22.6, "created_at": 1778608418},
]


def fetch_candles_around(ts, timeframe, lookback=200):
    ex = ccxt.binance({"enableRateLimit": True})
    candle_seconds = {"30m": 1800, "1h": 3600, "4h": 14400}
    seconds = candle_seconds[timeframe]
    # since must be ms, ts is in seconds
    since_ms = int((ts - lookback * seconds) * 1000)
    all_candles = []
    for _ in range(5):  # max 5 pagination rounds
        ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe, since=since_ms, limit=1000)
        if not ohlcv:
            break
        for c in ohlcv:
            ts_c = c[0] / 1000
            if ts_c <= ts:
                all_candles.append({
                    "timestamp": ts_c,
                    "open": c[1], "high": c[2], "low": c[3],
                    "close": c[4], "volume": c[5],
                })
        if ohlcv[-1][0] / 1000 >= ts:
            break
        since_ms = ohlcv[-1][0] + 1
    return all_candles[-lookback:]


def analyze(sig):
    candles = fetch_candles_around(sig["created_at"], sig["tf"], 200)
    if not candles:
        print(f"  [{sig['id']}] No candles found")
        return

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    adx_data = calc_adx(highs, lows, closes, 14)
    adx_fast = calc_adx(highs, lows, closes, 10)
    adx_slow = calc_adx(highs, lows, closes, 21)
    ema50_vals = calc_ema(closes, 50)
    ema20_vals = calc_ema(closes, 20)
    ema50 = ema50_vals[-1] if ema50_vals else 0
    ema20 = ema20_vals[-1] if ema20_vals else 0

    price = closes[-1]
    price_vs_ema50 = (price - ema50) / max(ema50, 1) * 100
    price_vs_ema20 = (price - ema20) / max(ema20, 1) * 100

    last_20 = closes[-20:]
    trend_20 = (last_20[-1] - last_20[0]) / max(last_20[0], 1) * 100

    # ADX trend over last candles
    adx_history = []
    for i in range(max(0, len(closes) - 34), len(closes)):
        h_slice = highs[max(0, i - 13):i + 1]
        l_slice = lows[max(0, i - 13):i + 1]
        c_slice = closes[max(0, i - 13):i + 1]
        if len(h_slice) >= 14:
            d = calc_adx(h_slice, l_slice, c_slice, 14)
            adx_history.append(d["adx"])

    # Find signal candle
    signal_idx = None
    for i, c in enumerate(candles):
        if abs(c["timestamp"] - sig["created_at"]) < 3600:
            signal_idx = i
            break

    di_spread = abs(adx_data["plus_di"] - adx_data["minus_di"])
    is_bull = price > ema50 and adx_data["plus_di"] > adx_data["minus_di"]
    is_bear = price < ema50 and adx_data["minus_di"] > adx_data["plus_di"]

    print(f"\n{'='*60}")
    print(f"Signal #{sig['id']} | {sig['tf']} bearish | Price: {price:.1f}")
    print(f"  Signal ADX: {sig['adx']:.1f} | +DI: {sig['+di']:.1f} | -DI: {sig['-di']:.1f}")
    print(f"  Computed ADX: {adx_data['adx']:.1f} | +DI: {adx_data['plus_di']:.1f} | -DI: {adx_data['minus_di']:.1f}")
    print(f"  ADX fast(10): {adx_fast['adx']:.1f} | ADX slow(21): {adx_slow['adx']:.1f}")
    print(f"  EMA20: {ema20:.1f} (price {price_vs_ema20:+.2f}%)")
    print(f"  EMA50: {ema50:.1f} (price {price_vs_ema50:+.2f}%)")
    print(f"  Last 20 candles trend: {trend_20:+.2f}%")
    print(f"  DI spread: {di_spread:.1f}")
    print(f"  Macro: is_bull={is_bull} is_bear={is_bear}")
    print(f"  -> price {'above' if price > ema50 else 'below'} EMA50, DI {'bullish' if adx_data['plus_di'] > adx_data['minus_di'] else 'bearish'}")

    if signal_idx:
        pre = candles[max(0, signal_idx - 10):signal_idx + 1]
        post = candles[signal_idx + 1:signal_idx + 20]
        if pre:
            print(f"  Price BEFORE: {pre[0]['close']:.1f} -> {pre[-1]['close']:.1f} ({(pre[-1]['close']-pre[0]['close'])/pre[0]['close']*100:+.2f}%)")
        if post:
            print(f"  Price AFTER:  {post[0]['close']:.1f} -> {post[-1]['close']:.1f} ({(post[-1]['close']-post[0]['close'])/post[0]['close']*100:+.2f}%)")
            mx = max(c["high"] for c in post)
            print(f"  Max high after: {mx:.1f} (+{(mx-post[0]['close'])/post[0]['close']*100:+.2f}%)")


def main():
    print("Diagnosing failed bearish signals...")
    for sig in FAILED_SIGNALS:
        analyze(sig)


if __name__ == "__main__":
    main()

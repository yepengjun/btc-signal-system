"""Backtest the signal engine over a historical period.

Walks through historical 4h candles, at each step fetches the corresponding
30m/1h windows, runs the signal engine logic, and records what action
would have been taken. Then compares against actual price movement to
evaluate signal quality.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ccxt
from app.indicators import calc_adx, calc_atr, calc_atr_series, calc_atr_percentile
from app.evolution import get_active_thresholds

SYMBOL = "BTC/USDT:USDT"
START_TS = int(datetime(2026, 3, 29, tzinfo=timezone.utc).timestamp()) * 1000
END_TS = int(datetime(2026, 5, 13, 23, 59, tzinfo=timezone.utc).timestamp()) * 1000

ex = ccxt.binance({"enableRateLimit": True})


def fetch_all_klines(timeframe: str) -> list[dict]:
    """Fetch all candles in the period, handling pagination."""
    all_candles = []
    since = START_TS
    while since < END_TS:
        ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        for c in ohlcv:
            ts = c[0]
            if ts >= END_TS:
                break
            if ts >= START_TS:
                all_candles.append({
                    "timestamp": ts / 1000,
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "volume": c[5],
                })
        last_ts = ohlcv[-1][0]
        since = last_ts + 1
        print(f"  {timeframe}: fetched {len(all_candles)} candles so far...")
    return all_candles


def analyze_timeframe(candles: list[dict], timeframe: str, thresholds: dict) -> dict:
    """Replicate _analyze_single_timeframe logic."""
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    base_trending = thresholds.get("adx_trending_threshold", 25)
    base_forming = thresholds.get("adx_forming_threshold", 20)

    adx_data = calc_adx(highs, lows, closes, period=14)
    adx_fast_data = calc_adx(highs, lows, closes, period=10)
    adx_slow_data = calc_adx(highs, lows, closes, period=21)

    atr_series = calc_atr_series(highs, lows, closes, period=14)
    tf_minutes = {"30m": 30, "1h": 60, "4h": 240}
    vol_window = round(14 * 24 * 60 / tf_minutes.get(timeframe, 60))
    vol_percentile = calc_atr_percentile(atr_series, window=vol_window)

    vol_adj_factor = thresholds.get("vol_adjustment_factor", 0.1)
    vol_adj = (vol_percentile - 50) * vol_adj_factor
    trending_adj = max(-5, min(5, vol_adj))
    forming_adj = max(-3, min(3, vol_adj * 0.6))
    adx_trending = min(base_trending + trending_adj, 34)
    adx_forming = min(base_forming + forming_adj, 28)

    adx = adx_data["adx"]
    adx_fast = adx_fast_data["adx"]
    adx_slow = adx_slow_data["adx"]
    plus_di = adx_data["plus_di"]
    minus_di = adx_data["minus_di"]

    current_price = closes[-1] if closes else 0
    atr_val = calc_atr(highs, lows, closes, period=14)
    atr_pct = (atr_val / max(current_price, 1)) * 100

    di_spread = abs(plus_di - minus_di)
    min_di_spread = thresholds.get("min_di_spread", 3)
    if timeframe in ("30m", "1h"):
        effective_min_di = max(min_di_spread, 5)
    else:
        effective_min_di = min_di_spread

    dx_series = adx_data.get("dx_series", [])
    momentum = "稳定"
    if len(dx_series) >= 6:
        recent_avg = sum(dx_series[-3:]) / 3
        earlier_avg = sum(dx_series[-6:-3]) / 3
        diff = recent_avg - earlier_avg
        if diff > 3:
            momentum = "加速"
        elif diff > 0.5:
            momentum = "稳定"
        elif diff > -2:
            momentum = "减弱"
        else:
            momentum = "衰竭"

    if adx_fast > adx_slow + 3 and momentum in ("衰竭", "减弱"):
        momentum = "减弱" if adx_fast - adx_slow < 8 else "稳定"

    if adx >= adx_trending:
        regime = "trending"
    elif adx >= adx_forming:
        regime = "forming"
    else:
        regime = "ranging"

    if di_spread < effective_min_di and regime in ("trending", "forming"):
        regime = "ranging"

    if regime in ("trending", "forming"):
        direction = "bullish" if plus_di > minus_di else "bearish"
    else:
        direction = None

    return {
        "adx": adx,
        "adx_fast": adx_fast,
        "adx_slow": adx_slow,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "regime": regime,
        "direction": direction,
        "momentum": momentum,
        "price": current_price,
        "vol_percentile": vol_percentile,
        "di_spread": di_spread,
        "atr": atr_val,
        "atr_pct": atr_pct,
    }


def generate_signal_at_point(candles_30m: list[dict], candles_1h: list[dict],
                              candles_4h: list[dict], thresholds: dict) -> dict:
    """Replicate the multi-TF alignment and action logic from signal_engine.py."""
    h30 = analyze_timeframe(candles_30m, "30m", thresholds)
    h1 = analyze_timeframe(candles_1h, "1h", thresholds)
    h4 = analyze_timeframe(candles_4h, "4h", thresholds)

    h4_reg = h4["regime"]
    h1_reg = h1["regime"]
    h30_reg = h30["regime"]

    action = "观望"
    side = "观望"
    reason = ""

    if h4_reg == "trending":
        if h4["direction"] == "bullish":
            if h1["direction"] == "bullish":
                action = "做多"
                side = "多"
                reason = f"4h+1h uptrend, ADX={h4['adx']:.0f}"
            else:
                action = "谨慎持有"
                side = "多"
                reason = f"4h uptrend but 1h not aligned"
        elif h4["direction"] == "bearish":
            if h1["direction"] == "bearish":
                action = "做空"
                side = "空"
                reason = f"4h+1h downtrend, ADX={h4['adx']:.0f}"
            else:
                action = "谨慎持有"
                side = "空"
                reason = f"4h downtrend but 1h not aligned"
    elif h4_reg == "forming" and h1_reg == "trending":
        if h4["direction"] and h4["direction"] != h1["direction"]:
            action = "观望"
            reason = "4h方向与1h相反"
        else:
            h4["direction"] = h1["direction"]
            action = "轻仓试探"
            side = "多" if h1["direction"] == "bullish" else "空"
            reason = f"4h forming + 1h {'bullish' if h1['direction'] == 'bullish' else 'bearish'}"
    elif h4_reg == "ranging":
        h1_dir = h1.get("direction")
        h30_dir = h30.get("direction")
        if (h1_reg in ("trending",) and h1_dir
                and h30_dir == h1_dir
                and h1["di_spread"] > 8
                and h4["di_spread"] >= 3):
            action = "轻仓试单"
            side = "多" if h1_dir == "bullish" else "空"
            reason = f"4h ranging + 1h/30m breakout {'bullish' if h1_dir == 'bullish' else 'bearish'}"
        else:
            action = "观望"
            reason = f"4h ranging (ADX={h4['adx']:.0f})"
    elif h4_reg == "exhaustion":
        action = "观望"
        reason = f"4h exhaustion (ADX={h4['adx']:.0f})"

    return {
        "action": action,
        "side": side,
        "reason": reason,
        "h4": h4,
        "h1": h1,
        "h30": h30,
    }


def main():
    print("=" * 80)
    print(f"Backtesting {SYMBOL} from 2026-03-29 to 2026-05-13")
    print("=" * 80)

    # Fetch all klines
    print("\nFetching 4h klines...")
    candles_4h = fetch_all_klines("4h")
    print(f"Total 4h candles: {len(candles_4h)}")

    print("\nFetching 1h klines...")
    candles_1h = fetch_all_klines("1h")
    print(f"Total 1h candles: {len(candles_1h)}")

    print("\nFetching 30m klines...")
    candles_30m = fetch_all_klines("30m")
    print(f"Total 30m candles: {len(candles_30m)}")

    thresholds = get_active_thresholds()

    results = []
    signal_count = 0
    long_signals = 0
    short_signals = 0
    wait_signals = 0
    trend_bullish_count = 0
    trend_bearish_count = 0
    ranging_count = 0
    forming_count = 0

    for i in range(20, len(candles_4h)):
        ts = candles_4h[i]["timestamp"]

        window_4h = candles_4h[max(0, i - 100):i + 1]
        if len(window_4h) < 30:
            continue

        ts_1h_cutoff = ts - 8 * 100 * 3600
        window_1h = [c for c in candles_1h if c["timestamp"] <= ts and c["timestamp"] >= ts_1h_cutoff]
        window_1h = window_1h[-100:] if len(window_1h) > 100 else window_1h
        if len(window_1h) < 20:
            continue

        ts_30m_cutoff = ts - 16 * 100 * 1800
        window_30m = [c for c in candles_30m if c["timestamp"] <= ts and c["timestamp"] >= ts_30m_cutoff]
        window_30m = window_30m[-200:] if len(window_30m) > 200 else window_30m
        if len(window_30m) < 20:
            continue

        sig = generate_signal_at_point(window_30m, window_1h, window_4h, thresholds)

        h4 = sig["h4"]
        h1 = sig["h1"]
        h30 = sig["h30"]

        if h4["regime"] == "trending":
            trend_bullish_count += 1 if h4["direction"] == "bullish" else 0
            trend_bearish_count += 1 if h4["direction"] == "bearish" else 0
        elif h4["regime"] == "ranging":
            ranging_count += 1
        elif h4["regime"] == "forming":
            forming_count += 1

        if sig["side"] == "多":
            signal_count += 1
            long_signals += 1
        elif sig["side"] == "空":
            signal_count += 1
            short_signals += 1
        else:
            wait_signals += 1

        future_candles = candles_4h[i + 1:i + 7]
        if future_candles:
            entry_price = h4["price"]
            max_high = max(c["high"] for c in future_candles)
            max_low = min(c["low"] for c in future_candles)
            exit_price = future_candles[-1]["close"]
            move_pct = (exit_price - entry_price) / entry_price * 100
            max_move_up = (max_high - entry_price) / entry_price * 100
            max_move_down = (entry_price - max_low) / entry_price * 100
        else:
            move_pct = None
            max_move_up = None
            max_move_down = None

        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        results.append({
            "time": dt,
            "price": h4["price"],
            "h4_regime": h4["regime"],
            "h4_direction": h4["direction"],
            "h4_adx": h4["adx"],
            "h1_direction": h1["direction"],
            "h1_adx": h1["adx"],
            "h30_direction": h30["direction"],
            "h30_adx": h30["adx"],
            "action": sig["action"],
            "side": sig["side"],
            "reason": sig["reason"],
            "move_24h_pct": round(move_pct, 2) if move_pct is not None else None,
            "max_move_up_pct": round(max_move_up, 2) if max_move_up is not None else None,
            "max_move_down_pct": round(max_move_down, 2) if max_move_down is not None else None,
        })

    # Summary Statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total evaluation points: {len(results)}")
    print(f"Time range: {results[0]['time']} to {results[-1]['time']}")
    print(f"Start price: {results[0]['price']:.1f}")
    print(f"End price: {results[-1]['price']:.1f}")
    print(f"Total gain: {(results[-1]['price'] - results[0]['price']) / results[0]['price'] * 100:.1f}%")

    print(f"\n4h Regime distribution:")
    print(f"  Trending bullish: {trend_bullish_count} ({trend_bullish_count / len(results) * 100:.1f}%)")
    print(f"  Trending bearish: {trend_bearish_count} ({trend_bearish_count / len(results) * 100:.1f}%)")
    print(f"  Forming:          {forming_count} ({forming_count / len(results) * 100:.1f}%)")
    print(f"  Ranging:          {ranging_count} ({ranging_count / len(results) * 100:.1f}%)")

    print(f"\nSignal distribution:")
    print(f"  Long signals:  {long_signals} ({long_signals / len(results) * 100:.1f}%)")
    print(f"  Short signals: {short_signals} ({short_signals / len(results) * 100:.1f}%)")
    print(f"  Wait:          {wait_signals} ({wait_signals / len(results) * 100:.1f}%)")

    # Signal Quality Analysis
    print(f"\n{'=' * 80}")
    print("SIGNAL QUALITY (what happened after each signal)")
    print("=" * 80)

    long_results = [r for r in results if r["side"] == "多" and r["move_24h_pct"] is not None]
    short_results = [r for r in results if r["side"] == "空" and r["move_24h_pct"] is not None]
    wait_results = [r for r in results if r["side"] == "观望" and r["move_24h_pct"] is not None]

    if long_results:
        avg_move = sum(r["move_24h_pct"] for r in long_results) / len(long_results)
        avg_max_up = sum(r["max_move_up_pct"] for r in long_results) / len(long_results)
        profitable = sum(1 for r in long_results if r["move_24h_pct"] > 0)
        print(f"\nLong signals ({len(long_results)} total):")
        print(f"  Avg 24h move:      {avg_move:+.2f}%")
        print(f"  Avg max favorable: +{avg_max_up:.2f}%")
        print(f"  Profitable:        {profitable}/{len(long_results)} ({profitable / len(long_results) * 100:.1f}%)")

    if short_results:
        avg_move = sum(r["move_24h_pct"] for r in short_results) / len(short_results)
        avg_max_down = sum(r["max_move_down_pct"] for r in short_results) / len(short_results)
        profitable = sum(1 for r in short_results if r["move_24h_pct"] < 0)
        print(f"\nShort signals ({len(short_results)} total):")
        print(f"  Avg 24h move:      {avg_move:+.2f}%")
        print(f"  Avg max favorable: -{avg_max_down:.2f}%")
        print(f"  Profitable:        {profitable}/{len(short_results)} ({profitable / len(short_results) * 100:.1f}%)")

    if wait_results:
        avg_move = sum(r["move_24h_pct"] for r in wait_results) / len(wait_results)
        avg_max_up = sum(r["max_move_up_pct"] for r in wait_results) / len(wait_results)
        print(f"\nWait signals ({len(wait_results)} total):")
        print(f"  Avg 24h move:      {avg_move:+.2f}%")
        print(f"  Avg missed upside: +{avg_max_up:.2f}%")

    # Opportunity Cost Analysis
    print(f"\n{'=' * 80}")
    print("OPPORTUNITY COST: When the system said 'wait' but price went up")
    print("=" * 80)
    big_missed = [r for r in wait_results if r["max_move_up_pct"] and r["max_move_up_pct"] > 1.0]
    if big_missed:
        print(f"Times where system said 'wait' but 24h had >1% upside: {len(big_missed)}")
        for r in big_missed[:10]:
            print(f"  {r['time']} | Price: {r['price']:.0f} | 4h: {r['h4_regime']} ADX={r['h4_adx']:.0f} | "
                  f"+{r['max_move_up_pct']:.2f}% upside available | Reason: {r['reason']}")

    # Detailed Signal Log
    print(f"\n{'=' * 80}")
    print("DETAILED SIGNAL LOG (all non-wait signals)")
    print("=" * 80)
    action_results = [r for r in results if r["side"] != "观望"]
    for r in action_results:
        move_str = f"{r['move_24h_pct']:+.2f}%" if r["move_24h_pct"] is not None else "N/A"
        print(f"  {r['time']} | {r['side']:>2} | Price: {r['price']:.0f} | "
              f"4h: {r['h4_regime']:>10} ADX={r['h4_adx']:5.1f} | "
              f"1h: {r['h1_direction'] or 'none':>8} ADX={r['h1_adx']:5.1f} | "
              f"24h: {move_str:>7} | {r['reason']}")

    # Save full results
    output_path = os.path.join(os.path.dirname(__file__), "backtest_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()

"""Re-verify previously failed signals with the new split-metric logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import get_connection
from app.binance import fetch_klines
from app.config import settings
from app.indicators import calc_adx, calc_atr

VERIFY_CANDLES = {"30m": 3, "1h": 3, "4h": 3}


def _find_signal_candle(candles, created_at):
    best_idx = None
    best_dist = float("inf")
    for i, c in enumerate(candles):
        if c["timestamp"] <= created_at:
            dist = created_at - c["timestamp"]
            if dist < best_dist:
                best_dist = dist
                best_idx = i
    return best_idx


def _calc_price_structure(verify_candle_list, direction):
    if len(verify_candle_list) < 2:
        return True
    if direction == "bearish":
        declines = sum(
            1 for i in range(1, len(verify_candle_list))
            if verify_candle_list[i]["close"] < verify_candle_list[i - 1]["close"]
        )
        last = verify_candle_list[-1]
        cr = last["high"] - last["low"]
        cnl = (last["close"] - last["low"]) / max(cr, 1) < 0.4 if cr > 0 else True
        return (declines >= len(verify_candle_list) // 2) and cnl
    elif direction == "bullish":
        advances = sum(
            1 for i in range(1, len(verify_candle_list))
            if verify_candle_list[i]["close"] > verify_candle_list[i - 1]["close"]
        )
        last = verify_candle_list[-1]
        cr = last["high"] - last["low"]
        cnh = (last["high"] - last["close"]) / max(cr, 1) < 0.4 if cr > 0 else True
        return (advances >= len(verify_candle_list) // 2) and cnh
    return True


def reverify(sig_id, tf, direction, price_at_signal, created_at, regime, adx_at_signal, signal_type):
    candles = fetch_klines(settings.binance_symbol, tf, limit=200)
    if not candles or len(candles) < 10:
        print(f"  [{sig_id}] SKIP: no candles")
        return

    signal_idx = _find_signal_candle(candles, created_at)
    if signal_idx is None:
        print(f"  [{sig_id}] SKIP: no signal candle found")
        return

    n_verify = VERIFY_CANDLES[tf]
    verify_start = signal_idx + 1
    verify_end = min(verify_start + n_verify, len(candles))
    if verify_start >= len(candles):
        print(f"  [{sig_id}] SKIP: too recent")
        return

    verify_candle_list = candles[verify_start:verify_end]
    current_price = verify_candle_list[-1]["close"]

    all_post = candles[signal_idx:verify_end]
    all_high = max(c["high"] for c in all_post)
    all_low = min(c["low"] for c in all_post)

    highs_all = [c["high"] for c in candles]
    lows_all = [c["low"] for c in candles]
    closes_all = [c["close"] for c in candles]
    atr_value = calc_atr(highs_all, lows_all, closes_all, period=14)
    atr_pct_price = (atr_value / max(price_at_signal, 1)) * 100

    tf_atr_mult_min = {"30m": 0.2, "1h": 0.3, "4h": 0.4}
    tf_atr_mult_sig = {"30m": 0.3, "1h": 0.5, "4h": 0.8}
    tf_min_move_floor = {"30m": 0.05, "1h": 0.08, "4h": 0.10}
    min_move_dynamic = max(atr_pct_price * tf_atr_mult_min.get(tf, 0.3), tf_min_move_floor.get(tf, 0.05))
    sig_move_dynamic = max(atr_pct_price * tf_atr_mult_sig.get(tf, 0.5), tf_min_move_floor.get(tf, 0.05) * 1.5)

    is_ranging = direction is None
    actual_direction = "neutral" if is_ranging else ("bullish" if current_price >= price_at_signal else "bearish")
    direction_correct = 1 if is_ranging else (1 if actual_direction == direction else 0)

    move_pct = abs(current_price - price_at_signal) / max(price_at_signal, 1) * 100
    significant_move = move_pct >= sig_move_dynamic

    adx_verify = calc_adx(highs_all, lows_all, closes_all, period=14)
    current_adx = adx_verify["adx"]

    price_structure_ok = _calc_price_structure(verify_candle_list, direction)

    move_sufficient = 1 if move_pct >= min_move_dynamic else 0
    structure_aligned = 1 if price_structure_ok else 0

    # Regime split logic
    if regime == "trending":
        trend_persisted = direction_correct and move_sufficient and structure_aligned
        regime_correct_loose = direction_correct and (move_sufficient or structure_aligned)
    elif regime == "forming":
        adx_rising = current_adx >= (adx_at_signal or 0) + 2
        has_price_action = move_pct >= (min_move_dynamic * 0.5)
        trend_persisted = adx_rising and has_price_action
        regime_correct_loose = 1 if trend_persisted else 0
    elif regime in ("breakout", "exhaustion"):
        if signal_type == "exhaustion_reversal":
            trend_persisted = direction_correct and move_pct >= (min_move_dynamic * 0.5)
        else:
            trend_persisted = significant_move and direction_correct
        regime_correct_loose = direction_correct
    elif regime in ("ranging", "low_volatility", "high_volatility"):
        trend_persisted = 0
        regime_correct_loose = 0
    elif regime == "low_vol_trend":
        trend_persisted = direction_correct and current_adx >= 25
        regime_correct_loose = direction_correct and current_adx >= 20
    else:
        trend_persisted = direction_correct
        regime_correct_loose = direction_correct

    # Longer-term check
    longer_term_valid = 0
    if not trend_persisted and direction and regime in ("trending", "forming"):
        lt_extend = 6
        lt_end = min(signal_idx + VERIFY_CANDLES[tf] + lt_extend + 1, len(candles))
        lt_candles = candles[signal_idx + 1:lt_end]
        if len(lt_candles) > VERIFY_CANDLES[tf]:
            lt_close = lt_candles[-1]["close"]
            if direction == "bullish" and lt_close >= price_at_signal:
                longer_term_valid = 1
            elif direction == "bearish" and lt_close <= price_at_signal:
                longer_term_valid = 1

    # Update DB
    conn = get_connection()
    conn.execute(
        "UPDATE signals SET price_at_verify = ?, actual_trending = ?, "
        "actual_direction = ?, regime_correct = ?, direction_correct = ?, "
        "verified = 1, max_favorable_excursion = ?, max_adverse_excursion = ?, "
        "target_hit = 0, stop_hit = 0, move_pct = ?, "
        "verify_adx = ?, verify_price = ?, verify_time = ?, "
        "move_sufficient = ?, structure_aligned = ?, "
        "regime_correct_loose = ?, longer_term_regime_valid = ? "
        "WHERE id = ?",
        (
            round(current_price, 1),
            1 if trend_persisted else 0,
            actual_direction,
            1 if trend_persisted else 0,
            direction_correct,
            round((all_high - price_at_signal) / max(price_at_signal, 1) * 100, 2) if direction == "bullish" else round((price_at_signal - all_low) / max(price_at_signal, 1) * 100, 2),
            round((price_at_signal - all_low) / max(price_at_signal, 1) * 100, 2) if direction == "bullish" else round((all_high - price_at_signal) / max(price_at_signal, 1) * 100, 2),
            round(move_pct, 2),
            round(current_adx, 1),
            round(current_price, 1),
            0,
            move_sufficient,
            structure_aligned,
            regime_correct_loose,
            longer_term_valid,
            sig_id,
        ),
    )
    conn.commit()
    conn.close()

    S = "OK" if trend_persisted else "FAIL"
    L = "OK" if regime_correct_loose else "FAIL"
    LT = "OK" if longer_term_valid else "FAIL"
    print(f"  [{sig_id}] {tf:>4} {regime:>12} {str(direction):>9} adx={adx_at_signal:.1f} move={move_pct:.3f}% [{S}S][{L}L][{LT}LT] ms={move_sufficient} sa={structure_aligned}")


def main():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, timeframe, direction, price_at_signal, created_at, regime, adx, signal_type "
        "FROM signals WHERE verified=1 AND unverifiable!=1 AND regime_correct=0 "
        "ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    print(f"Re-verifying {len(rows)} previously failed signals with new split logic...\n")
    for r in rows:
        rd = dict(r)
        reverify(
            rd["id"], rd["timeframe"], rd["direction"], rd["price_at_signal"],
            rd["created_at"], rd["regime"], rd["adx"], rd.get("signal_type", "trend_following")
        )

    # Summary
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM signals WHERE verified=1 AND unverifiable!=1").fetchone()[0]
    strict_ok = conn.execute("SELECT COUNT(*) FROM signals WHERE verified=1 AND unverifiable!=1 AND regime_correct=1").fetchone()[0]
    loose_ok = conn.execute("SELECT COUNT(*) FROM signals WHERE verified=1 AND unverifiable!=1 AND regime_correct_loose=1").fetchone()[0]
    dir_ok = conn.execute("SELECT COUNT(*) FROM signals WHERE verified=1 AND unverifiable!=1 AND direction_correct=1").fetchone()[0]
    lt_ok = conn.execute("SELECT COUNT(*) FROM signals WHERE verified=1 AND unverifiable!=1 AND longer_term_regime_valid=1").fetchone()[0]
    ms_ok = conn.execute("SELECT COUNT(*) FROM signals WHERE verified=1 AND unverifiable!=1 AND move_sufficient=1").fetchone()[0]
    sa_ok = conn.execute("SELECT COUNT(*) FROM signals WHERE verified=1 AND unverifiable!=1 AND structure_aligned=1").fetchone()[0]
    conn.close()

    print(f"\n{'='*60}")
    print(f"SUMMARY (total={total})")
    print(f"  strict regime_correct:   {strict_ok}/{total} = {strict_ok/max(total,1)*100:.1f}%")
    print(f"  loose  regime_correct:   {loose_ok}/{total} = {loose_ok/max(total,1)*100:.1f}%")
    print(f"  direction_correct:        {dir_ok}/{total} = {dir_ok/max(total,1)*100:.1f}%")
    print(f"  move_sufficient:          {ms_ok}/{total} = {ms_ok/max(total,1)*100:.1f}%")
    print(f"  structure_aligned:        {sa_ok}/{total} = {sa_ok/max(total,1)*100:.1f}%")
    print(f"  longer_term_valid:        {lt_ok}/{total} = {lt_ok/max(total,1)*100:.1f}%")
    print(f"  gap (loose-strict):       +{(loose_ok-strict_ok)} signals saved by loose mode")


if __name__ == "__main__":
    main()

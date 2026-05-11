from __future__ import annotations

"""Signal engine: multi-timeframe analysis and verdict generation."""

import time

from app.binance import fetch_klines
from app.config import settings
from app.indicators import calc_adx, calc_ema, calc_atr, calc_atr_series, calc_atr_percentile
from app.evolution import get_active_thresholds

TIMEFRAMES = ["30m", "1h", "4h"]


# ─── Position State Machine ───

def _analyze_position_state(
    h4: dict, h1: dict, h30: dict,
    history_rows: list,
    position: dict | None,
    thresholds: dict | None = None,
) -> dict:
    """Analyze position state based on trend evolution history.

    Returns dict with: action, level, reason, reduce_type
    Actions: 开仓 / 加仓 / 继续持有 / 减仓 / 平仓
    """
    thresholds = thresholds or {}
    adx_trending = thresholds.get("adx_trending_threshold", 25)
    adx_forming = thresholds.get("adx_forming_threshold", 20)
    adx_exit = thresholds.get("adx_exit_threshold", 20)
    di_crossover_window = thresholds.get("di_crossover_window", 2)
    adx_drop_reduce = thresholds.get("adx_drop_reduce", 5)

    direction = h4["direction"]
    adx4 = h4["adx"]
    adx1 = h1["adx"]
    momentum = h4["momentum"]
    regime = h4["regime"]
    price = h4["price"]

    # Check for DI crossover (exit signal)
    di_crossover = _check_di_crossover(history_rows)

    # Check for price divergence (price new high but ADX not new high)
    price_divergence = _check_price_divergence(history_rows, position, h4)

    # Get peak ADX in history
    peak_adx_history = _get_peak_adx_from_history(history_rows)

    # ── No position: generate open signal ──
    if not position or position.get("status") != "open":
        if regime == "trending":
            is_fresh = _is_fresh_trend(history_rows)
            if is_fresh:
                return {
                    "action": "开仓",
                    "side": "做多" if direction == "bullish" else "做空",
                    "level": "open",
                    "reason": f"4h {'看涨' if direction == 'bullish' else '看跌'}趋势首次确认，ADX={adx4}",
                    "target": h4.get("forecast", {}).get("target_conserv"),
                    "stop": None,
                }
            else:
                return {
                    "action": "趋势已延续，建议等待回调入场",
                    "level": "wait",
                    "reason": f"4h {'看涨' if direction == 'bullish' else '看跌'}趋势已持续，追高有风险",
                    "target": None,
                    "stop": None,
                }
        elif regime == "forming":
            return {
                "action": "趋势正在形成，建议关注",
                "level": "wait",
                "reason": "等待ADX突破25确认趋势",
                "target": None,
                "stop": None,
            }
        elif regime == "low_vol_trend":
            # low_vol_trend needs 1h confirmation before opening
            h1_regime = h1.get("regime", "")
            h1_dir = h1.get("direction")
            if h1_regime == "trending" and h1_dir == direction:
                return {
                    "action": "开仓（低波动趋势确认）",
                    "side": "做多" if direction == "bullish" else "做空",
                    "level": "open",
                    "reason": f"4h低波动趋势{'看涨' if direction == 'bullish' else '看跌'}，1h趋势确认同向",
                    "target": h4.get("forecast", {}).get("target_conserv"),
                    "stop": None,
                }
            return {
                "action": "低波动趋势未确认，建议观望",
                "level": "wait",
                "reason": f"4h低波动趋势但1h{'方向相反' if h1_dir and h1_dir != direction else '未形成趋势'}，等待确认",
                "target": None,
                "stop": None,
            }
        else:
            return {
                "action": "观望",
                "level": "wait",
                "reason": "市场处于盘整，等待突破",
                "target": None,
                "stop": None,
            }

    # ── Has position: analyze based on trend evolution ──
    pos_side = position.get("side", "")
    pos_dir = "bullish" if pos_side == "long" else "bearish"
    entry_adx = position.get("entry_adx") or adx4
    max_adx = position.get("max_adx") or adx4

    di_diff = abs(h4["plus_di"] - h4["minus_di"])

    # Multi-TF exhaustion reduce: 30m + 1h both exhaustion with same direction as position
    # Short-term pullback risk is high even if 4h is still trending
    h30_regime = h30.get("regime", "")
    h1_regime = h1.get("regime", "")
    h30_dir = h30.get("direction")
    h1_dir = h1.get("direction")

    if h30_regime == "exhaustion" and h1_regime == "exhaustion":
        if pos_dir == direction and h30_dir == pos_dir and h1_dir == pos_dir:
            # Volume confirmation: only reduce if 30m volume is low (缩量衰竭)
            # Normal/high volume exhaustion is less reliable
            h30_vol = h30.get("vol_percentile", 50)
            h1_vol = h1.get("vol_percentile", 50)

            # If 30m is low volume (< 20) and 1h is not high volume (< 70),
            # this is genuine 缩量衰竭 → reduce
            # If 30m has normal/high volume, exhaustion is less reliable → skip
            if h30_vol < 30 and h1_vol < 70:
                return {
                    "action": "减仓保护利润（多周期短期衰竭）",
                    "level": "reduce",
                    "reduce_type": "30pct",
                    "reason": f"30m和1h同时衰竭且30m缩量（30m ADX={h30.get('adx', 0):.0f} 量能{h30_vol:.0f}%），短期回调风险大，建议减仓锁定利润",
                    "target": None,
                    "stop": None,
                }

    adx_change = adx4 - entry_adx
    adx_from_peak = adx4 - max_adx
    aligned = (pos_dir == direction) and regime in ("trending", "low_vol_trend")

    # HIGH VOLATILITY: market enters high volatility → suggest reduce for protection
    if regime == "high_volatility" and pos_dir == direction:
        atr_pct = h4.get("atr_pct", 0)
        reduce_ratio = _calc_reduce_ratio(h4.get("vol_percentile", 50))
        return {
            "action": f"减仓保护利润（减{int(reduce_ratio*100)}%）",
            "level": "reduce",
            "reduce_type": f"{int(reduce_ratio*100)}pct",
            "reason": f"市场进入高波动状态（ATR%={atr_pct:.2f}），建议减仓{int(reduce_ratio*100)}%降低风险",
            "target": None,
            "stop": None,
        }

    # HIGH VOLATILITY + opposite direction → exit
    if regime == "high_volatility" and pos_dir != direction:
        return {
            "action": "平仓离场",
            "level": "exit",
            "reason": f"高波动状态且方向不利，建议立即离场",
            "target": None,
            "stop": None,
        }

    # EXHAUSTION: trend losing steam → suggest take profit
    vol_pct = h4.get("vol_percentile", 50)
    if regime == "exhaustion":
        if pos_dir == direction:
            # Volume-aware: low volume exhaustion = more reliable, reduce more
            reduce_ratio = _calc_reduce_ratio(vol_pct)
            vol_note = "缩量" if vol_pct < 20 else "放量" if vol_pct > 60 else "正常量能"
            return {
                "action": f"减仓获利（减{int(reduce_ratio*100)}%）",
                "level": "reduce",
                "reduce_type": f"{int(reduce_ratio*100)}pct",
                "reason": f"趋势正在衰竭（ADX={adx4:.0f} 量能{vol_note}），建议先减仓{int(reduce_ratio*100)}%锁定利润",
                "target": None,
                "stop": None,
            }
        else:
            return {
                "action": "平仓离场",
                "level": "exit",
                "reason": "趋势衰竭且方向不利，建议立即离场",
                "target": None,
                "stop": None,
            }

    # LOW VOLATILITY TREND: ADX >= trending but vol extremely low — about to expand
    if regime == "low_vol_trend":
        if pos_dir == direction:
            return {
                "action": "继续持有",
                "level": "hold",
                "reason": f"低波动趋势延续（ADX={adx4:.0f} 量能{h4.get('vol_percentile', 50):.0f}%），趋势即将加速",
                "target": h4.get("forecast", {}).get("target_conserv"),
                "stop": None,
            }
        else:
            return {
                "action": "谨慎观察",
                "level": "hold",
                "reason": f"低波动趋势中方向不利，关注突破方向",
                "target": None,
                "stop": None,
            }

    # LOW VOLATILITY: too quiet, hold or wait
    if regime == "low_volatility":
        return {
            "action": "继续持有",
            "level": "hold",
            "reason": f"市场低波动蓄力中，关注突破方向（ATR%={h4.get('atr_pct', 0):.2f}）",
            "target": h4.get("forecast", {}).get("target_conserv"),
            "stop": None,
        }

    # BREAKOUT: new trend starting → good for position
    if regime == "breakout" and pos_dir == direction:
        return {
            "action": "继续持有",
            "level": "add" if adx4 >= 25 and adx_change >= 1 else "hold",
            "reason": f"市场突破盘整，{'看涨' if direction == 'bullish' else '看跌'}趋势启动",
            "target": h4.get("forecast", {}).get("target_conserv"),
            "stop": None,
        }

    if regime == "breakout" and pos_dir != direction:
        return {
            "action": "平仓离场",
            "level": "exit",
            "reason": f"市场突破但方向相反（{'看涨' if direction == 'bullish' else '看跌'}），建议离场",
            "target": None,
            "stop": None,
        }

    if not aligned:
        if direction != pos_dir or regime in ("ranging", "low_volatility"):
            # Check for 2 consecutive direction mismatches
            should_exit, mismatch_count = _check_position_direction_mismatch(
                history_rows, pos_dir, direction
            )

            if regime in ("ranging", "low_volatility"):
                if mismatch_count >= 2:
                    return {
                        "action": "方向连续不一致，建议离场",
                        "level": "exit",
                        "reason": f"最近2个周期方向与{'多' if pos_dir == 'bullish' else '空'}单相反",
                        "target": h4.get("forecast", {}).get("target_conserv"),
                        "stop": None,
                    }
                return {
                    "action": "继续持有",
                    "level": "hold",
                    "reason": f"市场{'低波动蓄力' if regime == 'low_volatility' else '盘整'}中，方向暂未确认{'反转' if mismatch_count else '变化'}",
                    "target": h4.get("forecast", {}).get("target_conserv"),
                    "stop": None,
                }

            if mismatch_count >= 2:
                return {
                    "action": "平仓离场",
                    "level": "exit",
                    "reason": f"趋势方向{'反转' if direction != pos_dir else '消失'}，连续2个周期确认不利方向",
                    "target": None,
                    "stop": None,
                }

            return {
                "action": "继续持有，观察下个周期",
                "level": "hold",
                "reason": f"4h方向{'反转' if direction != pos_dir else '消失'}但仅1个周期，暂等确认",
                "target": h4.get("forecast", {}).get("target_conserv"),
                "stop": None,
            }

    # Direction aligned — determine state

    # EXIT: DI crossover or ADX dropped below 20
    if di_crossover or adx4 < 20:
        return {
            "action": "平仓离场",
            "level": "exit",
            "reason": "DI交叉反转" if di_crossover else f"ADX={adx4}低于20，趋势已衰竭",
            "target": None,
            "stop": None,
        }

    # REDUCE: ADX dropped significantly from peak, or DI narrowed, or price divergence
    if (adx_from_peak <= -adx_drop_reduce and max_adx >= 30) or di_diff < 5 or price_divergence:
        if position.get("reduced"):
            return {
                "action": "清仓离场",
                "level": "exit",
                "reason": "二次减仓信号，趋势衰减确认，建议全部离场",
                "target": None,
                "stop": None,
            }
        reduce_ratio = _calc_reduce_ratio(h4.get("vol_percentile", 50))
        reason_parts = []
        if adx_from_peak <= -adx_drop_reduce and max_adx >= 30:
            reason_parts.append(f"ADX从峰值{max_adx:.0f}回落至{adx4:.0f}")
        if di_diff < 5:
            reason_parts.append(f"DI差值缩小至{di_diff:.1f}，方向确定性降低")
        if price_divergence:
            reason_parts.append("价格新高但ADX未创新高（背离）")
        return {
            "action": f"减仓保护利润（减{int(reduce_ratio*100)}%）",
            "level": "reduce",
            "reduce_type": f"{int(reduce_ratio*100)}pct",
            "reason": "；".join(reason_parts),
            "target": None,
            "stop": None,
        }

    # ADD: ADX rising, momentum accelerating, price making new high/low + strict confirmation
    if adx4 >= 25 and adx_change >= 2 and momentum == "加速":
        vol_pct = h4.get("vol_percentile", 50)
        if _confirm_add_signal(h4, position, price, vol_pct):
            return {
                "action": "加仓",
                "level": "add",
                "reason": f"趋势加速，ADX从{entry_adx:.0f}升至{adx4:.0f}，动量增强，价格创新高",
                "target": h4.get("forecast", {}).get("target_conserv"),
                "stop": None,
            }

    # HOLD: everything stable
    return {
        "action": "继续持有",
        "level": "hold",
        "reason": f"趋势方向一致，ADX={adx4}稳定，动量{momentum or '平稳'}",
        "target": h4.get("forecast", {}).get("target_conserv"),
        "stop": None,
    }


def _is_fresh_trend(history_rows: list) -> bool:
    """Check if current trend is fresh (not trending in last 3 history entries)."""
    if not history_rows:
        return True
    recent = history_rows[:3]
    trending_count = sum(1 for r in recent if r.get("regime") == "trending")
    return trending_count < 2


def _check_forming_confirmation(history_rows: list, current_direction: str) -> bool:
    """Check if last 2 verdict_history entries confirm the forming direction.

    Returns True if direction is consistent (can trust forming signal).
    Returns False if direction flipped recently (forming is unreliable → treat as ranging).
    """
    if not history_rows or len(history_rows) < 2:
        return True  # not enough history, allow

    recent = history_rows[:2]
    for r in recent:
        hist_dir = r.get("direction")
        if hist_dir and hist_dir != current_direction:
            return False
    return True


def _check_position_direction_mismatch(
    history_rows: list, pos_dir: str, current_4h_dir: str
) -> tuple[bool, int]:
    """Check if position direction disagrees with recent verdict_history.

    Returns:
        (should_exit, mismatch_count)
        - mismatch_count >= 2 → immediate exit
        - mismatch_count == 1 → hold for now
    """
    if not history_rows:
        return False, 0

    mismatch = 0
    recent = history_rows[:2]
    for r in recent:
        hist_dir = r.get("direction")
        if hist_dir and hist_dir != pos_dir:
            mismatch += 1

    should_exit = mismatch >= 2
    return should_exit, mismatch


def _check_di_crossover(history_rows: list) -> bool:
    """Check if DI+ and DI- have crossed over in recent history."""
    if len(history_rows) < 2:
        return False
    for i in range(min(2, len(history_rows) - 1)):
        if history_rows[i].get("direction") and history_rows[i + 1].get("direction"):
            if history_rows[i]["direction"] != history_rows[i + 1]["direction"]:
                return True
    return False


def _check_price_divergence(history_rows: list, position: dict | None, h4: dict) -> bool:
    """Check if price is making new high but ADX is not (bearish divergence for longs)."""
    if not history_rows or len(history_rows) < 3:
        return False
    adx_values = [r["adx_4h"] for r in history_rows if r.get("adx_4h") is not None and r["adx_4h"] > 0]
    if not adx_values:
        return False
    peak_adx = max(adx_values)
    current_adx = h4.get("adx", 0)
    current_price = h4.get("price", 0)
    historical_prices = [r["price"] for r in history_rows if r.get("price") is not None and r["price"] > 0]
    peak_price = max(historical_prices) if historical_prices else 0

    if peak_adx > 0 and current_price >= peak_price * 0.995 and current_adx < peak_adx - 5:
        return True
    return False


def _get_peak_adx_from_history(history_rows: list) -> float:
    """Get peak ADX from history within last 24 hours."""
    if not history_rows:
        return 0
    import time
    now = time.time()
    cutoff = now - 24 * 3600  # 24 hours
    adx_values = [
        r.get("adx_4h", 0) for r in history_rows
        if r.get("adx_4h") and r.get("created_at", 0) >= cutoff
    ]
    return max(adx_values) if adx_values else 0


def _is_price_extreme(position: dict, current_price: float) -> bool:
    """Check if price is making a new extreme (high for long, low for short)."""
    entry_price = position.get("entry_price", current_price)
    if position.get("side") == "long":
        return current_price > entry_price * 1.005
    else:
        return current_price < entry_price * 0.995


def _confirm_add_signal(
    h4: dict, position: dict, current_price: float, vol_percentile: float
) -> bool:
    """Confirm add-on: price new extreme + momentum accelerating + ADX rising + not extreme vol.

    All conditions must be met to avoid reckless adding.
    """
    # 1. Price making new high/low
    if not _is_price_extreme(position, current_price):
        return False

    # 2. Momentum must be accelerating
    if h4.get("momentum") != "加速":
        return False

    # 3. ADX rising from entry
    entry_adx = position.get("entry_adx") or h4["adx"]
    if h4["adx"] < entry_adx + 2:
        return False

    # 4. Volatility not at extreme (don't add at vol_percentile > 85)
    if vol_percentile > 85:
        return False

    return True


def _calc_reduce_ratio(vol_percentile: float) -> float:
    """Dynamic reduce ratio based on volatility severity.

    Returns fraction of position to reduce (0.25 ~ 0.75).
    """
    if vol_percentile > 90:
        return 0.75  # severe volatility → reduce 75%
    if vol_percentile > 75:
        return 0.50  # high volatility → reduce 50%
    return 0.25  # moderate → reduce 25%


def _detect_high_volatility(
    candles: list[dict],
    atr_pct: float,
    adx: float,
    plus_di: float,
    minus_di: float,
    volumes: list[float],
) -> bool:
    """Detect high volatility regime.

    High volatility = expanded ATR but no dominant DI direction.
    Conditions (any 2 of 3):
    1. ATR% > 0.8% for 30m/1h or > 1.5% for 4h (relative to price)
    2. DI spread < 8 (no clear direction despite volatility)
    3. Volume spike: recent 5 candles avg volume > 1.5x previous 10 avg
    """
    # Volume spike check
    vol_spike = False
    if len(volumes) >= 15:
        recent_vol = sum(volumes[-5:]) / 5
        prev_vol = sum(volumes[-15:-5]) / 10
        if prev_vol > 0 and recent_vol > prev_vol * 1.5:
            vol_spike = True

    # DI spread narrow (volatility without direction)
    di_spread_narrow = abs(plus_di - minus_di) < 8

    # High ATR percentage (depends on timeframe)
    # ATR% thresholds: BTC typically ~0.5-1% per candle
    high_atr_pct = atr_pct > 1.2

    # Need at least 2 of 3 conditions
    conditions_met = sum([high_atr_pct, di_spread_narrow, vol_spike])
    return conditions_met >= 2


def _detect_low_volatility(
    atr_pct: float,
    adx: float,
    plus_di: float,
    minus_di: float,
    volumes: list[float],
    adx_forming_threshold: float = 20,
) -> bool:
    """Detect low volatility / accumulation phase.

    Conditions (all must be met):
    1. ADX < adx_forming_threshold (no clear trend)
    2. ATR% < 0.4% (very tight range, BTC typically 0.5-1%)
    3. DI spread < 10 (no direction dominance)
    4. Volume declining: recent 5 avg < 0.8x previous 10 avg
    """
    if adx >= adx_forming_threshold:
        return False
    low_atr = atr_pct < 0.4
    di_tight = abs(plus_di - minus_di) < 10

    vol_decline = False
    if len(volumes) >= 15:
        recent_vol = sum(volumes[-5:]) / 5
        prev_vol = sum(volumes[-15:-5]) / 10
        if prev_vol > 0 and recent_vol < prev_vol * 0.8:
            vol_decline = True

    # Need: low ATR + (tight DI or declining volume)
    return low_atr and (di_tight or vol_decline)


def _detect_breakout(
    candles: list[dict],
    atr_pct: float,
    adx: float,
    plus_di: float,
    minus_di: float,
    volumes: list[float],
    atr_val: float,
) -> bool:
    """Detect breakout phase — price breaking out of consolidation with volume.

    Strict conditions (ALL must be met):
    1. ADX 20-30 (trend starting but not yet strong)
    2. Price breaks 20-candle high/low by at least ATR (not just touching)
    3. Volume spike: recent 5 avg > 1.5x previous 10 avg (raised from 1.3x)
    4. ATR% rising: current ATR% > 0.6% (expanding from quiet)
    """
    if adx < 20 or adx > 30:
        return False

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    # Price breaking 20-candle range by at least ATR (not just touching)
    recent_high = max(highs[-21:-1]) if len(highs) >= 21 else max(highs[:-1])
    recent_low = min(lows[-21:-1]) if len(lows) >= 21 else min(lows[:-1])
    current = closes[-1]
    price_breakout = (current > recent_high + atr_val) or (current < recent_low - atr_val)

    # Volume spike — raised threshold from 1.3x to 1.5x
    vol_spike = False
    if len(volumes) >= 15:
        recent_vol = sum(volumes[-5:]) / 5
        prev_vol = sum(volumes[-15:-5]) / 10
        if prev_vol > 0 and recent_vol > prev_vol * 1.5:
            vol_spike = True

    # ATR expanding from quiet
    atr_expanding = atr_pct > 0.6

    # ALL 3 conditions must be met (price_breakout, vol_spike, atr_expanding)
    return price_breakout and vol_spike and atr_expanding


def _detect_exhaustion(
    adx: float,
    momentum: str,
    dx_series: list[float] | None,
    adx_fast: float,
    adx_slow: float,
    vol_percentile: float,
) -> bool:
    """Detect trend exhaustion — trend still above threshold but losing steam.

    Conditions (all must be met):
    1. ADX >= 35 (raised from 30 — was in STRONG trend, not just moderate)
    2. Momentum is "减弱" or "衰竭"
    3. ADX actually declining: adx_fast < adx_slow (ADX falling = trend weakening)

    Volume confirmation:
    - Low volume (vol_percentile < 15): exhaustion more reliable (缩量衰竭)
    - Normal/high volume (vol_percentile > 50): need stronger ADX decline evidence

    If momentum says "衰竭" but ADX is still rising (adx_fast > adx_slow + 2),
    the trend is still strengthening — not exhaustion.
    """
    if adx < 35:
        return False
    if momentum not in ("减弱", "衰竭"):
        return False

    # ADX must actually be declining, not just slowing down.
    # adx_fast(10) < adx_slow(21) means ADX is genuinely falling (trend weakening)
    # adx_fast > adx_slow means ADX is rising (trend strengthening) — NOT exhaustion
    if adx_fast >= adx_slow:
        return False  # ADX still rising or flat — trend strengthening

    # ADX peak drop confirmation
    if dx_series and len(dx_series) >= 10:
        peak_dx = max(dx_series[-10:])
        current_dx = dx_series[-1]
        if peak_dx - current_dx >= 5:  # raised from 3 to 5
            return True  # ADX dropped significantly from peak

    # Without sufficient peak drop, require momentum == "衰竭" AND low volume
    if momentum == "衰竭" and vol_percentile < 50:
        return True

    return False


def _analyze_single_timeframe(candles: list[dict], timeframe: str, thresholds: dict | None = None) -> dict:
    """Analyze a single timeframe and return analysis dict."""
    thresholds = thresholds or {}
    base_trending = thresholds.get("adx_trending_threshold", 25)
    base_forming = thresholds.get("adx_forming_threshold", 20)
    adx_exit = thresholds.get("adx_exit_threshold", 20)

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # Primary ADX (period 14)
    adx_data = calc_adx(highs, lows, closes, period=14)
    # Dual-track ADX
    adx_fast = calc_adx(highs, lows, closes, period=10)["adx"]
    adx_slow = calc_adx(highs, lows, closes, period=21)["adx"]

    # Volatility percentile (per-timeframe, 14-day fixed span)
    atr_series = calc_atr_series(highs, lows, closes, period=14)
    tf_minutes = {"30m": 30, "1h": 60, "4h": 240}
    vol_window = round(14 * 24 * 60 / tf_minutes.get(timeframe, 60))
    vol_percentile = calc_atr_percentile(atr_series, window=vol_window)

    # Volatility-adaptive effective thresholds (per-timeframe)
    # Low vol → lower threshold (easier to call trend); high vol → higher threshold
    vol_adj_factor = thresholds.get("vol_adjustment_factor", 0.1)
    vol_adj = (vol_percentile - 50) * vol_adj_factor
    trending_adj = max(-5, min(5, vol_adj))
    forming_adj = max(-3, min(3, vol_adj * 0.6))
    adx_trending = min(base_trending + trending_adj, 34)  # hard cap: never disable a timeframe
    adx_forming  = min(base_forming + forming_adj, 28)

    # ADX decay detection
    adx_decay = adx_fast - adx_slow

    atr_val = calc_atr(highs, lows, closes, period=14)
    ema20 = calc_ema(closes, 20)

    adx = adx_data["adx"]
    plus_di = adx_data["plus_di"]
    minus_di = adx_data["minus_di"]

    # ─── Market Regime Detection (6 states) ───
    # trending, ranging, high_volatility, low_volatility, breakout, exhaustion
    current_price = closes[-1]
    atr_pct = (atr_val / max(current_price, 1)) * 100  # ATR as % of price
    dx_series = adx_data.get("dx_series", [])

    # Compute momentum before regime detection
    # Momentum: compare two non-overlapping DX windows
    # recent: last 3 candles, earlier: 3 candles before that (no overlap)
    dx_len = len(dx_series)
    momentum = "稳定"
    if dx_len >= 6:
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

    # Override: if ADX_fast > adx_slow, trend is genuinely strengthening.
    # DX momentum may be slowing but the trend itself is still building up.
    # Don't show "衰竭" when ADX is rising — misleading for users.
    if adx_fast > adx_slow + 3 and momentum in ("衰竭", "减弱"):
        momentum = "减弱" if adx_fast - adx_slow < 8 else "稳定"

    is_high_vol = _detect_high_volatility(candles, atr_pct, adx, plus_di, minus_di, volumes)
    is_low_vol = _detect_low_volatility(atr_pct, adx, plus_di, minus_di, volumes, adx_forming)
    is_breakout = _detect_breakout(candles, atr_pct, adx, plus_di, minus_di, volumes, atr_val)
    is_exhaustion = _detect_exhaustion(adx, momentum, dx_series, adx_fast, adx_slow, vol_percentile)

    # Regime detection with correct priority:
    # 1. exhaustion    — ADX≥30 + momentum declining + ADX actually falling
    # 2. low_vol_trend — ADX≥trending + vol extremely low (slow trend, about to accelerate)
    #    MUST check BEFORE trending, otherwise ADX≥25 gets captured first
    # 3. breakout      — ADX 18-30 + price breakout + volume expansion
    #    MUST check BEFORE trending, otherwise ADX 25-30 gets captured first
    # 4. trending      — ADX ≥ adx_trending
    # 5. high_vol      — high volatility without direction
    # 6. forming       — ADX ≥ adx_forming
    # 7. low_vol       — low volatility accumulation
    # 8. ranging       — everything else
    if is_exhaustion:
        regime = "exhaustion"
    elif adx >= adx_trending and vol_percentile < 15:
        regime = "low_vol_trend"  # slow trend, low vol, about to expand
    elif is_breakout:
        regime = "breakout"
    elif adx >= adx_trending:
        regime = "trending"
    elif is_high_vol:
        regime = "high_volatility"
    elif adx >= adx_forming:
        regime = "forming"
    elif is_low_vol:
        regime = "low_volatility"
    else:
        regime = "ranging"

    # DI spread filter: reject directional signals if DI spread too small
    di_spread = abs(plus_di - minus_di)
    min_di_spread = thresholds.get("min_di_spread", 3)

    # For short timeframes (30m, 1h), require larger DI spread
    if timeframe in ("30m", "1h"):
        effective_min_di = max(min_di_spread, 5)  # higher bar for short TFs
    else:
        effective_min_di = min_di_spread

    if di_spread < effective_min_di and regime in ("trending", "forming", "low_vol_trend"):
        regime = "ranging"
        direction = None

    # Direction
    if regime in ("trending", "forming", "breakout", "exhaustion", "low_vol_trend"):
        direction = "bullish" if plus_di > minus_di else "bearish"
    else:
        direction = None

    # Momentum already computed above before regime detection

    # Confidence: based on ADX strength and regime
    if regime == "trending":
        confidence = min(70, max(50, int(adx)))
        # Short timeframe trending is less reliable
        if timeframe in ("30m", "1h"):
            confidence = max(35, confidence - 10)
    elif regime == "breakout":
        confidence = 50  # lowered from 60 — breakout false positives
    elif regime == "high_volatility":
        confidence = 45
    elif regime == "exhaustion":
        confidence = 45  # lowered from 55 — reversal unreliable
    elif regime == "forming":
        confidence = 50  # lowered from 55 — forming often flips
    elif regime == "low_volatility":
        confidence = 40
    elif regime == "low_vol_trend":
        confidence = 55  # lowered from 60 — direction can flip
    else:
        confidence = 50

    # Strength
    if regime == "trending":
        if adx >= 35:
            strength = "强"
        elif adx >= 25:
            strength = "中等"
        else:
            strength = "偏弱"
    elif regime == "breakout":
        strength = "突破"
    elif regime == "exhaustion":
        strength = "衰竭"
    elif regime == "low_volatility":
        strength = "极弱"
    elif regime == "low_vol_trend":
        strength = "蓄力"  # low vol trend, building up
    elif momentum == "衰竭":
        strength = "衰竭"
    else:
        strength = "无"

    # Duration: how many candles in current trend
    duration_hours = _estimate_duration(candles, timeframe)

    # Price structure analysis
    price_structure = _analyze_price_structure(candles)

    # Entry position
    entry_position = _analyze_entry_position(candles, ema20)

    # Forecast
    current_price = closes[-1]
    forecast = _calc_forecast(
        current_price, atr_val, timeframe, candles, adx_data, vol_percentile, adx
    )

    return {
        "adx": adx,
        "adx_fast": round(adx_fast, 1),
        "adx_slow": round(adx_slow, 1),
        "adx_decay": round(adx_decay, 1),
        "plus_di": plus_di,
        "minus_di": minus_di,
        "dx_series": adx_data.get("dx_series", [])[-20:],
        "regime": regime,
        "direction": direction,
        "confidence": confidence,
        "strength": strength,
        "momentum": momentum if regime not in ("ranging", "high_volatility", "low_volatility", "low_vol_trend") else None,
        "duration_hours": duration_hours,
        "price": round(current_price, 1),
        "vol_trend": round(
            sum(volumes[-5:]) / max(sum(volumes[-10:-5]), 1), 2
        ),
        "forecast": forecast,
        "squeeze": _check_squeeze(candles, ema20, atr_val, timeframe),
        "price_structure": price_structure,
        "entry_position": entry_position,
        "atr": round(atr_val, 1),
        "atr_pct": round(atr_pct, 2),
        "vol_percentile": round(vol_percentile, 1),
        "di_spread": round(di_spread, 1),
        # Effective thresholds actually used for regime detection
        "effective_trending": round(adx_trending, 1),
        "effective_forming": round(adx_forming, 1),
        "base_trending": base_trending,
        "base_forming": base_forming,
        "trending_adj": round(trending_adj, 1),
        "forming_adj": round(forming_adj, 1),
    }


def _estimate_duration(candles: list[dict], timeframe: str) -> float:
    """Estimate how many hours the current trend has lasted."""
    tf_hours = {"30m": 0.5, "1h": 1.0, "4h": 4.0}
    hours = tf_hours.get(timeframe, 1.0)
    closes = [c["close"] for c in candles]
    if len(closes) < 3:
        return hours
    trend_count = 0
    last_dir = None
    for i in range(len(closes) - 2, 0, -1):
        curr_dir = "up" if closes[i + 1] >= closes[i] else "down"
        if last_dir is None:
            last_dir = curr_dir
            trend_count = 1
        elif curr_dir == last_dir:
            trend_count += 1
        else:
            break
    return round(trend_count * hours, 1)


def _analyze_price_structure(candles: list[dict]) -> dict:
    """Analyze high/low slope, retracement, structure type."""
    recent = candles[-20:]
    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    high_slope = (highs[-1] - highs[0]) / max(highs[0], 1) * 100 if highs[0] else 0
    low_slope = (lows[-1] - lows[0]) / max(lows[0], 1) * 100 if lows[0] else 0

    hh = any(highs[i] > highs[i - 1] for i in range(1, len(highs)))
    hl = any(lows[i] > lows[i - 1] for i in range(1, len(lows)))
    lh = any(highs[i] < highs[i - 1] for i in range(1, len(highs)))
    ll = any(lows[i] < lows[i - 1] for i in range(1, len(lows)))

    if hh and hl:
        ptype = "bullish"
    elif lh and ll:
        ptype = "bearish"
    else:
        ptype = "neutral"

    range_high = max(highs)
    range_low = min(lows)
    current = recent[-1]["close"]
    retracement = (current - range_low) / max(range_high - range_low, 1)

    return {
        "type": ptype,
        "lower_highs": lh,
        "higher_lows": hl,
        "high_slope": round(high_slope, 2),
        "low_slope": round(low_slope, 2),
        "retracement": round(retracement, 2),
        "adx_divergence": False,
    }


def _analyze_entry_position(candles: list[dict], ema20: list[float]) -> dict:
    """Analyze where current price sits in range."""
    recent = candles[-20:]
    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]
    current = candles[-1]["close"]

    range_high = max(highs)
    range_low = min(lows)
    range_size = range_high - range_low
    percentile = (
        round((current - range_low) / max(range_size, 1) * 100, 1)
        if range_size > 0
        else 50
    )

    short_move = (current - highs[-1]) / max(highs[-1], 1) * 100

    ema_last = ema20[-1] if ema20 else current
    dist_ema = (current - ema_last) / max(ema_last, 1) * 100

    return {
        "range_high": round(range_high, 1),
        "range_low": round(range_low, 1),
        "percentile": percentile,
        "short_move_pct": round(short_move, 3),
        "dist_from_ema": round(dist_ema, 3),
    }


def _calc_forecast(
    current_price: float,
    atr: float,
    timeframe: str,
    candles: list[dict],
    adx_data: dict,
    vol_percentile: float = 50,
    adx: float = 25,
) -> dict:
    """Calculate price forecast (conservative and aggressive targets) with dynamic stop."""
    n_candles = {"30m": 20, "1h": 15, "4h": 8}
    n = n_candles.get(timeframe, 15)
    start_price = candles[-n]["close"] if len(candles) >= n else candles[0]["close"]
    move_so_far = current_price - start_price
    move_pct = move_so_far / start_price * 100

    adx_peak = adx_data["adx"]
    if adx_peak >= 30:
        factor = 1.5
    elif adx_peak >= 25:
        factor = 1.2
    else:
        factor = 0.8

    target_conserv = round(current_price + move_so_far * 0.5 + atr * factor, 1)
    target_aggress = round(current_price + move_so_far + atr * factor * 1.5, 1)

    # Dynamic stop distance: ATR × k_vol × k_adx
    k_vol = 1.0 + (vol_percentile / 100)  # 1.0 ~ 2.0
    k_adx = max(0.8, min(1.5, 1.0 - (adx - 25) / 50))  # ADX high → tighter stop
    stop_distance = atr * 1.5 * k_vol * k_adx

    return {
        "momentum": "稳定",
        "adx_peak": adx_peak,
        "adx_drop": 0.0,
        "adx_recent_peak": adx_peak,
        "atr": round(atr, 1),
        "trend_start_price": round(start_price, 1),
        "move_so_far": round(move_so_far, 1),
        "move_pct": round(move_pct, 2),
        "target_conserv": target_conserv,
        "target_aggress": target_aggress,
        "stop_distance": round(stop_distance, 1),
        "remain_hours": None,
    }


def _check_squeeze(
    candles: list[dict], ema: list[float], atr: float, timeframe: str
) -> dict:
    """Check for Bollinger Band squeeze with compression % and duration."""
    closes = [c["close"] for c in candles[-20:]]
    if len(closes) < 20:
        return {"active": False, "compression": 0, "src": timeframe, "duration_hours": 0}
    sma20 = sum(closes) / len(closes)
    std = (sum((x - sma20) ** 2 for x in closes) / len(closes)) ** 0.5
    upper = sma20 + 2 * std
    lower = sma20 - 2 * std
    band_width = upper - lower
    # Compression: how much the band has narrowed relative to ATR-based width
    recent_atr = atr
    if recent_atr > 0:
        compression = round((1 - band_width / (recent_atr * 4)) * 100, 1)
    else:
        compression = 0
    active = band_width < recent_atr * 1.5
    duration_hours = _estimate_squeeze_duration(candles, atr)
    return {
        "active": active,
        "compression": max(compression, 0),
        "src": timeframe,
        "duration_hours": duration_hours,
    }


def _estimate_squeeze_duration(candles: list[dict], atr: float) -> int:
    """Estimate how many hours the squeeze has been active."""
    closes = [c["close"] for c in candles[-40:]]
    if len(closes) < 25:
        return 0
    count = 0
    for i in range(len(closes) - 1, -1, -1):
        window = closes[max(0, i-19):i+1]
        if len(window) < 20:
            break
        sma = sum(window) / len(window)
        std = (sum((x - sma) ** 2 for x in window) / len(window)) ** 0.5
        if (sma + 2 * std) - (sma - 2 * std) < atr * 1.5:
            count += 1
        else:
            break
    # Each candle is ~4h (since we use 200 candles from the 4h fetch)
    return max(0, round(count * 0.5))


def _order_confidence(rr: float, verdict_confidence: int) -> str:
    """Order signal confidence considers both RR ratio and overall verdict confidence.

    Even with a great RR, if the multi-TF alignment confidence is low (< 45%),
    the order should not be presented as high confidence.
    """
    rr_label = "高" if rr >= 2 else ("中" if rr >= 1.5 else "低")
    if verdict_confidence < 45:
        # Verdict confidence is low — cap order confidence at "中"
        return "中" if rr_label == "高" else rr_label
    if verdict_confidence < 55:
        # Moderate confidence — cap at "高" only if RR is excellent
        if rr_label == "高" and rr >= 2.5:
            return "高"
        return "中" if rr_label == "高" else rr_label
    return rr_label


def generate_verdict(
    history_rows: list | None = None,
    position: dict | None = None,
    market_context: dict | None = None,
) -> dict:
    """Multi-timeframe analysis verdict with circuit breaker and funding/OI filter."""
    # Load evolved thresholds so signal engine self-adjusts
    all_thresholds = get_active_thresholds()

    # External market data
    market_ctx = market_context or {}
    funding_rate = market_ctx.get("funding_rate", 0.0)
    open_interest = market_ctx.get("open_interest", 0.0)

    symbol = settings.binance_symbol
    results = {}
    all_candles_4h = None
    tf_cfg = all_thresholds.get("tf_thresholds", {})

    # Fetch enough candles to cover 14 days for each timeframe
    tf_candle_limits = {"30m": 672, "1h": 336, "4h": 84}

    for tf in TIMEFRAMES:
        candles = fetch_klines(symbol, tf, limit=tf_candle_limits.get(tf, 200))
        # Use per-timeframe evolved thresholds if available, fall back to global
        tf_thresholds = dict(tf_cfg.get(tf, {}))
        if not tf_thresholds:
            tf_thresholds = {
                "adx_trending_threshold": all_thresholds["adx_trending_threshold"],
                "adx_forming_threshold": all_thresholds["adx_forming_threshold"],
            }
        # Copy other non-TF-specific keys from global thresholds
        for k, v in all_thresholds.items():
            if k not in tf_thresholds and k not in ("tf_thresholds",):
                tf_thresholds[k] = v
        results[tf] = _analyze_single_timeframe(candles, tf, tf_thresholds)
        if tf == "4h":
            all_candles_4h = candles

    h4 = results["4h"]
    h1 = results["1h"]
    h30 = results["30m"]

    # Force-correction: if a timeframe is marked unreliable by evolution,
    # downgrade it to ranging so it won't participate in multi-TF alignment.
    for tf in TIMEFRAMES:
        if tf_cfg.get(tf, {}).get("unreliable", False):
            results[tf]["regime"] = "ranging"
            results[tf]["direction"] = None
            results[tf]["confidence"] = 30
            results[tf]["strength"] = "无"

    # Peak ADX from historical signals (for decay tracking)
    peak_adx = _get_peak_adx_from_history(history_rows or [])
    adx_drop = h4["adx"] - peak_adx if peak_adx > 0 else 0.0

    # Forming state confirmation: require last 2 verdict entries to agree on direction
    history = history_rows or []
    for tf_key in results:
        tf_result = results[tf_key]
        if tf_result["regime"] == "forming" and tf_result["direction"]:
            # For 4h use full history_rows; for 1h/30m just use recent signals
            if tf_key == "4h":
                confirmed = _check_forming_confirmation(history, tf_result["direction"])
            else:
                confirmed = True  # skip for non-primary timeframes
            if not confirmed:
                tf_result["regime"] = "ranging"
                tf_result["direction"] = None
                tf_result["confidence"] = 40
                tf_result["strength"] = "无"

    current_price = h4["price"]

    # Per-timeframe effective thresholds (already computed in _analyze_single_timeframe)
    tf_thresholds = {}
    for tf_key in ["30m", "1h", "4h"]:
        tf = results[tf_key]
        tf_thresholds[tf_key] = {
            "vol_percentile": tf.get("vol_percentile"),
            "base_trending": tf.get("base_trending"),
            "base_forming": tf.get("base_forming"),
            "trending_adj": tf.get("trending_adj"),
            "forming_adj": tf.get("forming_adj"),
            "effective_trending": tf.get("effective_trending"),
            "effective_forming": tf.get("effective_forming"),
        }
    vol_threshold_info = {
        "timeframes": tf_thresholds,
    }

    # ─── Circuit Breaker ───
    circuit_breaker_reason = None
    # Flash crash: latest 4h candle range > ATR × 3
    if all_candles_4h and len(all_candles_4h) >= 1:
        last = all_candles_4h[-1]
        candle_range = last["high"] - last["low"]
        if candle_range > h4["atr"] * 3:
            circuit_breaker_reason = f"闪崩检测：K线波幅 {candle_range:.0f} > ATR×3 ({h4['atr']*3:.0f})"

    # Low ADX freeze
    if h4["adx"] < 15 and not circuit_breaker_reason:
        circuit_breaker_reason = f"ADX={h4['adx']:.0f} < 15，趋势冻结"

    # ADX severe decay — trend weakening, reduce confidence but don't block reversal
    if h4.get("adx_decay", 0) < -8 and not circuit_breaker_reason:
        # Don't set circuit breaker; instead, reduce confidence later
        pass  # confidence adjustment handled in alignment rules

    # DI spread too low
    if h4.get("di_spread", 0) < all_thresholds.get("min_di_spread", 3) and not circuit_breaker_reason:
        circuit_breaker_reason = f"DI价差过低（{h4['di_spread']:.1f}），趋势不可靠"

    # Liquidation cascade detection (BTC-specific):
    # past 6 candles (4h×6=24h) price move > 5% + volume > 3x average
    if all_candles_4h and len(all_candles_4h) >= 7 and not circuit_breaker_reason:
        recent_6 = all_candles_4h[-7:]
        price_moves = [c["close"] for c in recent_6]
        if price_moves:
            price_change = abs(price_moves[-1] - price_moves[0]) / max(price_moves[0], 1) * 100
            volumes_6 = [c["volume"] for c in recent_6]
            avg_vol = sum(volumes_6[:-1]) / max(len(volumes_6) - 1, 1)
            latest_vol = volumes_6[-1]
            if price_change > 5 and avg_vol > 0 and latest_vol > avg_vol * 3:
                circuit_breaker_reason = f"清算级联：24h 波动 {price_change:.1f}% + 量能 {latest_vol/avg_vol:.1f}x"

    # ─── Multi-Timeframe Alignment Priority ───
    alignment_rule = None
    h4_reg = h4["regime"]
    h1_reg = h1["regime"]
    orig_h4_direction = h4["direction"]  # save before potential flip

    # Priority 1: 4h trending → 4h dominates
    if h4_reg == "trending":
        alignment_rule = "规则1: 4h trending 主导"
        directions_match = (
            h4["direction"] == h1["direction"]
            and h30["direction"] == h4["direction"]
        )
        if directions_match:
            confidence = min(70, max(h4["confidence"], h1["confidence"]))
        else:
            confidence = max(40, h4["confidence"] - 10)
            alignment_rule += "（1h/30m 方向不一致，信心略降）"

    # Priority 2: 4h forming + 1h trending → early signal
    elif h4_reg == "forming" and h1_reg == "trending":
        alignment_rule = "规则2: 4h forming + 1h trending 早期信号"
        # Override h4 direction to follow 1h trending
        h4["direction"] = h1["direction"]
        h4["regime"] = "forming"  # keep forming flag
        confidence = max(35, h1["confidence"] - 10)

    # Priority 3: 4h ranging/low_vol_trend → check for breakout or low-vol trend
    elif h4_reg == "ranging":
        if h1_reg == "breakout" and h1["regime"] == "breakout":
            alignment_rule = "规则3b: 4h ranging + 1h breakout 轻仓试单"
            confidence = 45
        else:
            alignment_rule = "规则3: 4h ranging 观望"
            confidence = h4["confidence"]
    elif h4_reg == "low_vol_trend":
        alignment_rule = "规则3c: 4h 低波动趋势（即将变盘）"
        # Follow 1h direction for entry
        if h1_reg == "trending" and h1["direction"]:
            h4["direction"] = h1["direction"]
            confidence = max(45, h1["confidence"] - 5)
        else:
            confidence = 50

    # Priority 4: 4h exhaustion → reversal signal
    elif h4_reg == "exhaustion":
        # Check if smaller TFs confirm or contradict
        if h1_reg == "trending" and h1["direction"] == h4["direction"]:
            alignment_rule = "规则4b: 4h衰竭但1h趋势延续，可能是回调"
            # Don't flip direction — treat as pullback within trend
            confidence = max(40, h4["confidence"] - 10)
        else:
            alignment_rule = "规则4: 4h exhaustion 反向信号"
            h4["direction"] = "bullish" if h4["direction"] == "bearish" else "bearish" if h4["direction"] else None
            confidence = max(35, h4["confidence"] - 15)

    # Priority 5: all forming → wait
    elif h4_reg == "forming" and h1_reg == "forming":
        alignment_rule = "规则5: 全部 forming 观望"
        confidence = 35

    # Priority 6: high_volatility → risk management
    elif h4_reg == "high_volatility":
        alignment_rule = "规则6: 4h 高波动预警"
        confidence = max(30, h4["confidence"] - 15)

    # Priority 7: low_volatility → wait for expansion
    elif h4_reg == "low_volatility":
        alignment_rule = "规则7: 4h 低波动蓄力"
        confidence = max(30, h4["confidence"] - 10)
    else:
        alignment_rule = "默认: 使用 4h 独立判断"
        directions_match = (
            h4["direction"] == h1["direction"]
            and h30["direction"] == h4["direction"]
        )
        if directions_match and h4["regime"] == "trending":
            base_conf = max(h4["confidence"], h1["confidence"])
            confidence = min(70, base_conf)
        else:
            confidence = h4["confidence"]

    # Action advice (replaces vague directions_match logic)
    if h4["direction"] == "bullish" and h4["regime"] == "trending":
        if h1["direction"] == "bullish":
            action = "做多"
            side = "多"
            reason = "4h+1h 趋势一致，动量稳定"
        else:
            action = "谨慎持有"
            side = "多"
            reason = "4h 看涨但 1h 方向不一致"
    elif h4["direction"] == "bearish" and h4["regime"] == "trending":
        if h1["direction"] == "bearish":
            action = "做空"
            side = "空"
            reason = "4h+1h 趋势一致看跌"
        else:
            action = "谨慎持有"
            side = "空"
            reason = "4h 看跌但 1h 方向不一致"
    elif h4["regime"] == "breakout":
        # Require 1h direction consistency for breakout entries
        if h1["direction"] and h1["direction"] == h4["direction"]:
            action = "顺势入场"
            side = "多" if h4["direction"] == "bullish" else ("空" if h4["direction"] == "bearish" else "观望")
            reason = f"市场突破盘整区间，{'看' + ('涨' if h4['direction'] == 'bullish' else '跌') if h4['direction'] else '方向待确认'}，1h方向确认，建议轻仓试单"
        else:
            action = "突破但1h未确认"
            side = "观望"
            reason = f"4h出现突破信号，但1h方向{'相反' if h1['direction'] else '未确认'}，等待1h确认"
    elif h4["regime"] == "exhaustion":
        # Exhaustion with 1h trending same direction → pullback within trend
        if h1_reg == "trending" and h1["direction"] == orig_h4_direction:
            action = "谨慎持有"
            side = "多" if h4["direction"] == "bullish" else ("空" if h4["direction"] == "bearish" else "观望")
            reason = "4h 衰竭但 1h 趋势延续，可能是回调而非反转"
        else:
            # Direction was flipped in multi-TF rules → reversal signal
            # Don't open reversal trade immediately; wait for confirmation
            action = "趋势衰竭"
            side = "观望"
            reason = f"趋势正在衰竭（ADX={h4['adx']:.0f}动量减弱），建议获利了结，等待新趋势确认"
    elif h4["regime"] == "low_vol_trend":
        # Low vol trend needs 1h confirmation
        if h1_reg == "trending" and h1["direction"] == h4["direction"]:
            action = "低波动趋势确认"
            side = "多" if h4["direction"] == "bullish" else ("空" if h4["direction"] == "bearish" else "观望")
            reason = f"低波动趋势延续（ADX={h4['adx']:.0f}），1h趋势确认同向，可入场"
        else:
            action = "低波动趋势未确认"
            side = "观望"
            reason = f"4h低波动趋势但1h{'方向相反' if h1.get('direction') and h1['direction'] != h4['direction'] else '未形成趋势'}，等待确认"
    elif h4["regime"] == "high_volatility":
        action = "高波动预警"
        side = "观望"
        reason = f"市场高波动（ATR%={h4['atr_pct']:.2f}），建议观望或减仓"
    elif h4["regime"] == "low_volatility":
        action = "低波动蓄力"
        side = "观望"
        reason = f"市场低波动蓄力（ATR%={h4['atr_pct']:.2f}），关注突破方向"
    elif h4["regime"] == "forming":
        action = "等待确认"
        side = "观望"  # 形成态不开仓
        reason = f"趋势正在形成（{'看涨' if h4['direction'] == 'bullish' else '看跌'}），ADX={h4['adx']:.0f}未达阈值，暂不开仓"
    else:
        action = "观望"
        side = "观望"
        reason = "市场处于盘整，等待突破"

    # Funding rate filter (adaptive, not all-or-nothing)
    oi_warning = False
    oi_divergence = False
    funding_rate_pct = funding_rate * 100  # convert to percentage

    if funding_rate > 0.001:  # > 0.1% → block long (extreme)
        if side == "多":
            side = "观望"
            action = "观望"
            reason = f"资金费率极高 ({funding_rate_pct:.3f}%)，做多风险过大"
    elif funding_rate > 0.0005:  # 0.05% ~ 0.1% → reduce leverage for long
        if side == "多" and abs(funding_rate) > 0.0005:
            # leverage will be reduced below in dynamic leverage section
            pass
    elif funding_rate < -0.001:  # < -0.1% → block short (extreme)
        if side == "空":
            side = "观望"
            action = "观望"
            reason = f"资金费率极低 ({funding_rate_pct:.3f}%)，做空风险过大"
    elif funding_rate < -0.0005:  # -0.1% ~ -0.05% → reduce leverage for short
        if side == "空" and abs(funding_rate) > 0.0005:
            pass

    # OI divergence: OI declining + price moving in signal direction
    # Adds price direction check to distinguish normal profit-taking from
    # genuine divergence (OI down + price going against the signal).
    if open_interest > 0 and side != "观望":
        prev_oi = market_ctx.get("open_interest_prev", 0)
        if prev_oi > 0:
            oi_change = (open_interest - prev_oi) / max(prev_oi, 1)
            # Price moved in signal direction over last few candles
            price_history = market_ctx.get("price_history", [])
            if len(price_history) >= 3:
                price_moved_toward = (
                    price_history[-1] > price_history[-3]
                    if side == "多"
                    else price_history[-1] < price_history[-3]
                )
                # Real divergence: OI down + price toward signal = weakening conviction
                if oi_change < -0.02 and price_moved_toward:
                    oi_divergence = True
                    confidence = max(30, confidence - 20)

    # OI warning: declining OI (less strict)
    if not oi_divergence and open_interest > 0:
        prev_oi = market_ctx.get("open_interest_prev", 0)
        if prev_oi > 0 and open_interest < prev_oi * 0.98:
            oi_warning = True

    target_raw = h4["forecast"]["target_conserv"]
    stop_distance = h4["forecast"].get("stop_distance", h4["atr"] * 1.5)
    # Forecast is always bullish; for short signals, flip target to bearish
    if side == "空":
        target = round(2 * current_price - target_raw, 1)
    else:
        target = target_raw
    stop = (
        round(current_price - stop_distance, 1)
        if side == "多"
        else round(current_price + stop_distance, 1)
    )

    # Entry price: based on timing and direction

    # Initialize current_rr for safety check
    current_rr = None

    # Entry timing: align with trend direction
    entry_pct = h30["entry_position"]["percentile"]
    if side == "多":
        if entry_pct > 75:
            timing = "high"
            timing_label = "接近区间高位"
            timing_reason = f"价格在区间 {entry_pct:.0f}% 位置，追高风险，建议等待回调"
            entry_action = "limit"
        elif entry_pct < 40:
            timing = "low"
            timing_label = "接近区间低位"
            timing_reason = f"价格在区间 {entry_pct:.0f}% 位置，适合现价入场"
            entry_action = "market"
        else:
            timing = "good"
            timing_label = "正在回调"
            timing_reason = f"价格正在回调中（短期 {h30['entry_position']['short_move_pct']:.2f}%），可等企稳入场"
            entry_action = "limit"
    elif side == "空":
        if entry_pct < 25:
            timing = "low"
            timing_label = "接近区间低位"
            timing_reason = f"价格在区间 {entry_pct:.0f}% 位置，追空风险，建议等待反弹"
            entry_action = "limit"
        elif entry_pct > 60:
            timing = "high"
            timing_label = "接近区间高位"
            timing_reason = f"价格在区间 {entry_pct:.0f}% 位置，适合现价做空"
            entry_action = "market"
        else:
            timing = "good"
            timing_label = "正在反弹"
            timing_reason = f"价格正在反弹中（短期 {h30['entry_position']['short_move_pct']:.2f}%），可等反弹入场"
            entry_action = "limit"
    else:
        timing = "good"
        timing_label = "观望中"
        timing_reason = "当前无交易信号"
        entry_action = "wait"

    # Circuit breaker overrides all entry signals
    if circuit_breaker_reason:
        side = "观望"
        action = "观望"
        reason = circuit_breaker_reason
        entry_action = "wait"

    # ─── Position State Machine ───
    pos_state = _analyze_position_state(h4, h1, h30, history_rows or [], position, all_thresholds)

    # 具体下单信号
    atr4 = h4["atr"]
    atr1 = h1["atr"]
    atr30 = h30["atr"]

    # Dynamic leverage based on funding rate (adaptive formula)
    # max_leverage = min(20, 10 / (1 + abs(funding_rate) / 0.01))
    # Normal (fr=0): 10x → capped at 20x
    # fr=0.03%: ~7.4x
    # fr=0.05%: ~6.7x
    # fr=0.1%: ~5.0x
    base_leverage = min(20, 10 / (1 + abs(funding_rate) / 0.01))

    # High volatility leverage cap
    if h4.get("vol_percentile", 50) > 90:
        base_leverage = min(base_leverage, 5)
    leverage_str = f"{int(base_leverage)}x"

    # Entry price: based on timing and direction
    if side == "观望":
        entry_price = None
        entry_note = "趋势不明确，建议观望"
        current_rr = None
    elif side == "多":
        if entry_action == "market":
            entry_price = round(current_price, 1)
            entry_note = "现价入场"
        else:  # limit
            fib_entry = h30["entry_position"]["range_low"] + (h30["entry_position"]["range_high"] - h30["entry_position"]["range_low"]) * 0.382
            entry_price = round(fib_entry, 1)
            if timing == "high":
                entry_note = f"追高风险，等待回调至 {entry_price:.0f} 入场（斐波那契 0.382）"
            else:
                entry_note = f"回调至 {entry_price:.0f} 附近入场"
        # 现价 RR vs 理想入场 RR
        risk_at_current = abs(current_price - stop)
        reward_at_current = abs(target - current_price)
        current_rr = round(reward_at_current / max(risk_at_current, 1), 2)
    else:  # side == "空"
        if entry_action == "market":
            entry_price = round(current_price, 1)
            entry_note = "现价做空"
        else:  # limit
            fib_entry = h30["entry_position"]["range_low"] + (h30["entry_position"]["range_high"] - h30["entry_position"]["range_low"]) * 0.618
            entry_price = round(fib_entry, 1)
            if timing == "low":
                entry_note = f"追空风险，等待反弹至 {entry_price:.0f} 入场（斐波那契 0.618）"
            else:
                entry_note = f"反弹至 {entry_price:.0f} 附近入场"
        risk_at_current = abs(current_price - stop)
        reward_at_current = abs(target - current_price)
        current_rr = round(reward_at_current / max(risk_at_current, 1), 2)

    if side == "多":
        # R/R uses actual planned entry_price (not current_price)
        risk = abs(entry_price - stop)
        reward = abs(target - entry_price)
        rr = round(reward / max(risk, 1), 2)
        # Safety: if current RR < 1.0, market moved too far — force wait
        if current_rr is not None and current_rr < 1.0:
            side = "观望"
            entry_action = "wait"
            entry_note = f"现价盈亏比过低（{current_rr:.2f}），价格已偏离入场位，等待回调至 {entry_price:.0f} 附近再入场（目标 {target:.0f}，止损 {stop:.0f}）"
            base_pct = round(2.0 / max(rr, 1), 1)
            leverage_num = int(base_leverage)
            risk_price_pct = risk / max(entry_price, 1) * 100
            max_safe_pct = 2.0 / max(leverage_num * risk_price_pct / 100, 0.01)
            position_pct = round(min(base_pct, max_safe_pct), 1)
            order_signal = {
                "side": "观望",
                "entry_type": entry_action,
                "entry_price": entry_price,
                "entry_note": entry_note,
                "target": target,
                "stop": stop,
                "risk": risk,
                "reward": reward,
                "rr_ratio": rr,
                "current_rr": current_rr,
                "position_pct": position_pct,
                "leverage": leverage_str,
                "confidence": _order_confidence(rr, confidence),
            }
        else:
            base_pct = round(2.0 / max(rr, 1), 1)
            leverage_num = int(base_leverage)
            risk_price_pct = risk / max(entry_price, 1) * 100
            max_safe_pct = 2.0 / max(leverage_num * risk_price_pct / 100, 0.01)
            position_pct = round(min(base_pct, max_safe_pct), 1)
            order_signal = {
                "side": "做多",
                "entry_type": entry_action,
                "entry_price": entry_price,
                "entry_note": entry_note,
                "target": target,
                "stop": stop,
                "risk": risk,
                "reward": reward,
                "rr_ratio": rr,
                "current_rr": current_rr,
                "position_pct": position_pct,
                "leverage": leverage_str,
                "confidence": _order_confidence(rr, confidence),
            }
    elif side == "空":
        risk = abs(entry_price - stop)
        reward = abs(target - entry_price)
        rr = round(reward / max(risk, 1), 2)
        # Safety: if current RR < 1.0, market moved too far — force wait
        if current_rr is not None and current_rr < 1.0:
            side = "观望"
            entry_action = "wait"
            entry_note = f"现价盈亏比过低（{current_rr:.2f}），价格已偏离入场位，等待反弹至 {entry_price:.0f} 附近再入场（目标 {target:.0f}，止损 {stop:.0f}）"
            base_pct = round(2.0 / max(rr, 1), 1)
            leverage_num = int(base_leverage)
            risk_price_pct = risk / max(entry_price, 1) * 100
            max_safe_pct = 2.0 / max(leverage_num * risk_price_pct / 100, 0.01)
            position_pct = round(min(base_pct, max_safe_pct), 1)
            order_signal = {
                "side": "观望",
                "entry_type": entry_action,
                "entry_price": entry_price,
                "entry_note": entry_note,
                "target": target,
                "stop": stop,
                "risk": risk,
                "reward": reward,
                "rr_ratio": rr,
                "current_rr": current_rr,
                "position_pct": position_pct,
                "leverage": leverage_str,
                "confidence": _order_confidence(rr, confidence),
            }
        else:
            base_pct = round(2.0 / max(rr, 1), 1)
            leverage_num = int(base_leverage)
            risk_price_pct = risk / max(entry_price, 1) * 100
            max_safe_pct = 2.0 / max(leverage_num * risk_price_pct / 100, 0.01)
            position_pct = round(min(base_pct, max_safe_pct), 1)
            order_signal = {
                "side": "做空",
                "entry_type": entry_action,
                "entry_price": entry_price,
                "entry_note": entry_note,
                "target": target,
                "stop": stop,
                "risk": risk,
                "reward": reward,
                "rr_ratio": rr,
                "current_rr": current_rr,
                "position_pct": position_pct,
                "leverage": leverage_str,
                "confidence": _order_confidence(rr, confidence),
            }
    else:
        order_signal = {
            "side": "观望",
            "entry_price": None,
            "entry_note": entry_note,
            "target": None,
            "stop": None,
            "risk": None,
            "reward": None,
            "rr_ratio": None,
            "position_pct": None,
            "leverage": None,
            "confidence": "低",
        }

    summary = f"{'上涨' if h4['direction'] == 'bullish' else '下跌'}趋势（{h4['strength']}），已持续约 {h4['duration_hours']} 小时"

    details = []
    details.append(f"4h ADX={h4['adx']} 确认大趋势")
    details.append(f"1h ADX={h1['adx']} 方向{'一致' if h1['direction'] == h4['direction'] else '不一致'}")
    details.append(f"30m 短线动量与趋势{'一致' if h30['direction'] == h4['direction'] else '背离'}")
    if h4["regime"] == "trending":
        details.append("趋势保持稳定")

    # Ticker info
    closes_4h = all_candles_4h
    prev_close = closes_4h[-2]["close"] if len(closes_4h) >= 2 else current_price
    change_pct = round((current_price - prev_close) / max(prev_close, 1) * 100, 3)
    highs_24 = [c["high"] for c in closes_4h[-6:]]
    lows_24 = [c["low"] for c in closes_4h[-6:]]
    vol_24h = round(sum(c["volume"] for c in closes_4h[-6:]), 3)

    return {
        "verdict": {
            "regime": h4["regime"],
            "direction": h4["direction"],
            "signal": h4["direction"],
            "confidence": confidence,
            "strength": h4["strength"],
            "momentum": h4["momentum"],
            "duration_hours": h4["duration_hours"],
            "summary": summary,
            "details": details,
            "forecast": {
                "momentum": h4["momentum"],
                "adx_peak": round(peak_adx, 1) if peak_adx > 0 else round(h4["adx"], 1),
                "adx_drop": round(-adx_drop, 1) if peak_adx > 0 else 0.0,
                "adx_recent_peak": round(peak_adx, 1) if peak_adx > 0 else round(h4["adx"], 1),
                "atr": h4["forecast"]["atr"],
                "trend_start_price": h4["forecast"]["trend_start_price"],
                "move_so_far": h4["forecast"]["move_so_far"],
                "move_pct": h4["forecast"]["move_pct"],
                "target_conserv": h4["forecast"]["target_conserv"],
                "target_aggress": h4["forecast"]["target_aggress"],
                "stop_distance": h4["forecast"].get("stop_distance"),
                "remain_hours": None,
            },
            "forecast_src": "4h",
            "squeeze": h4["squeeze"],
            "advice": {
                "action": action,
                "side": side,
                "reason": reason,
                "target": target,
                "stop": stop,
                "long_advice": {
                    "action": "做多" if side == "多" else ("谨慎持有" if side == "观望" else "离场"),
                    "reason": reason if side == "多" else ("观望等待方向确认" if side == "观望" else "强看涨趋势，多单应离场"),
                    "target": target if side == "多" else None,
                    "stop": stop if side == "多" else None,
                },
                "short_advice": {
                    "action": "做空" if side == "空" else ("谨慎持有" if side == "观望" else "离场"),
                    "reason": reason if side == "空" else ("观望等待方向确认" if side == "观望" else "强看涨趋势，空单应离场"),
                    "target": target if side == "空" else None,
                    "stop": stop if side == "空" else None,
                },
            },
            "hold_long": {
                "action": pos_state["action"] if position and position.get("side") == "long" else ("放心持有" if side == "多" else ("观望" if side == "观望" else "建议离场")),
                "level": pos_state["level"] if position and position.get("side") == "long" else ("strong" if side == "多" else ("neutral" if side == "观望" else "exit")),
                "reason": pos_state["reason"] if position and position.get("side") == "long" else ("趋势与多单方向一致，信号强烈" if side == "多" else ("趋势不明，建议观望" if side == "观望" else "趋势与空单方向相反，建议及时止损")),
                "target": pos_state.get("target") if position and position.get("side") == "long" else (target if side == "多" else None),
                "stop": stop if side == "多" else None,
                "reduce_type": pos_state.get("reduce_type"),
            },
            "hold_short": {
                "action": pos_state["action"] if position and position.get("side") == "short" else ("放心持有" if side == "空" else ("观望" if side == "观望" else "建议离场")),
                "level": pos_state["level"] if position and position.get("side") == "short" else ("strong" if side == "空" else ("neutral" if side == "观望" else "exit")),
                "reason": pos_state["reason"] if position and position.get("side") == "short" else ("趋势与空单方向一致，信号强烈" if side == "空" else ("趋势不明，建议观望" if side == "观望" else "趋势与多单方向相反，建议及时止损")),
                "target": pos_state.get("target") if position and position.get("side") == "short" else (target if side == "空" else None),
                "stop": stop if side == "空" else None,
                "reduce_type": pos_state.get("reduce_type"),
            },
            "entry_timing": {
                "timing": timing,
                "action": entry_action,
                "label": timing_label,
                "reason": timing_reason,
                "percentile": entry_pct,
                "range_high": h30["entry_position"]["range_high"],
                "range_low": h30["entry_position"]["range_low"],
                "short_move_pct": h30["entry_position"]["short_move_pct"],
                "dist_from_ema": h30["entry_position"]["dist_from_ema"],
            },
            "order_signal": order_signal,
            "market_context": {
                "funding_rate": round(funding_rate_pct, 4),
                "open_interest": open_interest,
                "oi_warning": oi_warning,
                "oi_divergence": oi_divergence,
                "circuit_breaker": circuit_breaker_reason,
                "vol_percentile_4h": h4.get("vol_percentile"),
                "vol_threshold": vol_threshold_info,
                "alignment_rule": alignment_rule,
                "suggested_leverage": f"{int(base_leverage)}x",
            },
            "timeframes": {
                tf: {
                    "adx": results[tf]["adx"],
                    "adx_fast": results[tf].get("adx_fast"),
                    "adx_slow": results[tf].get("adx_slow"),
                    "adx_decay": results[tf].get("adx_decay"),
                    "plus_di": results[tf]["plus_di"],
                    "minus_di": results[tf]["minus_di"],
                    "dx_series": results[tf].get("dx_series", []),
                    "regime": results[tf]["regime"],
                    "direction": results[tf]["direction"],
                    "duration_hours": results[tf]["duration_hours"],
                    "price": results[tf]["price"],
                    "vol_trend": results[tf]["vol_trend"],
                    "momentum": results[tf]["momentum"],
                    "forecast": results[tf]["forecast"],
                    "squeeze": results[tf]["squeeze"],
                    "price_structure": results[tf]["price_structure"],
                    "entry_position": results[tf]["entry_position"],
                    "vol_percentile": results[tf].get("vol_percentile"),
                    "di_spread": results[tf].get("di_spread"),
                    "effective_trending": results[tf].get("effective_trending"),
                    "effective_forming": results[tf].get("effective_forming"),
                    "base_trending": results[tf].get("base_trending"),
                    "base_forming": results[tf].get("base_forming"),
                    "trending_adj": results[tf].get("trending_adj"),
                    "forming_adj": results[tf].get("forming_adj"),
                }
                for tf in TIMEFRAMES
            },
        },
        "ticker": {
            "price": current_price,
            "change_pct": change_pct,
            "high_24h": max(highs_24) if highs_24 else current_price,
            "low_24h": min(lows_24) if lows_24 else current_price,
            "volume_24h": vol_24h,
        },
        "timeframes": results,
    }

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
    di_crossover = _check_di_crossover(history_rows, h4)

    # Check for price divergence (price new high but ADX not new high)
    price_divergence = _check_price_divergence(history_rows, position, h4)

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
            # Half-confirmation: 1h forming + same direction — allow opening
            # with reduced conviction (lower risk_pct in auto-open).
            if h1_regime == "forming" and h1_dir == direction:
                return {
                    "action": "开仓（低波动趋势·1h形成中）",
                    "side": "做多" if direction == 'bullish' else "做空",
                    "level": "open",
                    "reason": f"4h低波动趋势{'看涨' if direction == 'bullish' else '看跌'}，1h趋势正在形成同向",
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
    adx_change = adx4 - entry_adx
    adx_from_peak = adx4 - max_adx

    # Collect reduce conditions and take max ratio.
    # When multiple risk conditions fire simultaneously (e.g. high volatility
    # + exhaustion), we want the most severe response, not the first match.
    # Exit conditions are still checked independently below.
    reduce_conditions: list[tuple[float, str, str, str]] = []  # (ratio, action, reason, reduce_type)

    h30_regime = h30.get("regime", "")
    h1_regime = h1.get("regime", "")
    h30_dir = h30.get("direction")
    h1_dir = h1.get("direction")

    # Multi-TF exhaustion reduce: 30m + 1h both exhaustion with same direction as position
    # Short-term pullback risk is high even if 4h is still trending
    if h30_regime == "exhaustion" and h1_regime == "exhaustion":
        if pos_dir == direction and h30_dir == pos_dir and h1_dir == pos_dir:
            h30_vol = h30.get("vol_percentile", 50)
            h1_vol = h1.get("vol_percentile", 50)
            if h30_vol < 30 and h1_vol < 70:
                ratio = _calc_reduce_ratio(max(h30_vol, h1_vol))
                reduce_conditions.append((
                    ratio,
                    f"减仓保护利润（多周期短期衰竭，减{int(ratio*100)}%）",
                    f"30m和1h同时衰竭且30m缩量（30m ADX={h30.get('adx', 0):.0f} 量能{h30_vol:.0f}%），短期回调风险大",
                    f"{int(ratio*100)}pct",
                ))

    aligned = (pos_dir == direction) and regime in ("trending", "low_vol_trend")

    # HIGH VOLATILITY: market enters high volatility → suggest reduce for protection
    if regime == "high_volatility":
        atr_pct = h4.get("atr_pct", 0)
        vol_pct_hv = h4.get("vol_percentile", 50)
        ratio = _calc_reduce_ratio(vol_pct_hv)
        if pos_dir == direction:
            reduce_conditions.append((
                ratio,
                f"减仓保护利润（高波动，减{int(ratio*100)}%）",
                f"市场进入高波动状态（ATR%={atr_pct:.2f}，波动率分位={vol_pct_hv:.0f}%）",
                f"{int(ratio*100)}pct",
            ))
        else:
            # Opposite direction + high volatility → exit
            return {
                "action": "平仓离场",
                "level": "exit",
                "reason": f"高波动状态且方向不利，建议立即离场",
                "target": None,
                "stop": None,
            }

    # EXHAUSTION: trend losing steam → suggest take profit
    if regime == "exhaustion":
        vol_pct = h4.get("vol_percentile", 50)
        if pos_dir == direction:
            ratio = _calc_reduce_ratio(vol_pct)
            vol_note = "缩量" if vol_pct < 20 else "放量" if vol_pct > 60 else "正常量能"
            reduce_conditions.append((
                ratio,
                f"减仓获利（减{int(ratio*100)}%）",
                f"趋势正在衰竭（ADX={adx4:.0f} 量能{vol_note}）",
                f"{int(ratio*100)}pct",
            ))
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
        # Use same strict add criteria as trending add-on (line 324)
        can_add = adx4 >= adx_trending and (adx_change >= 2 or adx4 >= adx_trending * 1.6) and momentum == "加速" and _confirm_add_signal(h4, position, price, h4.get("vol_percentile", 50))
        return {
            "action": "继续持有" + ("，满足加仓条件" if can_add else ""),
            "level": "add" if can_add else "hold",
            "reason": f"市场突破盘整，{'看涨' if direction == 'bullish' else '看跌'}趋势启动",
            "target": h4.get("forecast", {}).get("target_conserv"),
            "stop": None,
        }

    if regime == "breakout" and pos_dir != direction:
        # Breakout has high false positive rate (confidence 50), so require
        # 1-period mismatch confirmation before exit to avoid stop-loss whipsaw.
        # But if breakout momentum is "加速" (volume expansion + directional body),
        # exit immediately — market is choosing direction.
        breakout_momentum_confirmed = (
            h4.get("momentum") == "加速"
            and h4.get("vol_percentile", 50) > 40
            and di_diff > 8
        )
        if breakout_momentum_confirmed:
            return {
                "action": "平仓离场",
                "level": "exit",
                "reason": f"市场突破且动量确认反向（{'看涨' if direction == 'bullish' else '看跌'}），建议立即离场",
                "target": None,
                "stop": None,
            }
        # Fall through to mismatch check (not aligned path below)

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

    # REDUCE: ADX dropped significantly from peak, or DI narrowed, or price divergence.
    # Threshold: max_adx >= 1.5×trending threshold to reduce false triggers from normal ADX noise.
    # DI narrowing guard: require adx4 > adx_trending to avoid triggering on temporary DI squeeze
    # in low-vol regimes where narrow DI spread is normal and not a reduction signal.
    if (adx_from_peak <= -adx_drop_reduce and max_adx >= adx_trending * 1.5) or (di_diff < 5 and adx4 > adx_trending) or price_divergence:
        if position.get("reduce_count", 0) >= 1:
            return {
                "action": "清仓离场",
                "level": "exit",
                "reason": "二次减仓信号，趋势衰减确认，建议全部离场",
                "target": None,
                "stop": None,
            }
        ratio = _calc_reduce_ratio(h4.get("vol_percentile", 50))
        reason_parts = []
        if adx_from_peak <= -adx_drop_reduce and max_adx >= adx_trending * 1.5:
            reason_parts.append(f"ADX从峰值{max_adx:.0f}回落至{adx4:.0f}")
        if di_diff < 5:
            reason_parts.append(f"DI差值缩小至{di_diff:.1f}，方向确定性降低")
        if price_divergence:
            reason_parts.append("价格新高但ADX未创新高（背离）")
        reduce_conditions.append((
            ratio,
            f"减仓保护利润（减{int(ratio*100)}%）",
            "；".join(reason_parts),
            f"{int(ratio*100)}pct",
        ))

    # ─── Resolve reduce conditions: take max ratio ───
    if reduce_conditions:
        best_ratio, best_action, best_reason, best_type = max(reduce_conditions, key=lambda x: x[0])
        # Combine reasons if multiple conditions triggered
        if len(reduce_conditions) > 1:
            all_reasons = [rc[2] for rc in reduce_conditions]
            combined_reason = " | ".join(all_reasons)
        else:
            combined_reason = best_reason
        return {
            "action": best_action,
            "level": "reduce",
            "reduce_type": best_type,
            "reason": combined_reason,
            "target": None,
            "stop": None,
        }

    # ADD: ADX rising, momentum accelerating, price making new high/low + strict confirmation
    if adx4 >= adx_trending and (adx_change >= 2 or adx4 >= adx_trending * 1.6) and momentum == "加速":
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
    """Check if current trend is fresh (not trending in last 3 history entries).

    Fresh = 0 or 1 of the last 3 entries were "trending".
    "low_vol_trend" does NOT count as trending (it's a forming/continuation stage, not established momentum).
    Established = 2+ of 3 pure "trending" — trend has been running, do NOT open new positions.
    """
    if not history_rows:
        return True
    recent = history_rows[:3]
    # Only pure "trending" counts — "low_vol_trend" is continuation, not established
    trending_count = sum(1 for r in recent if r.get("regime") == "trending")
    return trending_count < 2  # block when 2+ of last 3 were pure trending


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

    # If current 4h direction also disagrees, force exit regardless of history
    if current_4h_dir and current_4h_dir != pos_dir:
        mismatch = max(mismatch + 1, 2)  # floor at 2 for forced exit

    should_exit = mismatch >= 2
    return should_exit, mismatch


def _check_di_crossover(history_rows: list, h4: dict | None = None) -> bool:
    """Check if DI+ and DI- have crossed over recently.

    Primary: use DI series from 4h timeframe to detect actual crossover
    (plus_di crossing minus_di). A single-period crossover is not enough —
    we also check that the DI spread remains narrow for confirmation.

    Fallback: use verdict_history direction flips when DI series unavailable.
    """
    # Prefer actual DI series from the 4h timeframe
    if h4 and h4.get("plus_di_series") and h4.get("minus_di_series"):
        pdi = h4["plus_di_series"]
        mdi = h4["minus_di_series"]
        n = min(len(pdi), len(mdi))
        if n >= 2:
            # Check last 2 periods for actual DI crossover
            pdi_now, mdi_now = pdi[-1], mdi[-1]
            pdi_prev, mdi_prev = pdi[-2], mdi[-2]
            cross_bullish = pdi_prev <= mdi_prev and pdi_now > mdi_now
            cross_bearish = pdi_prev >= mdi_prev and pdi_now < mdi_now
            if cross_bullish or cross_bearish:
                # Confirm: current spread should still be narrow enough
                current_spread = abs(pdi_now - mdi_now)
                if current_spread < 10:  # crossover confirmed, spread still tight
                    return True
            # Near-convergence without full crossover: spread < 3 = dangerous zone
            if abs(pdi_now - mdi_now) < 3:
                # Check if direction flipped in history for confirmation
                for i in range(min(2, len(history_rows) - 1)):
                    if history_rows[i].get("direction") and history_rows[i + 1].get("direction"):
                        if history_rows[i]["direction"] != history_rows[i + 1]["direction"]:
                            return True

        return False

    # Fallback: use verdict direction flips
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

    # Price must be near recent peak (within 0.2%, ~$156 at $78k) AND ADX
    # must have dropped meaningfully (>= 8 points) — tight tolerance avoids
    # false divergence from normal volatility.
    if peak_adx > 0 and current_price >= peak_price * 0.998 and current_adx < peak_adx - 8:
        return True
    return False


def _get_peak_adx_from_history(history_rows: list) -> tuple:
    """Get peak ADX from history within last 8 hours (rolling time window).

    Returns (peak_adx, peak_regime_group) where peak_regime_group is used for
    consistency checking — if the current regime group differs from the peak's,
    the peak is stale and should be ignored.

    Uses an 8-hour cutoff so peak ADX resets quickly after regime switches,
    preventing decay detection from an old peak in a different market state.
    """
    if not history_rows:
        return (0, None)
    import time
    now = time.time()
    cutoff = now - 8 * 3600  # 8 hours

    def _regime_group(r: str) -> str:
        if not r:
            return "other"
        if r in ("trending", "low_vol_trend"):
            return "trend"
        if r in ("forming", "breakout"):
            return "forming"
        if r == "exhaustion":
            return "exhaustion"
        return "other"

    peak_adx = 0
    peak_regime = None
    for r in history_rows:
        ts = r.get("created_at", 0)
        adx = r.get("adx_4h", 0)
        if ts >= cutoff and adx and adx > peak_adx:
            peak_adx = adx
            peak_regime = _regime_group(r.get("regime", ""))

    return (peak_adx, peak_regime)


def _is_price_extreme(position: dict, current_price: float) -> bool:
    """Check if price is making a new extreme relative to the position's peak.

    Uses max_price (highest price seen during position) for longs, or
    min_price for shorts, so add-on signals fire on genuine breakouts
    rather than just being above/below entry after a retracement.
    """
    if position.get("side") == "long":
        benchmark = position.get("max_price") or position.get("entry_price", current_price)
        return current_price > benchmark * 1.005
    else:
        benchmark = position.get("min_price") or position.get("entry_price", current_price)
        return current_price < benchmark * 0.995


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

    # 3. ADX rising from entry, OR already very high (>= 1.6×trending).
    # At high ADX, absolute +2 is hard — trend strength is implied by
    # the high level itself. This prevents missing adds in mature trends
    # where ADX naturally plateaus.
    entry_adx = position.get("entry_adx") or h4["adx"]
    _conf_add_high = max(h4.get("effective_trending", 25) * 1.6, 30)  # floor 30 prevents too-permissive skip
    if h4["adx"] >= _conf_add_high:
        pass  # ADX already very high, no further rise required
    elif h4["adx"] < entry_adx + 2:
        return False

    # 4. Volatility not at extreme (don't add at vol_percentile > 85)
    if vol_percentile > 85:
        return False

    return True


def _calc_reduce_ratio(vol_percentile: float) -> float:
    """Unified dynamic reduce ratio for all reduce conditions.

    Uses linear interpolation between tiers to avoid abrupt jumps:
    vol <= 75% → 25%, vol 75-90% → 25-50%, vol > 90% → 50-75%.

    Called by: multi-TF exhaustion, high volatility, trend exhaustion,
    ADX drop / DI narrowing / price divergence.
    When multiple conditions fire, the caller takes max ratio.
    """
    if vol_percentile <= 75:
        return 0.25
    if vol_percentile <= 90:
        # Linear interpolation: 25% at 75, 50% at 90
        return 0.25 + (vol_percentile - 75) / 15 * 0.25
    # vol > 90: interpolate from 50% to 75%
    cap = min(vol_percentile, 100)
    return 0.50 + (cap - 90) / 10 * 0.25


def _detect_macro_bull(candles_4h: list[dict], h4: dict) -> dict:
    """Detect macro bull/bear regime using price vs EMA50 and DI structure.

    Also detects shallow pullbacks in bull markets: when price dips below
    EMA50 but recent price structure shows we're coming from higher levels
    with a shallow drawdown (< 2%). In these conditions, bearish signals
    on short timeframes are likely pullback traps.

    Returns dict with is_bull, is_bear, ema50, price_vs_ema50_pct, di_bias,
    is_bull_pullback, is_bear_pullback, pullback_depth_pct, reason.
    """
    closes = [c["close"] for c in candles_4h]
    highs = [c["high"] for c in candles_4h]
    if len(closes) < 50:
        return {"is_bull": False, "is_bear": False, "ema50": 0, "price_vs_ema50_pct": 0,
                "di_bias": "neutral", "is_bull_pullback": False, "is_bear_pullback": False,
                "pullback_depth_pct": 0, "reason": "数据不足"}

    ema50_vals = calc_ema(closes, 50)
    ema50 = ema50_vals[-1]
    current_price = closes[-1]
    price_vs_ema50_pct = (current_price - ema50) / max(ema50, 1) * 100

    plus_di = h4.get("plus_di", 0)
    minus_di = h4.get("minus_di", 0)
    di_bias = "bullish" if plus_di > minus_di else "bearish"

    # Require BOTH price position AND DI direction
    is_bull = current_price > ema50 and plus_di > minus_di
    is_bear = current_price < ema50 and minus_di > plus_di

    # ─── Shallow pullback detection ───
    # Even if price is below EMA50, check if we're in a bull market pullback:
    # (1) Recent high (last 48 candles = ~8 days of 4h) was significantly above EMA50
    # (2) Current drawdown from that high is shallow (< 2.5%)
    # (3) The pullback happened recently (within last 12 candles)
    is_bull_pullback = False
    pullback_depth_pct = 0
    lookback = min(48, len(closes))  # ~8 days of 4h
    recent_high = max(highs[-lookback:])
    pullback_depth_pct = (recent_high - current_price) / max(recent_high, 1) * 100

    # High was above EMA50 by at least 1% (confirms we were in bull territory)
    high_vs_ema50 = (recent_high - ema50) / max(ema50, 1) * 100

    # Bull pullback: price dipped near EMA50 (or even below it) in an
    # established bull market. The recent high was well above EMA50 (>1%),
    # and the pullback from that high is shallow (< 3.5%). This catches
    # the classic slow-bull pattern where bearish DI looks strong during
    # a 0.5-1.5% pullback that gets bought up within hours.
    # Don't require price < EMA50 — in slow bulls, even the pullback low
    # may stay above EMA50. Just check the pullback is shallow and we came
    # from bull territory.
    if (high_vs_ema50 > 1.0
            and pullback_depth_pct < 3.5
            and pullback_depth_pct > 0.3):
        is_bull_pullback = True

    # ─── Bear pullback (shallow rally in established bear market) ───
    # Symmetric to bull pullback: recent low was well below EMA50,
    # and the rally from that low is shallow (< 3.5%).
    lows = [c["low"] for c in candles_4h]
    lookback_bear = min(48, len(lows))
    recent_low = min(lows[-lookback_bear:])
    rally_depth_pct = (current_price - recent_low) / max(recent_low, 1) * 100
    low_vs_ema50 = (ema50 - recent_low) / max(ema50, 1) * 100

    is_bear_pullback = False
    if (low_vs_ema50 > 1.0
            and rally_depth_pct < 3.5
            and rally_depth_pct > 0.3):
        is_bear_pullback = True

    if is_bull:
        reason = f"宏观看涨：价格高于EMA50 (+{price_vs_ema50_pct:.1f}%), +DI={plus_di:.0f} > -DI={minus_di:.0f}"
    elif is_bear and is_bear_pullback:
        reason = f"宏观看跌（浅反弹）：价格低于EMA50 ({price_vs_ema50_pct:.1f}%), 但从近期低点仅反弹 {rally_depth_pct:.1f}%, 前期低点低于EMA50 {low_vs_ema50:.1f}%"
    elif is_bear and is_bull_pullback:
        reason = f"宏观看跌（牛拉回穿越EMA50）：价格低于EMA50 ({price_vs_ema50_pct:.1f}%), 但牛拉回浅 ({pullback_depth_pct:.1f}%), 前期高点高于EMA50 {high_vs_ema50:.1f}%"
    elif is_bear:
        reason = f"宏观看跌：价格低于EMA50 ({price_vs_ema50_pct:.1f}%), -DI={minus_di:.0f} > +DI={plus_di:.0f}"
    else:
        reason = f"宏观中性：价格{('高于' if current_price > ema50 else '低于')}EMA50 ({price_vs_ema50_pct:.1f}%), DI{'多头占优' if di_bias == 'bullish' else '空头占优'}"

    return {
        "is_bull": is_bull,
        "is_bear": is_bear,
        "ema50": round(ema50, 1),
        "price_vs_ema50_pct": round(price_vs_ema50_pct, 2),
        "di_bias": di_bias,
        "is_bull_pullback": is_bull_pullback,
        "is_bear_pullback": is_bear_pullback,
        "pullback_depth_pct": round(pullback_depth_pct, 2),
        "reason": reason,
    }


def _detect_high_volatility(
    candles: list[dict],
    atr_pct: float,
    adx: float,
    plus_di: float,
    minus_di: float,
    volumes: list[float],
    vol_percentile: float = None,
) -> bool:
    """Detect high volatility regime.

    High volatility = expanded ATR but no dominant DI direction.

    ATR% is MANDATORY — must exceed the threshold first. Then at least one
    auxiliary condition (DI spread narrow OR volume spike) must also hold.
    This prevents false positives where low ATR% + narrow DI + volume spike
    incorrectly triggers high_volatility.

    Safety gate: if vol_percentile < 20, skip — market is historically
    low-vol and volume spike alone doesn't make it "high volatility".
    """
    # Safety gate: historically low vol cannot be "high volatility"
    # regardless of short-term volume spikes.
    if vol_percentile is not None and vol_percentile < 20:
        return False

    # ATR% is a hard gate — high volatility requires genuinely large price swings.
    # 0.8% is the minimum ATR% for BTC 4h candles; below this, price action
    # is not volatile enough to qualify as "high volatility" regardless of DI/volume.
    if atr_pct < 0.8:
        return False

    # At least one auxiliary condition must confirm the high-vol regime.
    # DI spread narrow (volatility without direction)
    di_spread_narrow = abs(plus_di - minus_di) < 8

    # Volume spike: recent 5 candles avg volume > 1.5x previous 10 avg
    vol_spike = False
    if len(volumes) >= 15:
        recent_vol = sum(volumes[-5:]) / 5
        prev_vol = sum(volumes[-15:-5]) / 10
        if prev_vol > 0 and recent_vol > prev_vol * 1.5:
            vol_spike = True

    return di_spread_narrow or vol_spike


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
    # Require at least 21 candles for meaningful 20-candle range comparison
    if len(highs) < 21:
        return False  # insufficient data for breakout detection
    recent_high = max(highs[-21:-1])
    recent_low = min(lows[-21:-1])
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
    plus_di_series: list[float] | None = None,
    minus_di_series: list[float] | None = None,
    plus_di: float = 0,
    minus_di: float = 0,
) -> bool:
    """Detect trend exhaustion — trend losing steam after running strong.

    Path 1: DI convergence early detection (fast, no ADX decline required)
    ────────────────────────────────────────────────────────────────────
    A. Quick convergence: ADX >= 40 + spread <= 12 + +DI rising (5c >= +2)
       → Catches the moment DI lines snap together
    B. Deep convergence: ADX >= 35 + spread shrank >= 50% from peak
       + weaker DI rose >= 30% from trough + current spread <= 10
       → Catches gradual exhaustion with full confirmation

    Path 2: Classic ADX-based (slower, requires ADX to turn down)
    ────────────────────────────────────────────────────────────────────
    ADX >= 30 + momentum weakening + adx_fast < adx_slow

    Volume confirmation:
    - Low volume (vol_percentile < 15): exhaustion more reliable
    - Normal/high volume (vol_percentile > 50): need stronger evidence
    """
    # ─── Path 1A: Quick DI convergence (fastest trigger) ───
    if (adx >= 40 and plus_di_series and minus_di_series
            and len(plus_di_series) >= 5):
        spread_now = abs(plus_di - minus_di)
        spread_5ago = abs(plus_di_series[-5] - minus_di_series[-5])
        plus_di_5chg = plus_di - plus_di_series[-5]

        if spread_now <= 12 and spread_5ago - spread_now >= 4 and plus_di_5chg >= 2:
            return True

    # ─── Path 1B: Deep DI convergence (gradual exhaustion) ───
    if adx >= 35 and plus_di_series and minus_di_series:
        recent_di_spreads = []
        for p, m in zip(plus_di_series, minus_di_series):
            recent_di_spreads.append(abs(p - m))

        if len(recent_di_spreads) >= 5:
            peak_spread = max(recent_di_spreads[-10:])
            current_spread = recent_di_spreads[-1]
            trough_spread = min(recent_di_spreads[-10:])

            # Weaker DI must have risen from its low (converging from behind)
            weaker_di_current = min(plus_di, minus_di)
            weaker_di_trough = min(min(plus_di_series[-10:]), min(minus_di_series[-10:]))

            weaker_di_rise_pct = 0
            if weaker_di_trough > 0:
                weaker_di_rise_pct = (weaker_di_current - weaker_di_trough) / weaker_di_trough * 100

            spread_shrink_pct = 0
            if peak_spread > 0:
                spread_shrink_pct = (peak_spread - current_spread) / peak_spread * 100

            # Convergence confirmed: spread shrank >= 50% AND weaker DI rose >= 30%
            if spread_shrink_pct >= 50 and weaker_di_rise_pct >= 30 and current_spread <= 10:
                return True  # DI convergence = early exhaustion signal

    if adx < 30:
        return False

    # ADX must be declining, not rising. If adx_fast > adx_slow + 2,
    # trend is still strengthening — NOT exhaustion.
    if adx_fast > adx_slow + 2:
        return False  # ADX still rising — trend strengthening

    # With ADX declining confirmed, check momentum weakening.
    # Either explicit momentum flag OR implicit from fast/slow ADX gap.
    momentum_weakening = (
        momentum in ("减弱", "衰竭")
        or adx_fast < adx_slow  # ADX genuinely falling
    )
    if not momentum_weakening:
        return False

    # DX peak drop confirmation: compare current DX to recent max.
    # Works with as few as 3 data points; uses up to 10 for stability.
    if dx_series and len(dx_series) >= 3:
        lookback = min(10, len(dx_series))
        peak_dx = max(dx_series[-lookback:])
        current_dx = dx_series[-1]
        # Scale threshold by lookback: fewer points = require bigger drop
        dx_drop_threshold = 5 if lookback >= 10 else 3
        if peak_dx - current_dx >= dx_drop_threshold:
            return True  # ADX dropped significantly from peak

    # Without sufficient peak drop, require momentum == "衰竭" or low volume
    if momentum == "衰竭" and vol_percentile < 50:
        return True

    # High ADX + ADX declining but momentum not yet "衰竭":
    # if ADX is very high (>= 45), even mild weakening is meaningful
    if adx >= 45 and adx_fast < adx_slow - 1:
        return True

    return False


def _check_volume_spike_override(candles: list[dict], adx: float,
                                 plus_di: float, minus_di: float,
                                 vol_percentile: float) -> bool:
    """Detect when a volume spike + strong directional move invalidates an
    exhaustion signal. Returns True if exhaustion should be suppressed.

    In a strong trend, a volume breakout means the market chose direction —
    giving a counter-trend exhaustion signal at that point is guessing tops.
    """
    if len(candles) < 3:
        return False

    recent = candles[-2:]
    volumes = [c["volume"] for c in candles[:-2]]
    if not volumes:
        return False

    avg_vol = sum(volumes) / len(volumes)
    if avg_vol == 0:
        return False

    for c in recent:
        vol_ratio = c["volume"] / avg_vol
        body = c["close"] - c["open"]
        candle_range = c["high"] - c["low"]
        if candle_range == 0:
            continue
        body_ratio = abs(body) / candle_range

        if vol_ratio >= 2.0 and body_ratio >= 0.6:
            return True

    return False


def _analyze_single_timeframe(candles: list[dict], timeframe: str, thresholds: dict | None = None, previous_regime: str | None = None, previous_direction: str | None = None) -> dict:
    """Analyze a single timeframe and return analysis dict."""
    thresholds = thresholds or {}
    base_trending = thresholds.get("adx_trending_threshold", 25)
    base_forming = thresholds.get("adx_forming_threshold", 20)
    adx_exit = thresholds.get("adx_exit_threshold", 20)

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # ADX periods: per-timeframe configurable, evolved over time
    adx_primary = thresholds.get("adx_period", 14)
    adx_fast_p = thresholds.get("adx_fast_period", 10)
    adx_slow_p = thresholds.get("adx_slow_period", 21)

    # Primary ADX
    adx_data = calc_adx(highs, lows, closes, period=adx_primary)
    # Dual-track ADX for decay detection
    adx_fast = calc_adx(highs, lows, closes, period=adx_fast_p)["adx"]
    adx_slow = calc_adx(highs, lows, closes, period=adx_slow_p)["adx"]

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
    ema50 = calc_ema(closes, 50)

    adx = adx_data["adx"]
    plus_di = adx_data["plus_di"]
    minus_di = adx_data["minus_di"]

    # ─── Market Regime Detection (6 states) ───
    # trending, ranging, high_volatility, low_volatility, breakout, exhaustion
    current_price = closes[-1]
    atr_pct = (atr_val / max(current_price, 1)) * 100  # ATR as % of price
    dx_series = adx_data.get("dx_series", [])

    # Compute momentum before regime detection
    # Momentum: compare two non-overlapping DX windows.
    # Apply 2-period EMA smoothing to DX before comparison to avoid
    # false acceleration signals during rapid reversals where raw DX
    # may spike then crash, misleading the momentum calculation.
    dx_len = len(dx_series)
    raw_momentum = "稳定"
    if dx_len >= 6:
        # 2-period EMA smoothing
        ema_dx = [dx_series[0]]
        for i in range(1, len(dx_series)):
            ema_dx.append(dx_series[i] * 0.5 + ema_dx[-1] * 0.5)
        recent_avg = sum(ema_dx[-3:]) / 3
        earlier_avg = sum(ema_dx[-6:-3]) / 3
        diff = recent_avg - earlier_avg
        if diff > 3:
            raw_momentum = "加速"
        elif diff > 0.5:
            raw_momentum = "稳定"
        elif diff > -2:
            raw_momentum = "减弱"
        else:
            raw_momentum = "衰竭"

    # Display-adjusted momentum: if ADX_fast > adx_slow, trend is genuinely
    # strengthening. DX momentum may be slowing but the trend itself is still
    # building up. Don't show "衰竭" when ADX is rising — misleading for users.
    # But pass raw_momentum to _detect_exhaustion for accurate detection.
    momentum = raw_momentum
    if adx_fast > adx_slow + 3 and momentum in ("衰竭", "减弱"):
        momentum = "减弱" if adx_fast - adx_slow < 8 else "稳定"

    is_high_vol = _detect_high_volatility(candles, atr_pct, adx, plus_di, minus_di, volumes, vol_percentile)
    is_low_vol = _detect_low_volatility(atr_pct, adx, plus_di, minus_di, volumes, adx_forming)
    is_breakout = _detect_breakout(candles, atr_pct, adx, plus_di, minus_di, volumes, atr_val)
    is_exhaustion = _detect_exhaustion(
        adx, raw_momentum, dx_series, adx_fast, adx_slow, vol_percentile,
        plus_di_series=adx_data.get("plus_di_series"),
        minus_di_series=adx_data.get("minus_di_series"),
        plus_di=plus_di,
        minus_di=minus_di,
    )

    # Volume spike override: if recent candles show extreme volume + strong
    # directional body, suppress exhaustion — market is choosing direction.
    vol_spike_override = _check_volume_spike_override(candles, adx, plus_di, minus_di, vol_percentile)
    if is_exhaustion and vol_spike_override:
        is_exhaustion = False  # suppress exhaustion, fall through to trending/breakout

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
    elif adx >= adx_trending and adx <= adx_trending * 1.5 and vol_percentile < 15:
        regime = "low_vol_trend"  # slow trend, low vol, about to expand
    elif is_breakout:
        regime = "breakout"
    elif adx >= adx_trending:
        regime = "trending"  # strong established trend (ADX > 1.5×trending = mature, even at low vol)
    elif is_high_vol:
        regime = "high_volatility"
    elif adx >= adx_forming:
        regime = "forming"
    elif is_low_vol:
        regime = "low_volatility"
    else:
        # Fallback: check for strong directional move despite low ADX.
        # During sharp one-sided drops/rallys, ADX lags because DX averages
        # over 14 periods. Wide DI spread + meaningful price move = directional
        # trend even if ADX hasn't caught up yet.
        #
        # Dual-condition: either absolute DI spread OR relative DI ratio must
        # be sufficient, ensuring both early trend startup (low ADX, widening
        # DI ratio) and established directional moves are caught.
        di_spread_for_check = abs(plus_di - minus_di)
        di_ratio = max(plus_di, minus_di) / max(min(plus_di, minus_di), 1)

        # ATR% threshold by timeframe: shorter TFs naturally have smaller ATR%
        _min_atr_pct = {"30m": 0.25, "1h": 0.35, "4h": 0.5}.get(timeframe, 0.5)

        if (di_spread_for_check >= 12 or di_ratio >= 1.6) and atr_pct > _min_atr_pct:
            regime = "forming"  # directional move confirmed by DI + price
        else:
            regime = "ranging"

    # ─── Regime hysteresis: prevent flickering between regimes across cycles ───
    # Require stronger evidence when switching away from an established regime.
    # This avoids whipsaw between similar states (e.g., trending ↔ forming)
    # when ADX hovers near the threshold boundary.
    if previous_regime and regime != previous_regime:
        _regime_strength = {
            "trending": 4, "breakout": 3, "exhaustion": 4,
            "low_vol_trend": 3, "high_volatility": 3,
            "forming": 2, "low_volatility": 1, "ranging": 1,
        }
        new_strength = _regime_strength.get(regime, 0)
        old_strength = _regime_strength.get(previous_regime, 0)
        # If the new regime is weaker or only marginally different, stay with the old one.
        # EXCEPTION 1: when momentum == "衰竭" + ADX dropping, force switch to exhaustion
        # even if hysteresis would block it — this fixes the late exhaustion detection
        # that causes trend-following signals at the end of trends.
        is_forced_exhaustion = (
            regime == "exhaustion"
            and previous_regime in ("trending", "low_vol_trend")
            and momentum == "衰竭"
            and adx_fast < adx_slow
        )
        # EXCEPTION 2: previous regime was high_volatility but vol_percentile < 20.
        # This means the prior high_vol was a false positive (volume spike + low ATR
        # triggered it), and the new detection correctly says vol is not high.
        # Don't hysteresis-lock to a disproven high_volatility.
        is_high_vol_disproven = (
            previous_regime == "high_volatility"
            and vol_percentile is not None
            and vol_percentile < 20
        )
        if not is_forced_exhaustion and not is_high_vol_disproven and new_strength <= old_strength and adx < adx_trending + 5:
            # Exhaustion hysteresis escape: if ADX drops well below exhaustion
            # threshold (25 << 30), the trend is too weak to still be in
            # exhaustion. Keep the new computed regime to prevent
            # direction whipsaw at low ADX levels.
            if previous_regime == "exhaustion" and adx < adx_trending and adx_fast < adx_slow:
                pass  # stay with the newly computed regime — ADX no longer supports exhaustion
            else:
                regime = previous_regime

    # DI spread filter: reject directional signals if DI spread too small
    di_spread = abs(plus_di - minus_di)
    min_di_spread = thresholds.get("min_di_spread", 3)

    # ─── DI Spread Trend: track last 6 candles for convergence detection ───
    # Compute DI spread for each of the last 6 candles to detect
    # convergence (spread shrinking = trend weakening)
    di_spread_history = []
    for i in range(max(0, len(candles) - 6), len(candles)):
        window_highs = highs[:i + 1]
        window_lows = lows[:i + 1]
        window_closes = closes[:i + 1]
        if len(window_closes) >= 2 * adx_primary + 1:
            adx_tmp = calc_adx(window_highs, window_lows, window_closes, period=adx_primary)
            di_spread_history.append(round(abs(adx_tmp["plus_di"] - adx_tmp["minus_di"]), 1))
    # Pad if not enough data
    while len(di_spread_history) < 6:
        di_spread_history.insert(0, di_spread_history[0] if di_spread_history else di_spread)

    # DI spread converging: last 3 spread values are strictly decreasing
    # and the latest is < 70% of the first (catches gradual convergence).
    di_spread_converging = False
    if len(di_spread_history) >= 3:
        last3 = di_spread_history[-3:]
        di_spread_converging = (last3[0] > last3[1] > last3[2]) and last3[2] < last3[0] * 0.7

    # For short timeframes (30m, 1h), require larger DI spread.
    # Make spread threshold proportional to ADX: low ADX = lower spread bar,
    # but never below the base min_di_spread. This prevents missing early
    # forming trends where DI spread is naturally small.
    if timeframe in ("30m", "1h"):
        adx_dynamic_spread = max(min_di_spread, round(adx * 0.2, 1))
        effective_min_di = max(adx_dynamic_spread, 3)  # floor at 3
    else:
        effective_min_di = min_di_spread

    # ─── Price structure analysis (needed for direction override) ───
    # Must be computed before direction hysteresis to support low_vol_trend override
    price_structure = _analyze_price_structure(candles)

    if di_spread < effective_min_di and regime in ("trending", "forming", "low_vol_trend"):
        # Require di_spread to be CLEARLY below threshold to override regime.
        # A 0.5-point dead zone prevents flickering when spread hovers near
        # effective_min_di (e.g., 2.9 vs 3.0).
        DI_SPREAD_DEAD_ZONE = 0.5
        # EXCEPTION: low_vol_trend with ADX >= adx_trending — narrow DI spread
        # is expected in low-vol accumulation phase, don't downgrade to ranging.
        if not (regime == "low_vol_trend" and adx >= adx_trending):
            if di_spread < effective_min_di - DI_SPREAD_DEAD_ZONE:
                regime = "ranging"
                direction = None

    # Direction with hysteresis: require a minimum DI spread to flip direction.
    # Prevents long→short→long whipsaw when +DI and -DI are within 1 point.
    DIR_DEAD_ZONE = 1.0
    di_diff = plus_di - minus_di

    if regime in ("trending", "forming", "breakout", "exhaustion", "low_vol_trend"):
        if previous_direction:
            # In exhaustion regime with narrow DI spread, the DI lines are
            # converging and direction is unreliable. Lock to previous
            # direction instead of flipping — prevents bullish↔bearish
            # whipsaw when ADX is high but momentum is fading.
            # Dynamic threshold: DI_spread < ADX * 0.25, floored at 5.0.
            # This adapts to different trend strengths — a stronger trend
            # (ADX=50) needs wider spread (<12.5) to lock, while a weaker
            # one (ADX=30) locks at narrower spread (<7.5).
            # The floor of 5.0 prevents over-sensitive locking at low ADX.
            if regime == "exhaustion":
                di_spread_lock_threshold = max(5.0, adx * 0.25)
                if di_spread < di_spread_lock_threshold:
                    direction = previous_direction
                # Also require a wider dead zone to flip when near the lock zone
                elif previous_direction == "bullish" and di_diff < -2.0:
                    direction = "bearish"
                elif previous_direction == "bearish" and di_diff > 2.0:
                    direction = "bullish"
                else:
                    direction = previous_direction
                # When ADX drops below 18 in exhaustion, the trend is too weak
                # to reliably determine direction — mark as None to avoid whipsaw.
                if adx < 18:
                    direction = None
            else:
                # Require crossing the dead zone to flip
                if previous_direction == "bullish" and di_diff < -DIR_DEAD_ZONE:
                    direction = "bearish"
                elif previous_direction == "bearish" and di_diff > DIR_DEAD_ZONE:
                    direction = "bullish"
                else:
                    # DI spread in dead zone — for low_vol_trend, use price
                    # structure as tiebreaker (DI lines converge by nature in
                    # low-vol accumulation, price structure is more reliable)
                    ps_type = price_structure.get("type")
                    ema20_val = ema20[-1] if ema20 else current_price
                    if regime == "low_vol_trend":
                        if ps_type == "bullish" and current_price > ema20_val:
                            direction = "bullish"
                        elif ps_type == "bearish" and current_price < ema20_val:
                            direction = "bearish"
                        else:
                            direction = previous_direction  # stay with hysteresis
                    else:
                        direction = previous_direction  # stay in dead zone
        else:
            # No previous direction — for low_vol_trend with narrow DI spread,
            # fall back to price structure for initial direction assignment
            ps_type = price_structure.get("type")
            ema20_val = ema20[-1] if ema20 else current_price
            if regime == "low_vol_trend" and ps_type != "neutral":
                if ps_type == "bullish" and current_price > ema20_val:
                    direction = "bullish"
                elif ps_type == "bearish" and current_price < ema20_val:
                    direction = "bearish"
                else:
                    direction = None
            elif di_diff > DIR_DEAD_ZONE:
                direction = "bullish"
            elif di_diff < -DIR_DEAD_ZONE:
                direction = "bearish"
            else:
                direction = None
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
        if adx >= adx_trending * 1.4:
            strength = "强"
        elif adx >= adx_trending:
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

    # Duration: how many candles in current trend (not applicable for non-directional regimes)
    duration_hours = _estimate_duration(candles, timeframe, thresholds, direction=direction, regime=regime)
    if regime in ("ranging", "low_volatility"):
        duration_hours = None

    # Price structure analysis (already computed at line 1175 for direction logic)

    # Entry position
    entry_position = _analyze_entry_position(candles, ema20, atr_val)

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
        "plus_di_series": adx_data.get("plus_di_series", [])[-20:],
        "minus_di_series": adx_data.get("minus_di_series", [])[-20:],
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
        "di_spread_history": di_spread_history,
        "di_spread_converging": di_spread_converging,
        # Effective thresholds actually used for regime detection
        "effective_trending": round(adx_trending, 1),
        "effective_forming": round(adx_forming, 1),
        "base_trending": base_trending,
        "base_forming": base_forming,
        "trending_adj": round(trending_adj, 1),
        "forming_adj": round(forming_adj, 1),
        "ema20": round(ema20[-1], 1) if ema20 else round(current_price, 1),
        "ema50": round(ema50[-1], 1) if ema50 else round(current_price, 1),
    }


def _estimate_duration(candles: list[dict], timeframe: str, thresholds: dict = None, direction: str = None, regime: str = None) -> float:
    """Estimate how long the CURRENT REGIME has been active.

    Walks backwards through candle data, computing simplified regime labels
    at each step. Stops at the first regime transition, so the returned
    duration reflects when the current regime state began.

    This is more accurate than counting DI direction persistence — DI can
    maintain the same direction across multiple regime changes (e.g. DI
    stays bearish while regime cycles through exhaustion → low_vol_trend
    → ranging), but the dashboard should show only the current regime's
    elapsed time.
    """
    if thresholds is None:
        thresholds = {}
    tf_hours = {"30m": 0.5, "1h": 1.0, "4h": 4.0}
    hours_per_candle = tf_hours.get(timeframe, 1.0)

    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    n = len(closes)
    if n < 20:
        return round(hours_per_candle * 3, 1)

    # Compute +DM/-DM series once — O(n)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Smooth with EMA — use same adx_period as signal engine
    period = thresholds.get("adx_period", 14)
    alpha = 1.0 / period
    plus_smooth = sum(plus_dm[0:period]) / max(period, 1)
    minus_smooth = sum(minus_dm[0:period]) / max(period, 1)

    # Build DI ratio series
    di_ratios = [0.5] * n
    for i in range(1, n):
        plus_smooth = alpha * plus_dm[i] + (1 - alpha) * plus_smooth
        minus_smooth = alpha * minus_dm[i] + (1 - alpha) * minus_smooth
        total = plus_smooth + minus_smooth
        di_ratios[i] = plus_smooth / total if total > 0 else 0.5

    # Compute ATR for noise filter and low_vol detection
    atr_vals = _calc_atr_simple(candles)
    atr = atr_vals[-1] if atr_vals else 0
    atr_pct = (atr / max(closes[-1], 1)) * 100

    # Build ADX series for regime detection — pad early candles with None
    dx_series = [abs(di_ratios[i] - 0.5) * 200 for i in range(n)]
    adx_period = period
    seed_end = adx_period * 2
    if seed_end > n:
        seed_end = n
    adx_list = [None] * (seed_end - 1)  # early candles: no ADX yet
    if seed_end > 0:
        adx_val = sum(dx_series[:seed_end]) / seed_end
    else:
        adx_val = 50
    adx_list.append(adx_val)
    for i in range(seed_end, n):
        adx_val = (adx_val * (adx_period - 1) + dx_series[i]) / adx_period
        adx_list.append(adx_val)

    # Latest DI direction
    latest_ratio = di_ratios[-1]
    di_trend_dir = "up" if latest_ratio > 0.5 else "down"
    latest_spread = abs(latest_ratio - 0.5) * 2
    if latest_spread < 0.15:
        return round(hours_per_candle * 2, 1)

    # ─── Reversal duration: count from ADX peak ────────────────────────
    if direction is not None and direction != ("bullish" if di_trend_dir == "up" else "bearish"):
        if len(adx_list) > 5:
            search_start = max(0, len(adx_list) // 2)
            peak_idx = max(range(search_start, len(adx_list)), key=lambda i: adx_list[i])
            trend_count = max(1, len(adx_list) - 1 - peak_idx)
            return round(trend_count * hours_per_candle, 1)
        return round(hours_per_candle * 4, 1)

    # ─── Regime-change detection walk-back ─────────────────────────────
    # Compute a simplified regime label at each candle. Stop at the first
    # candle where the regime is DIFFERENT from the current one.
    # Grouping is too coarse — trending and low_vol_trend share the same
    # "trend" ADX level, but the transition between them matters.

    adx_trending = thresholds.get("adx_trending", 25)
    adx_forming = thresholds.get("adx_forming", 18)

    def _candle_regime(idx: int) -> str:
        """Simplified regime for a single candle based on ADX level."""
        a = adx_list[idx]
        if a is None:
            return "ranging"
        if a >= adx_trending:
            candle_range = highs[idx] - lows[idx]
            candle_atr_pct = (candle_range / max(closes[idx], 1)) * 100
            if candle_atr_pct < 0.15:
                return "low_vol_trend"
            return "trending"
        if a >= adx_forming:
            return "forming"
        return "ranging"

    current_regime = regime or _candle_regime(n - 1)

    # Non-directional regimes: return a fixed estimate
    if current_regime in ("ranging", "low_volatility"):
        return round(hours_per_candle * 2, 1)

    # Walk backwards, stop at ANY regime change or DI direction flip
    trend_count = 1
    for i in range(n - 2, max(0, n - 101), -1):
        prev_regime = _candle_regime(i)

        if prev_regime != current_regime:
            # Regime changed — stop here
            break

        # Also stop if DI direction flipped meaningfully
        ratio = di_ratios[i]
        prev_dir = "up" if ratio > 0.5 else "down"
        if prev_dir != di_trend_dir and abs(ratio - 0.5) > 0.05:
            break

        trend_count += 1

    return round(trend_count * hours_per_candle, 1)


def _linear_regression_slope(values: list[float]) -> float:
    """Compute the slope of the best-fit line through the given values."""
    n = len(values)
    if n < 2:
        return 0.0
    sum_x = (n - 1) * n / 2
    sum_y = sum(values)
    sum_xy = sum(i * v for i, v in enumerate(values))
    sum_x2 = sum(i * i for i in range(n))

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope


def _calc_atr_simple(candles: list[dict], period: int = 14) -> list[float]:
    """Compute ATR from candle data. Needs high/low/close; falls back to close range if not available."""
    if not candles:
        return []

    has_ohlcv = "high" in candles[0] and "low" in candles[0]
    trs = []
    for i in range(len(candles)):
        if has_ohlcv:
            h = candles[i]["high"]
            l = candles[i]["low"]
            if i == 0:
                tr = h - l
            else:
                pc = candles[i - 1]["close"]
                tr = max(h - l, abs(h - pc), abs(l - pc))
        else:
            # Fallback: use close-to-close change as TR proxy
            tr = abs(candles[i]["close"] - candles[i - 1]["close"]) if i > 0 else 0
        trs.append(tr)

    alpha = 1.0 / period
    seed = sum(trs[:period]) / max(period, 1)
    atr = seed
    atrs = [seed] * period
    for i in range(period, len(trs)):
        atr = alpha * trs[i] + (1 - alpha) * atr
        atrs.append(atr)
    return atrs


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


def _analyze_entry_position(candles: list[dict], ema20: list[float], atr: float = 0) -> dict:
    """Analyze where current price sits in range.

    If the 20-candle range is too narrow (< ATR×0.5), percentile is unreliable
    — price can be at '99%' and '1%' with only a few points difference.
    In that case, force limit-order at mid-range regardless of percentile.
    """
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

    # Range too narrow: percentile meaningless → force mid-range limit
    min_range_threshold = atr * 0.5 if atr > 0 else current * 0.005
    range_too_narrow = range_size < min_range_threshold

    return {
        "range_high": round(range_high, 1),
        "range_low": round(range_low, 1),
        "range_size": round(range_size, 1),
        "percentile": percentile,
        "short_move_pct": round(short_move, 3),
        "dist_from_ema": round(dist_ema, 3),
        "range_too_narrow": range_too_narrow,
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
    is_bullish = move_so_far >= 0

    # ADX strength scaling factor (uses current ADX, not historical peak)
    adx_strength = adx
    if adx_strength >= 30:
        factor = 1.5
    elif adx_strength >= 25:
        factor = 1.2
    else:
        factor = 0.8

    # Direction-aware target: extend in the direction of the trend
    move_remain = move_so_far * 0.5  # project half more of the existing move
    if is_bullish:
        target_conserv = round(current_price + move_remain + atr * factor, 1)
        target_aggress = round(current_price + move_so_far + atr * factor * 1.5, 1)
    else:
        target_conserv = round(current_price + move_remain - atr * factor, 1)
        target_aggress = round(current_price + move_so_far - atr * factor * 1.5, 1)

    # Dynamic stop distance: ATR × k_vol × k_adx
    k_vol = 1.0 + (vol_percentile / 100)  # 1.0 ~ 2.0
    k_adx = max(0.8, min(1.5, 1.0 - (adx - 25) / 50))  # ADX high → tighter stop
    stop_distance = atr * 1.5 * k_vol * k_adx
    # Minimum stop distance: 0.15% of price prevents zero/tiny stops
    stop_distance = max(stop_distance, current_price * 0.0015)

    return {
        "momentum": "稳定",
        "adx_current": adx_strength,
        "adx_drop": 0.0,
        "adx_recent_peak": adx_strength,
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
    duration_hours = _estimate_squeeze_duration(candles, atr, timeframe)
    return {
        "active": active,
        "compression": max(compression, 0),
        "src": timeframe,
        "duration_hours": duration_hours,
    }


def _estimate_squeeze_duration(candles: list[dict], atr: float, timeframe: str) -> int:
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
    # Map timeframe to candle duration in hours
    candle_hours = {"30m": 0.5, "1h": 1, "4h": 4}
    hours_per_candle = candle_hours.get(timeframe, 4)
    return max(0, count * hours_per_candle)


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


# ─── Aggressive Signal Strategies ─────────────────────────────────────
# Five additional signal paths that fire only when the main verdict is "观望".
# Each has smaller position size, tighter risk, and independent cooldown.
# Unified tag prefix: aggressive_*

AGGRESSIVE_COOLDOWNS = {
    "aggressive_1h_trend": 7200,       # 2h
    "aggressive_failed_pullback": 7200, # 2h
    "aggressive_squeeze_breakout": 14400,# 4h
    "aggressive_volume_spike": 3600,    # 1h
    "aggressive_pattern_breakout": 7200,# 2h
    "aggressive_volatility_expansion": 7200, # 2h
    "aggressive_momentum_accel": 3600,  # 1h — short-lived signal
    "aggressive_adx7_early": 7200,     # 2h — noisy early signal
}


def _detect_bb_squeeze_detail(candles: list[dict], atr: float) -> dict | None:
    """Detailed BB squeeze state: band width, historical comparison, range levels."""
    closes = [c["close"] for c in candles[-20:]]
    if len(closes) < 20:
        return None
    sma20 = sum(closes) / len(closes)
    std = (sum((x - sma20) ** 2 for x in closes) / len(closes)) ** 0.5
    upper = sma20 + 2 * std
    lower = sma20 - 2 * std
    band_width = upper - lower

    # Use same threshold as _check_squeeze (atr * 1.5) for consistency
    if band_width >= atr * 1.5:
        return None

    all_closes = [c["close"] for c in candles]
    bb_widths = []
    for i in range(20, len(all_closes) + 1):
        window = all_closes[i - 20:i]
        w_sma = sum(window) / 20
        w_std = (sum((x - w_sma) ** 2 for x in window) / 20) ** 0.5
        bb_widths.append(w_std * 4)

    if len(bb_widths) < 5:
        return None
    avg_width = sum(bb_widths[-20:]) / min(len(bb_widths), 20)
    if avg_width == 0 or band_width >= avg_width * 0.5:
        return None

    recent_high = max(c["high"] for c in candles[-5:])
    recent_low = min(c["low"] for c in candles[-5:])
    squeeze_range = recent_high - recent_low

    return {
        "upper": round(upper, 1),
        "lower": round(lower, 1),
        "band_width": round(band_width, 1),
        "avg_width": round(avg_width, 1),
        "compression_pct": round((1 - band_width / avg_width) * 100, 1),
        "recent_high": round(recent_high, 1),
        "recent_low": round(recent_low, 1),
        "squeeze_range": round(squeeze_range, 1),
    }


def _detect_double_pattern(candles_1h: list[dict], candles_30m: list[dict], atr_1h: float = 0, atr_30m: float = 0) -> dict | None:
    """Detect double top (M) or double bottom (W) on 1h/30m.

    Improvements:
    - Extended target: 1.618 Fibonacci extension for strong momentum
    - Dynamic stop: neckline ± ATR buffer instead of absolute extreme
    - Anti-chase: skip if entry is too far above/below neckline
    """
    for tf_candles, tf_name, atr_tf in [(candles_1h, "1h", atr_1h), (candles_30m, "30m", atr_30m)]:
        if len(tf_candles) < 20:
            continue
        closes = [c["close"] for c in tf_candles]
        highs = [c["high"] for c in tf_candles]
        lows = [c["low"] for c in tf_candles]

        swing_highs = []
        swing_lows = []
        for i in range(2, len(tf_candles) - 2):
            if all(highs[i] >= highs[j] for j in range(max(0, i-2), min(len(highs), i+3)) if j != i):
                swing_highs.append(i)
            if all(lows[i] <= lows[j] for j in range(max(0, i-2), min(len(lows), i+3)) if j != i):
                swing_lows.append(i)

        for a in range(len(swing_highs)):
            for b in range(a + 1, len(swing_highs)):
                sh1, sh2 = swing_highs[a], swing_highs[b]
                dist = sh2 - sh1
                if dist < 5 or dist > 15:
                    continue
                price_match = abs(highs[sh1] - highs[sh2]) / max(highs[sh1], 1) < 0.01
                if not price_match:
                    continue
                nl_idx = min(range(sh1, sh2 + 1), key=lambda i: lows[i])
                neckline = lows[nl_idx]
                amplitude = (highs[sh1] - neckline) / max(neckline, 1)
                if amplitude < 0.005:
                    continue
                current_price = closes[-1]
                if current_price < neckline:
                    # === Double Top ===
                    # Price broke below neckline — bearish breakdown confirmed
                    pattern_height = highs[sh1] - neckline

                    # Anti-chase: if price dropped too far below neckline, skip
                    chase_threshold = pattern_height * 0.5
                    if neckline - current_price > chase_threshold:
                        continue

                    # Extended target: 1.618 Fibonacci extension from neckline
                    extended_target = neckline - pattern_height * 1.618

                    # Dynamic stop: neckline + ATR buffer (not absolute top)
                    if atr_tf > 0:
                        dynamic_stop = neckline + atr_tf * 0.5
                        stop = min(highs[sh1] * 1.002, dynamic_stop)
                    else:
                        stop = round(highs[sh1] * 1.002, 1)

                    return {
                        "type": "double_top",
                        "timeframe": tf_name,
                        "direction": "bearish",
                        "direction_label": "看跌",
                        "entry": round(current_price, 1),
                        "stop": round(stop, 1),
                        "target": round(extended_target, 1),
                        "confidence": 40,
                        "position_pct": 12,
                    }

        for a in range(len(swing_lows)):
            for b in range(a + 1, len(swing_lows)):
                sl1, sl2 = swing_lows[a], swing_lows[b]
                dist = sl2 - sl1
                if dist < 5 or dist > 15:
                    continue
                price_match = abs(lows[sl1] - lows[sl2]) / max(lows[sl1], 1) < 0.01
                if not price_match:
                    continue
                nl_idx = max(range(sl1, sl2 + 1), key=lambda i: highs[i])
                neckline = highs[nl_idx]
                amplitude = (neckline - lows[sl1]) / max(lows[sl1], 1)
                if amplitude < 0.005:
                    continue
                current_price = closes[-1]
                if current_price > neckline:
                    # === Double Bottom ===
                    pattern_height = neckline - lows[sl1]

                    # Anti-chase: if entry is already too far above neckline (> 50% of pattern height),
                    # the reward zone is too small — skip this signal
                    chase_threshold = pattern_height * 0.5
                    if current_price - neckline > chase_threshold:
                        continue

                    # Extended target: 1.618 Fibonacci extension from neckline
                    extended_target = neckline + pattern_height * 1.618

                    # Dynamic stop: neckline - ATR buffer (not absolute bottom)
                    # Use the tighter of: absolute bottom - 0.2%, or neckline - 0.5*ATR
                    if atr_tf > 0:
                        dynamic_stop = neckline - atr_tf * 0.5
                        stop = max(lows[sl1] * 0.998, dynamic_stop)
                    else:
                        stop = round(lows[sl1] * 0.998, 1)

                    return {
                        "type": "double_bottom",
                        "timeframe": tf_name,
                        "direction": "bullish",
                        "direction_label": "看涨",
                        "entry": round(current_price, 1),
                        "stop": round(stop, 1),
                        "target": round(extended_target, 1),
                        "confidence": 40,
                        "position_pct": 12,
                    }
    return None


def _check_aggressive_1h_trend(h4, h1, h30, all_thresholds) -> dict | None:
    """Strategy 1: Single timeframe trend — 1h trending without 4h confirmation."""
    if h4["regime"] in ("exhaustion", "high_volatility"):
        return None
    if h1["regime"] not in ("trending", "breakout", "forming"):
        return None
    if h1.get("adx", 0) < 20:  # lowered from 25 to catch forming trends
        return None
    h1_di_spread = h1.get("di_spread", 0)
    if h1_di_spread < 8:
        return None
    if not h1.get("direction"):
        return None
    if h30.get("direction") != h1["direction"]:
        return None

    dir_label = "看涨" if h1["direction"] == "bullish" else "看跌"
    return {
        "signal_type": "aggressive_1h_trend",
        "direction": h1["direction"],
        "direction_label": dir_label,
        "action": f"1h独立趋势（轻仓试{dir_label}）",
        "side": "多" if h1["direction"] == "bullish" else "空",
        "confidence": 40,
        "position_pct": 10,
        "entry_type": "market",
        "entry_note": f"1h独立趋势{'看涨' if h1['direction'] == 'bullish' else '看跌'}（ADX={h1['adx']:.0f}, DI价差={h1_di_spread:.1f}），4h未反对",
        "reason": (
            f"4h{h4['regime']}（ADX={h4['adx']:.0f}），但1h趋势明确"
            f"（ADX={h1['adx']:.0f}, DI价差={h1_di_spread:.1f}），"
            f"30m同向确认，可轻仓试{dir_label}"
        ),
    }


def _check_failed_pullback(h4, h1, h30, all_candles_4h) -> dict | None:
    """Strategy 2: Failed pullback continuation."""
    # Allow in directional regimes where 1h pullbacks are meaningful
    if h4["regime"] not in ("trending", "low_vol_trend", "breakout", "forming", "low_volatility"):
        return None
    orig_dir = h4["direction"]
    if not orig_dir:
        return None

    h1_pdi = h1.get("plus_di_series", [])
    h1_mdi = h1.get("minus_di_series", [])
    if len(h1_pdi) < 6 or len(h1_mdi) < 6:
        return None

    if orig_dir == "bullish":
        recent_opposition = any(
            h1_mdi[-i] > h1_pdi[-i] for i in range(1, min(5, len(h1_pdi)))
        )
        current_recovery = h1_pdi[-1] > h1_mdi[-1]
        if not (recent_opposition and current_recovery):
            return None
        dir_label = "看涨"
        side = "多"
    else:
        recent_opposition = any(
            h1_pdi[-i] > h1_mdi[-i] for i in range(1, min(5, len(h1_pdi)))
        )
        current_recovery = h1_mdi[-1] > h1_pdi[-1]
        if not (recent_opposition and current_recovery):
            return None
        dir_label = "看跌"
        side = "空"

    if h30.get("direction") != orig_dir:
        return None

    current_price = h4["price"]
    if orig_dir == "bullish":
        stop = round(current_price - h1["atr"] * 1.5, 1)
        target = round(current_price + h1["atr"] * 3, 1)
    else:
        stop = round(current_price + h1["atr"] * 1.5, 1)
        target = round(current_price - h1["atr"] * 3, 1)

    return {
        "signal_type": "aggressive_failed_pullback",
        "direction": orig_dir,
        "direction_label": dir_label,
        "action": f"浅回调延续（追{dir_label}）",
        "side": side,
        "confidence": 38,
        "position_pct": 15,
        "entry_type": "market",
        "entry_note": f"强势趋势中浅回调结束，{'多' if side == '多' else '空'}头恢复",
        "stop": stop,
        "target": target,
        "reason": (
            f"4h{dir_label}趋势中（ADX={h4['adx']:.0f}），1h短暂反向后DI恢复正向，"
            f"30m同向确认，回调失败，追入原趋势方向"
        ),
    }


def _check_squeeze_breakout(h4, h1, h30, candles_1h) -> dict | None:
    """Strategy 3: BB Squeeze breakout — bidirectional breakout from extreme compression."""
    squeeze = _detect_bb_squeeze_detail(candles_1h, h1["atr"])
    if not squeeze:
        return None

    current_price = h4["price"]
    high_break = current_price > squeeze["recent_high"]
    low_break = current_price < squeeze["recent_low"]
    if not (high_break or low_break):
        return None

    if high_break:
        direction = "bullish"
        direction_label = "看涨"
        side = "多"
        stop = squeeze["recent_low"]
        target = round(current_price + squeeze["squeeze_range"] * 2, 1)
    else:
        direction = "bearish"
        direction_label = "看跌"
        side = "空"
        stop = squeeze["recent_high"]
        target = round(current_price - squeeze["squeeze_range"] * 2, 1)

    return {
        "signal_type": "aggressive_squeeze_breakout",
        "direction": direction,
        "direction_label": direction_label,
        "action": f"挤压突破（{direction_label}）",
        "side": side,
        "confidence": 35,
        "position_pct": 5,
        "entry_type": "market",
        "entry_note": f"BB挤压突破{'上轨' if high_break else '下轨'}，区间宽度{squeeze['squeeze_range']:.0f}，压缩{squeeze['compression_pct']:.0f}%",
        "stop": round(stop, 1),
        "target": target,
        "reason": (
            f"1h布林带极度压缩（带宽{squeeze['band_width']:.0f} < ATR×2={h1['atr']*2:.0f}，"
            f"压缩{squeeze['compression_pct']:.0f}%），"
            f"价格突破{'近期高点' if high_break else '近期低点'}，"
            f"挤压释放{'做多' if high_break else '做空'}"
        ),
    }


def _check_volume_spike(h4, h1, h30, candles_30m=None, candles_1h=None) -> dict | None:
    """Strategy 4: Volume spike breakout — independent volume-driven signal."""
    for tf_name, min_move_pct, tf_candles in [
        ("30m", 0.003, candles_30m or fetch_klines(settings.binance_symbol, "30m", limit=25)),
        ("1h", 0.005, candles_1h or fetch_klines(settings.binance_symbol, "1h", limit=25)),
    ]:
        if tf_candles is None or len(tf_candles) < 25:
            continue
        vols = [c["volume"] for c in tf_candles]
        avg_vol = sum(vols[:-1]) / max(len(vols) - 1, 1)
        latest_vol = vols[-1]
        if avg_vol == 0 or latest_vol < avg_vol * 2.5:
            continue

        last = tf_candles[-1]
        candle_range = last["high"] - last["low"]
        body = abs(last["close"] - last["open"])
        if candle_range == 0:
            continue
        body_ratio = body / candle_range
        move_pct = candle_range / max(last["open"], 1)
        if move_pct < min_move_pct or body_ratio < 0.7:
            continue

        bullish = last["close"] > last["open"]
        direction = "bullish" if bullish else "bearish"
        dir_label = "看涨" if bullish else "看跌"
        side = "多" if bullish else "空"

        if direction == "bullish" and h4["regime"] == "trending" and h4["direction"] == "bearish":
            continue
        if direction == "bearish" and h4["regime"] == "trending" and h4["direction"] == "bullish":
            continue

        stop_mult = h1["atr"] * 1.5
        if bullish:
            stop = round(last["low"] - stop_mult * 0.3, 1)
            target = round(last["close"] + stop_mult * 2, 1)
        else:
            stop = round(last["high"] + stop_mult * 0.3, 1)
            target = round(last["close"] - stop_mult * 2, 1)

        return {
            "signal_type": "aggressive_volume_spike",
            "direction": direction,
            "direction_label": dir_label,
            "action": f"量能异动（{dir_label}）",
            "side": side,
            "confidence": 38,
            "position_pct": 10,
            "entry_type": "market",
            "entry_note": f"{tf_name}量能突增（{latest_vol/avg_vol:.1f}x均值），K线幅度{move_pct*100:.2f}%",
            "stop": stop,
            "target": target,
            "reason": (
                f"{tf_name}巨量异动（成交量{latest_vol/avg_vol:.1f}x过去20根均值），"
                f"价格单向移动{move_pct*100:.2f}%，收盘接近{'最高' if bullish else '最低'}，"
                f"4h未明确反对"
            ),
        }
    return None


def _check_pattern_breakout(h4, h1, h30, candles_1h=None, candles_30m=None, macro=None) -> dict | None:
    """Strategy 5: Double top/bottom breakout — pure price pattern signal."""
    if candles_1h is None:
        candles_1h = fetch_klines(settings.binance_symbol, "1h", limit=25)
    if candles_30m is None:
        candles_30m = fetch_klines(settings.binance_symbol, "30m", limit=30)
    pattern = _detect_double_pattern(candles_1h, candles_30m, atr_1h=h1.get("atr", 0), atr_30m=h30.get("atr", 0))
    if not pattern:
        return None

    if pattern["direction"] == "bullish" and h4["regime"] == "trending" and h4["direction"] == "bearish":
        return None
    if pattern["direction"] == "bearish" and h4["regime"] == "trending" and h4["direction"] == "bullish":
        return None

    # Macro filter: skip pattern signals that contradict macro direction
    if macro:
        if pattern["direction"] == "bearish" and (macro.get("is_bull") or macro.get("is_bull_pullback")):
            return None
        if pattern["direction"] == "bullish" and (macro.get("is_bear") or macro.get("is_bear_pullback")):
            return None

    return {
        "signal_type": "aggressive_pattern_breakout",
        "pattern_subtype": pattern["type"],
        "direction": pattern["direction"],
        "direction_label": pattern["direction_label"],
        "action": f"形态突破（{pattern['timeframe']} {pattern['direction_label']}）",
        "side": "多" if pattern["direction"] == "bullish" else "空",
        "confidence": pattern["confidence"],
        "position_pct": pattern["position_pct"],
        "entry_type": "market",
        "entry_note": f"{pattern['timeframe']} {pattern['type']}形态颈线突破",
        "stop": pattern["stop"],
        "target": pattern["target"],
        "reason": (
            f"{pattern['timeframe']}检测到{pattern['type']}形态，"
            f"颈线已突破，目标{pattern['target']:.0f}（1.618扩展），止损{pattern['stop']:.0f}（动态）"
        ),
    }


def _check_volatility_expansion(h4, h1, h30, candles_30m=None, candles_1h=None) -> dict | None:
    """Strategy 6: Volatility expansion — sudden ATR spike + large one-way move.

    Conditions: 30m ATR > 20-bar avg × 2 AND price moved > 500 points one-way;
    closes near extreme; 4h not in strong opposite trend.
    """
    for tf_name, tf_candles in [
        ("30m", candles_30m or fetch_klines(settings.binance_symbol, "30m", limit=25)),
        ("1h", candles_1h or fetch_klines(settings.binance_symbol, "1h", limit=25)),
    ]:
        if tf_candles is None or len(tf_candles) < 25:
            continue

        highs = [c["high"] for c in tf_candles]
        lows = [c["low"] for c in tf_candles]
        closes = [c["close"] for c in tf_candles]

        # Compute ATR series
        from app.indicators import calc_atr_series
        atr_series = calc_atr_series(highs, lows, closes, period=14)
        if len(atr_series) < 20:
            continue

        atr_avg = sum(atr_series[-20:-1]) / 19
        current_atr = atr_series[-1]
        if atr_avg == 0 or current_atr < atr_avg * 2:
            continue

        # Price movement: last candle one-way move > 0.5% of price (adaptive to BTC level)
        last = tf_candles[-1]
        move_up = last["high"] - last["open"]
        move_down = last["open"] - last["low"]
        close_high = last["close"] > last["open"]

        # Use the larger directional move (not total range)
        directional_move = move_up if close_high else move_down
        directional_move_pct = directional_move / max(last["open"], 1) * 100
        if directional_move_pct < 0.5:
            continue

        # Close must be near extreme (>70% of candle in that direction)
        candle_range = last["high"] - last["low"]
        if candle_range == 0:
            continue
        body_ratio = abs(last["close"] - last["open"]) / candle_range
        if body_ratio < 0.7:
            continue

        bullish = close_high
        direction = "bullish" if bullish else "bearish"
        dir_label = "看涨" if bullish else "看跌"
        side = "多" if bullish else "空"

        # Check 4h not in strong opposite
        if direction == "bullish" and h4["regime"] == "trending" and h4["direction"] == "bearish":
            continue
        if direction == "bearish" and h4["regime"] == "trending" and h4["direction"] == "bullish":
            continue

        stop_mult = current_atr * 1.5
        if bullish:
            stop = round(last["low"] - stop_mult * 0.3, 1)
            target = round(last["close"] + current_atr * 2, 1)
        else:
            stop = round(last["high"] + stop_mult * 0.3, 1)
            target = round(last["close"] - current_atr * 2, 1)

        return {
            "signal_type": "aggressive_volatility_expansion",
            "direction": direction,
            "direction_label": dir_label,
            "action": f"波动爆发（{dir_label}）",
            "side": side,
            "confidence": 42,
            "position_pct": 8,
            "entry_type": "market",
            "entry_note": f"{tf_name} ATR突增（{current_atr/atr_avg:.1f}x均值），单向幅度{directional_move_pct:.2f}%",
            "stop": stop,
            "target": target,
            "reason": (
                f"{tf_name}波动率爆发（当前ATR={current_atr:.0f} > 20根均值={atr_avg:.0f} 的2倍），"
                f"价格单向移动{directional_move_pct:.2f}%，"
                f"收盘接近{'最高' if bullish else '最低'}，"
                f"4h未明确反对"
            ),
        }
    return None


def _check_momentum_accel(h4, h1, h30, candles_30m=None, candles_1h=None) -> dict | None:
    """Strategy 7: Momentum acceleration — catch drops/rally in progress.

    Detects when 3+ consecutive candles move in same direction with
    cumulative drop/rally > 500 points AND the last candle is expanding.
    Fires BEFORE ADX confirms, capturing the middle section of fast moves.
    """
    for tf_name, tf_candles in [
        ("30m", candles_30m or fetch_klines(settings.binance_symbol, "30m", limit=15)),
        ("1h", candles_1h or fetch_klines(settings.binance_symbol, "1h", limit=15)),
    ]:
        if tf_candles is None or len(tf_candles) < 6:
            continue

        # Check last 3-5 candles for consecutive directional movement
        recent = tf_candles[-5:]
        bearish_count = 0
        bullish_count = 0
        cumulative_move = 0.0
        last_vol = recent[-1]["volume"]

        for c in recent:
            if c["close"] < c["open"]:
                bearish_count += 1
                cumulative_move += c["open"] - c["close"]
            else:
                bullish_count += 1
                cumulative_move += c["close"] - c["open"]

        # Need at least 3 consecutive same-direction candles
        is_bearish = bearish_count >= 3 and bullish_count == 0
        is_bullish = bullish_count >= 3 and bearish_count == 0
        if not (is_bearish or is_bullish):
            # Also check last 3 only
            last3 = tf_candles[-3:]
            if len(last3) < 3:
                continue
            last3_bearish = all(c["close"] < c["open"] for c in last3)
            last3_bullish = all(c["close"] > c["open"] for c in last3)
            if last3_bearish:
                is_bearish = True
                cumulative_move = last3[0]["open"] - last3[-1]["close"]
            elif last3_bullish:
                is_bullish = True
                cumulative_move = last3[-1]["close"] - last3[0]["open"]
            else:
                continue

        # Total directional move must exceed 0.5% of price
        move_pct_check = cumulative_move / max(tf_candles[-1]["open"], 1) * 100
        if move_pct_check < 0.5:
            continue

        # Volume expansion on last candle: > 1.5x average of previous 5
        vols = [c["volume"] for c in tf_candles[-6:]]
        avg_vol = sum(vols[:-1]) / max(len(vols) - 1, 1)
        if avg_vol > 0 and last_vol < avg_vol * 1.5:
            continue

        # Last candle body ratio: not a doji
        last = tf_candles[-1]
        body = abs(last["close"] - last["open"])
        candle_range = last["high"] - last["low"]
        if candle_range == 0 or body / candle_range < 0.6:
            continue

        direction = "bearish" if is_bearish else "bullish"
        dir_label = "看跌" if is_bearish else "看涨"
        side = "空" if is_bearish else "多"

        # Check 4h not in strong opposite
        if direction == "bullish" and h4["regime"] == "trending" and h4["direction"] == "bearish":
            continue
        if direction == "bearish" and h4["regime"] == "trending" and h4["direction"] == "bullish":
            continue

        # Stop: beyond the extreme of the run; target: 1x cumulative move extension
        if is_bearish:
            run_high = max(c["high"] for c in tf_candles[-5:])
            stop = round(run_high + h1["atr"] * 0.6, 1)
            target = round(last["close"] - cumulative_move, 1)
        else:
            run_low = min(c["low"] for c in tf_candles[-5:])
            stop = round(run_low - h1["atr"] * 0.6, 1)
            target = round(last["close"] + cumulative_move, 1)

        return {
            "signal_type": "aggressive_momentum_accel",
            "direction": direction,
            "direction_label": dir_label,
            "action": f"动量加速（{dir_label}）",
            "side": side,
            "confidence": 45,
            "position_pct": 8,
            "entry_type": "market",
            "entry_note": f"{tf_name}连续{max(bearish_count, bullish_count)}根{'阴' if is_bearish else '阳'}线，累计移动{move_pct_check:.2f}%",
            "stop": stop,
            "target": target,
            "reason": (
                f"{tf_name}动量加速：连续{max(bearish_count, bullish_count)}根同向K线，"
                f"累计{'下跌' if is_bearish else '上涨'}{move_pct_check:.2f}%，"
                f"最后一根放量（{last_vol/avg_vol:.1f}x均值），"
                f"4h未明确反对"
            ),
        }
    return None


def _check_adx7_early(h4, h1, h30, candles_30m=None, candles_1h=None, all_thresholds=None) -> dict | None:
    """Strategy 8: Fast ADX early warning — catch trend startup before ADX confirms.

    Uses per-TF fast ADX period + DI crossover for earliest trend detection.
    Conditions: fast ADX > 15 + DI crossover + 30m same direction + 4h not opposed.
    """
    tf_fast_periods = {"30m": 10, "1h": 10, "4h": 7}  # defaults
    if all_thresholds and "tf_thresholds" in all_thresholds:
        for tf in tf_fast_periods:
            tf_fast_periods[tf] = all_thresholds["tf_thresholds"].get(tf, {}).get("adx_fast_period", tf_fast_periods[tf])

    for tf_name, tf_candles in [
        ("30m", candles_30m or fetch_klines(settings.binance_symbol, "30m", limit=20)),
        ("1h", candles_1h or fetch_klines(settings.binance_symbol, "1h", limit=20)),
    ]:
        if tf_candles is None or len(tf_candles) < 15:
            continue

        fast_p = tf_fast_periods[tf_name]

        highs = [c["high"] for c in tf_candles]
        lows = [c["low"] for c in tf_candles]
        closes = [c["close"] for c in tf_candles]

        adx_now = calc_adx(highs, lows, closes, period=fast_p)
        adx_prev = calc_adx(highs[:-1], lows[:-1], closes[:-1], period=fast_p)

        # DI crossover: current candle +DI crosses above -DI (bullish) or vice versa
        di_cross_bullish = (
            adx_now["plus_di"] > adx_now["minus_di"]
            and adx_prev["plus_di"] <= adx_prev["minus_di"]
        )
        di_cross_bearish = (
            adx_now["minus_di"] > adx_now["plus_di"]
            and adx_prev["minus_di"] <= adx_prev["plus_di"]
        )

        if not (di_cross_bullish or di_cross_bearish):
            continue

        if adx_now["adx"] < 15 or adx_now["adx"] <= adx_prev["adx"]:
            continue

        # Price must be moving in the direction (last candle confirms)
        last = tf_candles[-1]
        if di_cross_bullish and last["close"] <= last["open"]:
            continue
        if di_cross_bearish and last["close"] >= last["open"]:
            continue

        direction = "bullish" if di_cross_bullish else "bearish"
        dir_label = "看涨" if di_cross_bullish else "看跌"
        side = "多" if di_cross_bullish else "空"

        # Check 4h not in strong opposite
        if direction == "bullish" and h4["regime"] == "trending" and h4["direction"] == "bearish":
            continue
        if direction == "bearish" and h4["regime"] == "trending" and h4["direction"] == "bullish":
            continue

        # Stop: 0.5% or 1x ATR, whichever is larger — avoids whipsaw on normal noise
        current_price = closes[-1]
        atr1h = h1["atr"]
        stop_pct = current_price * 0.005
        tight_stop = max(stop_pct, atr1h * 1.0)
        if di_cross_bullish:
            stop = round(current_price - tight_stop, 1)
            target = round(current_price + tight_stop * 3, 1)
        else:
            stop = round(current_price + tight_stop, 1)
            target = round(current_price - tight_stop * 3, 1)

        return {
            "signal_type": "aggressive_adx7_early",
            "direction": direction,
            "direction_label": dir_label,
            "action": f"ADX(7)早期预警（{dir_label}）",
            "side": side,
            "confidence": 30,
            "position_pct": 5,
            "leverage": 10,
            "entry_type": "market",
            "entry_note": f"ADX({fast_p}) DI金叉（{tf_name}），ADX={adx_now['adx']:.1f} +DI={adx_now['plus_di']:.0f} vs -DI={adx_now['minus_di']:.0f}",
            "stop": stop,
            "target": target,
            "reason": (
                f"{tf_name} ADX({fast_p}) 早期预警：DI交叉翻转（+DI={adx_now['plus_di']:.0f} {'>' if di_cross_bullish else '<'} -DI={adx_now['minus_di']:.0f}），"
                f"ADX({fast_p})={adx_now['adx']:.1f} 正在上升，"
                f"超早期试探性入场（仓位5%，杠杆10x，止损{tight_stop:.0f}）"
            ),
        }
    return None


def _check_aggressive_signals(
    h4, h1, h30, all_candles_4h, all_thresholds, market_ctx, macro,
    pre_fetched_candles: dict | None = None,
) -> dict | None:
    """Evaluate all aggressive strategies, return the most appropriate one.

    Signal persistence model:
    - Structural signals (pattern, squeeze) are persistent — once fired,
      they remain valid for their full cooldown period.
    - Transient signals (volume spike, momentum accel) are instantaneous —
      they only appear when conditions are met on the current candle.
    - To prevent flickering: persistent signals beat transient ones when
      both are available, and transient signals are only selected when
      no persistent signal is active.
    """
    now = time.time()
    cooldowns = market_ctx.get("aggressive_cooldowns", {})
    prev_active = market_ctx.get("last_aggressive_signal")

    # Persistent signal types: structural patterns that remain valid
    # until invalidated (not just one-candle events)
    PERSISTENT_TYPES = {
        "aggressive_pattern_breakout",
        "aggressive_squeeze_breakout",
        "aggressive_failed_pullback",
    }

    # Helper: get pre-fetched candles or fetch on demand
    _candles_30m = None
    _candles_1h = None
    if pre_fetched_candles:
        _candles_30m = pre_fetched_candles.get("30m")
        _candles_1h = pre_fetched_candles.get("1h")

    fires = []
    for stype, checker in [
        ("aggressive_1h_trend", lambda: _check_aggressive_1h_trend(h4, h1, h30, all_thresholds)),
        ("aggressive_failed_pullback", lambda: _check_failed_pullback(h4, h1, h30, all_candles_4h)),
        ("aggressive_squeeze_breakout", lambda: _check_squeeze_breakout(h4, h1, h30, _candles_1h or fetch_klines(settings.binance_symbol, "1h", limit=25))),
        ("aggressive_volume_spike", lambda: _check_volume_spike(h4, h1, h30, _candles_30m, _candles_1h)),
        ("aggressive_pattern_breakout", lambda: _check_pattern_breakout(h4, h1, h30, _candles_1h or fetch_klines(settings.binance_symbol, "1h", limit=25), _candles_30m or fetch_klines(settings.binance_symbol, "30m", limit=30), macro)),
        ("aggressive_volatility_expansion", lambda: _check_volatility_expansion(h4, h1, h30, _candles_30m, _candles_1h)),
        ("aggressive_momentum_accel", lambda: _check_momentum_accel(h4, h1, h30, _candles_30m, _candles_1h)),
        ("aggressive_adx7_early", lambda: _check_adx7_early(h4, h1, h30, _candles_30m, _candles_1h, all_thresholds)),
    ]:
        if cooldowns.get(stype, False):
            continue
        result = checker()
        if result:
            is_persistent = stype in PERSISTENT_TYPES
            fires.append((stype, result, is_persistent))

    if not fires:
        return None

    # Selection logic:
    # 1. If the previous active signal is persistent and still firing, keep it
    # 2. Persistent signals > transient (structural > noise)
    # 3. Among persistent, pick by original priority order
    # 4. If no persistent, pick highest-priority transient
    if prev_active:
        prev_type = prev_active.split(",")[0] if "," in prev_active else prev_active
        for stype, result, is_p in fires:
            if stype == prev_type and is_p:
                result["_aggressive"] = True
                result["_cooldown_ts"] = now
                return result

    persistent_fires = [(s, r) for s, r, p in fires if p]
    if persistent_fires:
        stype, result = persistent_fires[0]
    else:
        stype, result, _ = fires[0]

    result["_aggressive"] = True
    result["_cooldown_ts"] = now
    return result


def generate_verdict(
    history_rows: list | None = None,
    position: dict | None = None,
    market_context: dict | None = None,
    pre_fetched_klines: dict | None = None,
    previous_regimes: dict | None = None,
) -> dict:
    """Multi-timeframe analysis verdict with circuit breaker and funding/OI filter.

    Args:
        pre_fetched_klines: optional dict mapping timeframe to list[candle dict],
            allows caller to pass pre-fetched candles (e.g. from parallel fetch).
        previous_regimes: optional dict mapping timeframe to previous cycle's regime,
            used for hysteresis to prevent regime flickering.
    """
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

    # Fetch enough candles (5-7 days of data per TF)
    tf_candle_limits = {"30m": 200, "1h": 200, "4h": 100}

    for tf in TIMEFRAMES:
        if pre_fetched_klines and tf in pre_fetched_klines:
            candles = pre_fetched_klines[tf]
        else:
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
        prev_regimes = previous_regimes or {}
        prev_direction = prev_regimes.get(f"{tf}_direction")
        results[tf] = _analyze_single_timeframe(candles, tf, tf_thresholds, previous_regime=prev_regimes.get(tf), previous_direction=prev_direction)
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
    peak_adx, peak_regime = _get_peak_adx_from_history(history_rows or [])

    # Regime consistency check: if the current regime group differs from the
    # peak's regime group, the peak belongs to a different market state and
    # should be ignored. This prevents decay detection from carrying over an
    # old trend's peak into a new regime (e.g. exhaustion peak → low_vol_trend).
    def _current_regime_group(r: str) -> str:
        if not r:
            return "other"
        if r in ("trending", "low_vol_trend"):
            return "trend"
        if r in ("forming", "breakout"):
            return "forming"
        if r == "exhaustion":
            return "exhaustion"
        return "other"

    if peak_regime and peak_regime != _current_regime_group(h4["regime"]):
        peak_adx = 0  # stale peak from different regime — ignore

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

    # ─── Macro Bull/Bear Detection ───
    # Detect macro trend using 4h EMA50 + DI structure.
    # Used to: (1) block short signals in macro bull, (2) allow light longs in ranging,
    # (3) lower ADX threshold for "soft trend" detection.
    if all_candles_4h:
        macro = _detect_macro_bull(all_candles_4h, h4)
    else:
        macro = {"is_bull": False, "is_bear": False, "ema50": 0, "price_vs_ema50_pct": 0, "di_bias": "neutral", "reason": "无K线数据"}

    # ─── Bull/Bear Context: Adjust 4h effective ADX threshold ───
    # In a macro bull, reduce the trending threshold by 2-3 points to catch "soft trends"
    # where ADX is 20-25 but price is climbing steadily. Symmetric for macro bear.
    # This is a runtime-only adjustment — evolution.json is not modified.
    bull_adj = 0.0
    _eff_trending = h4.get("effective_trending", 25)
    _eff_forming = h4.get("effective_forming", 20)
    if macro["is_bull"] and h4["adx"] >= _eff_forming and h4["adx"] < _eff_trending:
        bull_adj = -2.5
    elif macro["is_bear"] and h4["adx"] >= _eff_forming and h4["adx"] < _eff_trending:
        bull_adj = +2.5

    if bull_adj != 0.0:
        old_eff = h4.get("effective_trending", 25)
        new_eff = max(18, round(old_eff + bull_adj, 1))
        h4["effective_trending"] = new_eff
        h4["trending_adj"] = h4.get("trending_adj", 0) + bull_adj

        # Re-evaluate regime if ADX now crosses the adjusted threshold
        if h4["regime"] in ("ranging", "forming", "low_volatility") and h4["adx"] >= new_eff:
            h4["regime"] = "trending"
            h4["direction"] = "bullish" if h4["plus_di"] > h4["minus_di"] else "bearish"

    # ─── Macro Filter: suppress counter-trend exhaustion on small TFs ───
    # In a strong macro trend, 30m/1h exhaustion signals against the macro
    # direction are "guessing tops/bottoms" and consistently fail verification.
    # Suppress them: downgrade to ranging with direction=None.
    for tf_key in ["30m", "1h"]:
        tf = results[tf_key]
        if tf["regime"] != "exhaustion" or tf["direction"] is None:
            continue

        macro_bull = macro.get("is_bull") or macro.get("is_bull_pullback")
        macro_bear = macro.get("is_bear") or macro.get("is_bear_pullback")

        # Bearish exhaustion in macro bull → suppress
        if tf["direction"] == "bearish" and macro_bull:
            ema50 = macro.get("ema50", 0)
            price_vs = macro.get("price_vs_ema50_pct", 0)
            if tf["confidence"]:
                tf["confidence"] = max(25, tf["confidence"] - 20)
            # Add suppressed flag for evolution tracking
            tf["macro_suppressed"] = True
            tf["regime"] = "ranging"
            tf["direction"] = None
            tf["strength"] = "无"

        # Bullish exhaustion in macro bear → suppress
        elif tf["direction"] == "bullish" and macro_bear:
            if tf["confidence"]:
                tf["confidence"] = max(25, tf["confidence"] - 20)
            tf["macro_suppressed"] = True
            tf["regime"] = "ranging"
            tf["direction"] = None
            tf["strength"] = "无"

    # Refresh h1/h30 references after macro filter modifications
    h1 = results["1h"]
    h30 = results["30m"]

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

    # DI spread too low (with hysteresis dead zone to prevent flickering)
    _min_di = all_thresholds.get("min_di_spread", 3)
    if h4.get("di_spread", 0) < _min_di - 0.5 and not circuit_breaker_reason:
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

    # ─── Check previous decay state (for hysteresis) ───
    # Prevents DECAYING from flickering on/off when metrics hover near thresholds.
    # Once DECAYING fires, it requires stronger improvement to clear.
    prev_was_decaying = False
    if history_rows and len(history_rows) >= 1:
        prev_advice = (history_rows[0].get("advice") or "")
        # "观望（跌趋势衰减中）" or similar contains "衰减" when DECAYING active
        prev_was_decaying = "衰减" in prev_advice

    # ─── Trend Exhaustion & Late-Trend Filters ───────────────────────────
    # Three-tier decay model (not binary block):
    #   NONE      — no decay, normal entries
    #   DECAYING  — trend weakening, block entries, wait for direction confirmation
    #   EXHAUSTED — trend over (ADI extreme or below threshold), block entries
    trend_decay_level = "NONE"  # NONE | DECAYING | EXHAUSTED
    exhaustion_reason = None

    h4_adx_threshold = tf_cfg.get("4h", {}).get("adx_trending_threshold", 25)

    # Filter 1: ADX ceiling — extreme ADX means trend is mature/ending.
    # ADX > 45 is rarely sustainable in crypto; new entries at this level are chasing.
    # Exception: if ADX is still rising fast (fast > slow + 5), the trend may have legs.
    adx_ceiling = 45
    if h4["adx"] > adx_ceiling and h1["adx"] > adx_ceiling:
        # Both 4h and 1h at extreme — trend is very mature
        if not (h4["adx_fast"] > h4["adx_slow"] + 5 and h1["adx_fast"] > h1["adx_slow"] + 5):
            trend_decay_level = "EXHAUSTED"
            exhaustion_reason = f"ADX极值（4h={h4['adx']:.0f}, 1h={h1['adx']:.0f}），趋势末端，禁止追单"
    elif h4["adx"] > adx_ceiling:
        # Only 4h at extreme — risky but not necessarily over
        if not (h4["adx_fast"] > h4["adx_slow"] + 5):
            trend_decay_level = "DECAYING"
            exhaustion_reason = f"4h ADX={h4['adx']:.0f}超阈值，趋势可能即将衰减"

    # Filter 2: ADX falling from peak — trend is actively decaying.
    # Peak ADX is tracked from last 24h of verdict_history (rolling window).
    # Uses relative drop (20% of peak) instead of fixed 5 points, which adapts
    # to different ADX levels (ADX 40→32 is different from 25→20).
    # Short-circuit: if already EXHAUSTED from Filter 1, skip remaining filters.
    if trend_decay_level == "EXHAUSTED":
        pass  # already blocked, skip Filters 2-4
    elif peak_adx > 0:
        adx_drop_pct = (peak_adx - h4["adx"]) / max(peak_adx, 1)
        h4_di_spread = abs(h4["plus_di"] - h4["minus_di"])
        di_crossed = h4_di_spread < 5  # DI spread < 5 = direction uncertain

        if adx_drop_pct >= 0.20:
            # ADX dropped ≥20% from peak — significant decay
            # Soft zone: allow DECAYING if ADX is within 1 point of threshold,
            # to avoid hard boundary effects (e.g., 24.9 vs 25.0 shouldn't
            # flip from DECAYING to EXHAUSTED).
            soft_zone = h4_adx_threshold - 1.0
            if h4["adx"] >= h4_adx_threshold:
                trend_decay_level = "DECAYING"
                exhaustion_reason = f"ADX从峰值{peak_adx:.0f}回落至{h4['adx']:.0f}（- {adx_drop_pct*100:.0f}%），趋势已衰减"
                if di_crossed:
                    trend_decay_level = "EXHAUSTED"
                    exhaustion_reason = f"ADX从峰值{peak_adx:.0f}回落至{h4['adx']:.0f}（- {adx_drop_pct*100:.0f}%），DI差收窄（{h4_di_spread:.0f}），趋势已结束"
            elif h4["adx"] >= soft_zone:
                # ADX slightly below threshold but close — still allow DECAYING
                trend_decay_level = "DECAYING"
                exhaustion_reason = f"ADX从峰值{peak_adx:.0f}回落至{h4['adx']:.0f}（- {adx_drop_pct*100:.0f}%），略低于趋势阈值"
                if di_crossed:
                    exhaustion_reason += "，DI差收窄，谨慎观望"
            else:
                # ADX significantly below threshold — but check if DI spread
                # is still wide (strong directional conviction despite ADX drop).
                # During sharp one-sided moves, ADX can compress as DI becomes
                # extremely one-sided (minus_DI dominates), which doesn't mean
                # the trend is over — it means it's very directional.
                if h4_di_spread >= 15:
                    trend_decay_level = "DECAYING"
                    exhaustion_reason = f"ADX从峰值{peak_adx:.0f}回落至{h4['adx']:.0f}（- {adx_drop_pct*100:.0f}%），低于趋势阈值，但DI差仍大（{h4_di_spread:.0f}），趋势仍在"
                elif di_crossed:
                    trend_decay_level = "EXHAUSTED"
                    exhaustion_reason = f"ADX从峰值{peak_adx:.0f}回落至{h4['adx']:.0f}（- {adx_drop_pct*100:.0f}%），DI差收窄（{h4_di_spread:.0f}），趋势已结束"
                else:
                    # ADX below threshold and neither wide spread nor DI cross
                    # — moderate decay, let it through but flag as DECAYING
                    trend_decay_level = "DECAYING"
                    exhaustion_reason = f"ADX从峰值{peak_adx:.0f}回落至{h4['adx']:.0f}（- {adx_drop_pct*100:.0f}%），趋势动能减弱"
        elif adx_drop_pct >= 0.10:
            # ADX dropped 10-20% — mild decay, note but don't block
            if di_crossed:
                trend_decay_level = "DECAYING"
                exhaustion_reason = f"ADX从峰值{peak_adx:.0f}回落至{h4['adx']:.0f}（- {adx_drop_pct*100:.0f}%），DI差收窄（{h4_di_spread:.0f}），趋势动能减弱"

    # Filter 3: DI convergence — +DI and -DI closing in means directional
    # momentum is fading. Only trigger when ADX > 30: at low ADX (<30) small
    # DI spread is normal for weak trends; at high ADX (>30) a small spread
    # means "high strength but no direction" — a fake trend.
    h4_di_spread = abs(h4["plus_di"] - h4["minus_di"])
    if trend_decay_level == "NONE" and h4_di_spread < h4["adx"] * 0.25 and h4["adx"] > 30:
        trend_decay_level = "DECAYING"
        exhaustion_reason = f"DI差收窄（{h4_di_spread:.0f} < ADX*0.25={h4['adx']*0.25:.0f}），趋势动能减弱"

    # Filter 4: Price structure reversal — last 3 candles show reversal pattern.
    # Bearish: if last 2 candles are bullish (close > open) and the last candle
    # engulfs the previous one, the downtrend is reversing.
    # Bullish: opposite — last 2 candles bearish + engulfing.
    if trend_decay_level == "NONE" and len(all_candles_4h) >= 4:
        last3 = all_candles_4h[-3:]
        c0, c1, c2 = last3[0], last3[1], last3[2]
        # Check for reversal candlestick pattern
        if h4["direction"] == "bearish":
            # Bearish reversal: recent candles showing buying pressure
            bullish_count = sum(1 for c in last3 if c["close"] > c["open"])
            last_range = c2["high"] - c2["low"]
            last_body = c2["close"] - c2["open"]
            # Long lower shadow = rejection of lower prices
            lower_shadow = c2["close"] - c2["low"] if c2["close"] > c2["open"] else c2["open"] - c2["low"]
            long_shadow = lower_shadow > (last_range * 0.5) if last_range > 0 else False
            if bullish_count >= 2 and long_shadow:
                trend_decay_level = "DECAYING"
                exhaustion_reason = "价格结构反转：连续阳线+长下影，看跌动能衰减"
        elif h4["direction"] == "bullish":
            bearish_count = sum(1 for c in last3 if c["close"] < c["open"])
            last_range = c2["high"] - c2["low"]
            upper_shadow = c2["high"] - c2["close"] if c2["close"] < c2["open"] else c2["high"] - c2["open"]
            long_upper = upper_shadow > (last_range * 0.5) if last_range > 0 else False
            if bearish_count >= 2 and long_upper:
                trend_decay_level = "DECAYING"
                exhaustion_reason = "价格结构反转：连续阴线+长上影，看涨动能衰减"

    # ─── DECAYING Hysteresis ───
    # Prevents flickering when metrics hover near thresholds. Once DECAYING
    # fires, require stronger improvement to clear: ADX must be at least
    # trending threshold AND DI spread must be wide enough.
    if prev_was_decaying and trend_decay_level == "NONE":
        h4_di_spread = abs(h4["plus_di"] - h4["minus_di"])
        adx_recovered = h4["adx"] >= h4_adx_threshold and h4_di_spread >= 8
        if not adx_recovered:
            # Still in decay — don't clear
            trend_decay_level = "DECAYING"
            exhaustion_reason = exhaustion_reason or f"上一周期处于衰减状态，指标未充分恢复（ADX={h4['adx']:.0f}, DI差={h4_di_spread:.0f}）"

    # ─── Filter 5: DI Convergence Warning ──────────────────────────────
    # Detect trend weakening BEFORE exhaustion kicks in.
    # This is an auxiliary signal — does NOT block entries, just warns.
    # Conditions:
    #   (a) DI spread has been shrinking for last 3 candles (converging)
    #   (b) ADX was strong (> 30) but is now declining
    #   (c) Price is moving against the original trend direction
    convergence_warning = None
    h4_converging = h4.get("di_spread_converging", False)
    h1_converging = h1.get("di_spread_converging", False)
    if h4["adx"] > 30 and h4["direction"]:
        # Check if ADX is declining (fast < slow = ADX falling)
        adx_declining = h4["adx_fast"] < h4["adx_slow"]
        # Check if price is moving against current direction
        if all_candles_4h and len(all_candles_4h) >= 4:
            price_reversing = False
            last3_closes = [c["close"] for c in all_candles_4h[-3:]]
            if h4["direction"] == "bearish" and last3_closes[2] > last3_closes[0]:
                price_reversing = True
            elif h4["direction"] == "bullish" and last3_closes[2] < last3_closes[0]:
                price_reversing = True

            # Need at least one TF (4h or 1h) converging + ADX declining + price reversing
            if (h4_converging or h1_converging) and adx_declining and price_reversing:
                spread_hist = h4.get("di_spread_history", [])
                if len(spread_hist) >= 3:
                    from_str = spread_hist[-3]
                    to_str = spread_hist[-1]
                    convergence_warning = {
                        "detected": True,
                        "old_direction": h4["direction"],
                        "reason": f"DI spread 从 {from_str:.1f} 收敛至 {to_str:.1f}，ADX 从峰值回落，价格缓慢{'回升' if h4['direction'] == 'bearish' else '回落'}",
                    }

    # ─── Two-Stage Signal System ───────────────────────────────────────
    # Stage 1: 预警 (Warning/Preparation) — early signals at trend inflection points.
    #   Light position (10-15%) or alert. Lower confidence but earlier timing.
    # Stage 2: 确认 (Confirmation) — traditional multi-TF aligned entry.
    #   Full position when ADX/DI conditions fully met.
    #
    # Stage 1 triggers when ANY of these conditions are met:
    #   S1-A: DI convergence + price breakout (trend reversal early)
    #   S1-B: 4h forming + 1h trending (trend startup early)
    #   S1-C: 4h ranging + 1h/30m breakout (ranging breakout early)
    stage_1_warning = None

    # S1-A: DI convergence + price breakout — early reversal signal
    # When an established trend shows DI spread collapsing AND price breaks
    # recent structure, this is the earliest sign of trend change.
    # Only applies when ADX is in the [trending, 1.5×trending) range —
    # established trend but not yet mature/extreme.
    h4_s1a_trending = tf_cfg.get("4h", {}).get("adx_trending_threshold", 25)
    if h4["direction"] and h4["adx"] >= h4_s1a_trending and h4["adx"] < h4_s1a_trending * 1.5:
        h1_dir = h1.get("direction")
        new_dir = "bullish" if h4["direction"] == "bearish" else "bearish"
        new_label = "看涨" if new_dir == "bullish" else "看跌"
        old_label = "看跌" if new_dir == "bullish" else "看涨"

        # Check price breakout: last candle breaks the high/low of previous 6 candles
        if all_candles_4h and len(all_candles_4h) >= 8:
            last6_high = max(c["high"] for c in all_candles_4h[-7:-1])
            last6_low = min(c["low"] for c in all_candles_4h[-7:-1])
            last_close = all_candles_4h[-1]["close"]

            price_broke_high = last_close > last6_high
            price_broke_low = last_close < last6_low

            # DI spread must be narrowing (< 8 means losing directional conviction)
            di_spread_narrow = h4_di_spread < 8

            # Volume check: last candle volume > 1.2x average of previous 6
            vols_4h = [c["volume"] for c in all_candles_4h[-7:]]
            vol_amplified = (vols_4h[-1] > sum(vols_4h[:-1]) / max(len(vols_4h) - 1, 1) * 1.2) if len(vols_4h) >= 3 else False

            if new_dir == "bullish" and price_broke_high and di_spread_narrow:
                stage_1_warning = {
                    "type": "convergence_breakout",
                    "direction": new_dir,
                    "direction_label": new_label,
                    "confidence": 30,
                    "position_pct_hint": 10,
                    "reason": (
                        f"{old_label}趋势末端：DI spread收窄至{h4_di_spread:.1f}，"
                        f"价格突破近6根高点{last6_high:.0f}，"
                        f"{'量能放大，' if vol_amplified else ''}"
                        f"可轻仓试探{new_label}"
                    ),
                    "volume_confirmed": vol_amplified,
                }
            elif new_dir == "bearish" and price_broke_low and di_spread_narrow:
                stage_1_warning = {
                    "type": "convergence_breakout",
                    "direction": new_dir,
                    "direction_label": new_label,
                    "confidence": 30,
                    "position_pct_hint": 10,
                    "reason": (
                        f"{old_label}趋势末端：DI spread收窄至{h4_di_spread:.1f}，"
                        f"价格跌破近6根低点{last6_low:.0f}，"
                        f"{'量能放大，' if vol_amplified else ''}"
                        f"可轻仓试探{new_label}"
                    ),
                    "volume_confirmed": vol_amplified,
                }

    # S1-B: 4h forming + 1h trending — early trend startup (already handled by h4_forming_quality)
    # This maps to the existing h4_forming_quality path; just tag it for stage_1 display
    # (set later in Priority 2 block)

    # S1-C: 4h ranging + 1h/30m breakout — ranging breakout (already handled by ranging_breakout)
    # This maps to the existing ranging_breakout path; just tag it for stage_1 display
    # (set later in Priority 3 block)

    # ─── Multi-Timeframe Alignment Priority ───
    alignment_rule = None
    h4_reg = h4["regime"]
    h1_reg = h1["regime"]
    orig_h4_direction = h4["direction"]  # save before potential flip
    _dir_override_log = []  # track direction flips for debugging

    # Track if this is a "forming quality signal" — 4h forming but 1h strongly trending.
    # Used in action advice to allow controlled entry instead of blanket wait.
    h4_forming_quality = False

    # Track ranging breakout signal for action advice and market_context output.
    ranging_breakout = None

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

    # Priority 2: 4h forming + 1h trending → early signal with quality check
    elif h4_reg == "forming" and h1_reg == "trending":
        alignment_rule = "规则2: 4h forming + 1h trending 早期信号"
        # Quality filter: quantify "不反对" with DI alignment.
        # Compute how strongly 4h DI opposes 1h direction:
        #   If 1h is bullish, check 4h -DI vs +DI. If -DI >> +DI, 4h opposes.
        #   If 1h is bearish, check 4h +DI vs -DI. If +DI >> -DI, 4h opposes.
        # Skip if 4h DI opposition exceeds threshold (spread > 5 against).
        h4_di_opposition = None
        di_opposition_threshold = 5
        if h1["direction"] == "bullish":
            h4_di_opposition = h4["minus_di"] - h4["plus_di"]
        else:
            h4_di_opposition = h4["plus_di"] - h4["minus_di"]

        if h4_di_opposition is not None and h4_di_opposition > di_opposition_threshold:
            # 4h DI structure strongly opposes 1h direction — skip
            alignment_rule += f"（4h DI结构反对1h方向（opposition={h4_di_opposition:.1f} > {di_opposition_threshold}），暂不介入）"
            confidence = max(30, h1["confidence"] - 15)
        elif h4["direction"] and h4["direction"] != h1["direction"]:
            # 4h has opposite direction but DI spread not large enough to block
            # Lower confidence but still allow — weak opposition
            alignment_rule += "（4h方向与1h轻微相反，降低仓位）"
            h4["direction"] = h1["direction"]
            _dir_override_log.append(f"P2: {orig_h4_direction} → {h1['direction']} (1h forming override)")
            h4_forming_quality = True
            confidence = max(30, h1["confidence"] - 10)
            stage_1_warning = {
                "type": "trend_startup",
                "direction": h1["direction"],
                "direction_label": "看涨" if h1["direction"] == "bullish" else "看跌",
                "confidence": 30,
                "position_pct_hint": 10,
                "reason": (
                    f"4h趋势正在形成（{'看涨' if h1['direction'] == 'bullish' else '看跌'}），"
                    f"ADX={h4['adx']:.0f}，1h趋势已确认同向（ADX={h1['adx']:.0f}），"
                    f"但4h DI轻微反对，减仓试探"
                ),
            }
        else:
            # 4h direction matches or is None — 1h trend is trustworthy
            h4["direction"] = h1["direction"]
            _dir_override_log.append(f"P2: {orig_h4_direction} → {h1['direction']} (1h forming match)")
            h4_forming_quality = True
            confidence = max(35, h1["confidence"] - 5)
            # Stage 1: early trend startup signal
            stage_1_warning = {
                "type": "trend_startup",
                "direction": h1["direction"],
                "direction_label": "看涨" if h1["direction"] == "bullish" else "看跌",
                "confidence": 35,
                "position_pct_hint": 15,
                "reason": (
                    f"4h趋势正在形成（{'看涨' if h1['direction'] == 'bullish' else '看跌'}），"
                    f"ADX={h4['adx']:.0f}，1h趋势已确认同向（ADX={h1['adx']:.0f}），"
                    f"可轻仓试探，等待4h确认后加仓"
                ),
            }

    # Priority 2b: 4h forming + 1h forming, same direction with wide DI spread
    # → all-TF directional alignment, allow light entry
    elif h4_reg == "forming" and h1_reg == "forming" and h4.get("direction") and h4["direction"] == h1.get("direction"):
        alignment_rule = "规则2b: 4h+1h 同时 forming 同向确认"
        h4_forming_quality = True
        h4["direction"] = h1["direction"]  # ensure alignment
        _dir_override_log.append(f"P2b: {orig_h4_direction} → {h1['direction']} (4h+1h forming same dir)")
        confidence = max(30, h4["confidence"] - 5)
        # Stage 1: light entry
        stage_1_warning = {
            "type": "trend_startup",
            "direction": h1["direction"],
            "direction_label": "看涨" if h1["direction"] == "bullish" else "看跌",
            "confidence": 30,
            "position_pct_hint": 10,
            "reason": (
                f"4h+1h同时形成态（{'看涨' if h1['direction'] == 'bullish' else '看跌'}），"
                f"4h ADX={h4['adx']:.0f}, 1h ADX={h1['adx']:.0f}，"
                f"多时间框架方向一致，可轻仓试探"
            ),
        }

    # Priority 3: 4h ranging/low_vol_trend → check for breakout or low-vol trend
    elif h4_reg == "ranging":
        # ─── Ranging Breakout: 4h ranging + 1h/30m breakout 轻仓试单 ───
        # When 4h is consolidating (ADX < 23), check if smaller timeframes
        # show a clear directional trend worth trading with reduced size.
        h1_dir = h1.get("direction")
        h30_dir = h30.get("direction")
        h1_di_spread = h1.get("di_spread", 0)

        # Conditions for ranging breakout:
        # (a) 1h has clear direction (DI spread > 8)
        # (b) 1h is trending, breakout, OR forming (sharp moves can have low ADX)
        # (c) 30m confirms same direction
        # (d) 4h DI spread not extremely low (< 3 means truly dead)
        # (e) Macro filter: block bearish breakout in macro bull
        # (f) Cooldown: at least 2h since last P3 ranging breakout of same direction
        # EXCEPTION: if 4h is upgrading from ranging to trending/forming with
        # strong ADX momentum, skip cooldown — main trend overrides P3 cooling.
        recent_p3_signals = market_ctx.get("recent_p3_signals", [])
        p3_cooldown_seconds = 7200  # 2h between P3 ranging breakout signals
        now_time = time.time()

        # Trend upgrade override: when 4h ADX is rising fast and approaching
        # trending threshold, the market is transitioning out of ranging.
        # Don't let P3 cooldown block this upgrade path.
        upgrading_from_ranging = (
            h4["adx"] >= h4.get("effective_trending", 25) - 3
            and h4["adx_fast"] > h4["adx_slow"]
        )

        p3_in_cooldown = False
        if not upgrading_from_ranging:
            p3_in_cooldown = any(
                (now_time - s["created_at"]) < p3_cooldown_seconds
                for s in recent_p3_signals if s.get("direction") == h1_dir
            )

        if (h1_reg in ("trending", "breakout", "forming") and h1_dir
                and h30_dir == h1_dir
                and h1_di_spread > 15
                and h4_di_spread >= 5
                and h4["adx"] > 10
                and not ((macro["is_bull"] or macro.get("is_bull_pullback")) and h1_dir == "bearish")
                and not p3_in_cooldown):
            ranging_breakout = {
                "direction": h1_dir,
                "confidence": 35,
                "signal_subtype": "ranging_breakout",
                "reason": (
                    f"4h盘整（ADX={h4['adx']:.0f}），"
                    f"1h{'趋势' if h1_reg == 'trending' else '突破'}{'看涨' if h1_dir == 'bullish' else '看跌'}"
                    f"（DI价差={h1_di_spread:.1f}），"
                    f"30m同向确认，可轻仓试单"
                ),
            }
            alignment_rule = "规则3a: 4h ranging + 1h/30m breakout 轻仓试单"
            h4["direction"] = h1_dir
            _dir_override_log.append(f"P3a: {orig_h4_direction} → {h1_dir} (ranging breakout)")
            h4["signal_type"] = "ranging_breakout"
            confidence = 35
            # P3 ranging breakout: bind stop to breakout structure.
            # Use recent 4h swing low/high as stop reference, not just ATR.
            # This prevents fakeout losses when price reverses back into the range.
            if len(all_candles_4h) >= 10:
                recent_10 = all_candles_4h[-10:]
                if h1_dir == "bullish":
                    breakout_stop = min(c["low"] for c in recent_10)
                    p3_stop = round(breakout_stop - atr4 * 0.5, 1)
                else:
                    breakout_stop = max(c["high"] for c in recent_10)
                    p3_stop = round(breakout_stop + atr4 * 0.5, 1)
                # Ensure stop is at least as wide as ATR*1.0
                min_p3_stop_dist = current_price * 0.001  # 0.1% floor
                if abs(p3_stop - current_price) < min_p3_stop_dist:
                    p3_stop = round(current_price - atr4 if h1_dir == "bullish" else current_price + atr4, 1)
            else:
                # Not enough candles, fall back to ATR-based stop
                p3_stop = round(current_price - atr4 * 1.0 if h1_dir == "bullish" else current_price + atr4 * 1.0, 1)

            # Stage 1: ranging breakout signal
            stage_1_warning = {
                "type": "ranging_breakout",
                "direction": h1_dir,
                "direction_label": "看涨" if h1_dir == "bullish" else "看跌",
                "confidence": 35,
                "position_pct_hint": 10,
                "stop": p3_stop,
                "reason": (
                    f"4h盘整期突破：1h{'趋势' if h1_reg == 'trending' else '突破'}{'看涨' if h1_dir == 'bullish' else '看跌'}，"
                    f"30m同向确认，可轻仓试探（正常仓位25%），"
                    f"止损基于近期4h波段{'低点' if h1_dir == 'bullish' else '高点'}，"
                    f"等待4h趋势确认后加仓"
                ),
            }
        # ─── Macro Bias Long: 4h ranging + bullish macro bias ───
        # In a bull market, price can climb steadily without large candles.
        # Allow light long entry when:
        #   (a) Macro is bullish or in bull pullback
        #   (b) 4h price > EMA20 and +DI > -DI
        #   (c) At least one TF (1h or 30m) shows止跌结构:
        #       - Price in lower 40% of 20-candle range (not chasing highs)
        #       - OR recent 3 candles show bullish reversal (higher lows)
        # (d) Cooldown: at least 3h since last macro bias signal
        # This prevents blindly longing in a ranging downtrend.
        elif (macro["is_bull"] or macro.get("is_bull_pullback")) and not ranging_breakout:
            # Cooldown for macro bias signals (same trend-upgrade override)
            p3_cooldown_seconds = 10800  # 3h between macro bias signals

            if upgrading_from_ranging:
                macro_bias_in_cooldown = False  # trend upgrade overrides cooldown
            else:
                macro_bias_in_cooldown = any(
                    (now_time - s["created_at"]) < p3_cooldown_seconds
                    for s in recent_p3_signals if s.get("signal_type") == "macro_bias"
                )

            if macro_bias_in_cooldown:
                alignment_rule = "规则3b: 4h ranging + 宏观看涨但冷却中"
                confidence = h4["confidence"]
            else:
                ema20_4h = h4.get("ema20", h4["price"])
                if h4["price"] > ema20_4h and h4["plus_di"] > h4["minus_di"]:
                    # Price structure confirmation from 1h or 30m
                    h1_pct = h1.get("entry_position", {}).get("percentile", 50)
                    h30_pct = h30.get("entry_position", {}).get("percentile", 50)
                    h1_ps = h1.get("price_structure", {})
                    h30_ps = h30.get("price_structure", {})

                    # 止跌条件: (1) 价格在区间下半部分(< 50%), 或 (2) 出现higher lows
                    h1_bullish = h1_pct < 50 or h1_ps.get("higher_lows", False)
                    h30_bullish = h30_pct < 50 or h30_ps.get("higher_lows", False)

                    if h1_bullish or h30_bullish:
                        ranging_breakout = {
                            "direction": "bullish",
                            "confidence": 30,
                            "signal_subtype": "macro_bias",
                            "reason": (
                                f"4h盘整（ADX={h4['adx']:.0f}），但宏观看涨"
                                f"（价格>EMA20, +DI={h4['plus_di']:.0f} > -DI={h4['minus_di']:.0f}），"
                                f"{'1h' if h1_bullish else '30m'}止跌结构确认，可轻仓试探做多"
                            ),
                        }
                        alignment_rule = "规则3b: 4h ranging + 宏观看涨 bias"
                        h4["direction"] = "bullish"
                        _dir_override_log.append(f"P3b: {orig_h4_direction} → bullish (macro bias long)")
                        h4["signal_type"] = "macro_bias_long"
                        confidence = 30
                        # Stage 1: macro bias signal
                        stage_1_warning = {
                            "type": "macro_bias",
                            "direction": "bullish",
                            "direction_label": "看涨",
                            "confidence": 30,
                            "position_pct_hint": 8,
                            "reason": (
                                f"4h盘整但宏观看涨，价格高于EMA20（{ema20_4h:.0f}），"
                                f"+DI={h4['plus_di']:.0f} > -DI={h4['minus_di']:.0f}，"
                                f"可轻仓试探（建议8%仓位）"
                            ),
                        }
                    else:
                        # Macro bullish but no止跌结构 — price may be falling in range
                        alignment_rule = "规则3b: 4h ranging + 宏观看涨但无止跌结构"
                        confidence = h4["confidence"]
                else:
                    # Macro bullish but price/EMA20 or DI doesn't support entry
                    alignment_rule = "规则3b: 4h ranging + 宏观看涨但条件不足"
                    confidence = h4["confidence"]
        else:
            alignment_rule = "规则3: 4h ranging 观望"
            confidence = h4["confidence"]
    elif h4_reg == "low_vol_trend":
        alignment_rule = "规则3c: 4h 低波动趋势（即将变盘）"
        # Follow 1h direction for entry
        if h1_reg == "trending" and h1["direction"]:
            h4["direction"] = h1["direction"]
            _dir_override_log.append(f"P3c: {orig_h4_direction} → {h1['direction']} (low_vol_trend 1h confirm)")
            confidence = max(45, h1["confidence"] - 5)
        else:
            confidence = 50

    # Priority 3d: Trend exhaustion reversal — ADX high but declining.
    # This catches the case where regime is still "trending" (hysteresis
    # preventing switch to exhaustion) but the trend is clearly ending.
    # Triggers when: ADX >= 40, ADX declining (fast < slow), price structure
    # showing reversal, AND smaller TFs confirm opposite direction.
    elif h4_reg in ("trending", "low_vol_trend") and h4["adx"] >= h4_adx_threshold * 1.6:
        adx_declining = h4["adx_fast"] < h4["adx_slow"]
        momentum_weakening = h4.get("momentum") in ("减弱", "衰竭")
        price_reversal = False
        h30_ps = h30.get("price_structure", {})
        if h4["direction"] == "bullish":
            # Bearish reversal: price making lower highs + lower lows
            price_reversal = h30_ps.get("type") == "bearish"
        elif h4["direction"] == "bearish":
            # Bullish reversal: price making higher lows + higher highs
            price_reversal = h30_ps.get("type") == "bullish"

        if adx_declining and (momentum_weakening or price_reversal):
            # Hysteresis: if the previous 4h signal was already an exhaustion_reversal,
            # the direction was already flipped. Don't flip again — maintain it.
            # This prevents whipsaw when P3d confirmation conditions oscillate
            # between consecutive API calls (e.g., 30m price_structure flipping).
            prev_4h_signal_type = (previous_regimes or {}).get("4h_signal_type")
            is_already_reversed = prev_4h_signal_type == "exhaustion_reversal"

            if is_already_reversed:
                # Already reversed in a previous cycle — maintain direction without
                # re-evaluating. Prevents flip-back when confirmation weakens.
                # CRITICAL: also lock direction to previous reversed value, because
                # the base ADX/DI calculation below may produce a different direction
                # if DI lines shift slightly (e.g., +DI 14/-DI 19 vs +DI 15/-DI 18).
                #
                # If base direction catches up to the reversal (trend flipped as
                # predicted), keep the reversal direction — it's validated, not invalid.
                prev_4h_direction = (previous_regimes or {}).get("4h_direction")
                if prev_4h_direction:
                    h4["direction"] = prev_4h_direction
                    alignment_rule = "规则3d: 衰竭反转已锁定（防止来回切换）"
                    h4["signal_type"] = "exhaustion_reversal"
                    h4["regime"] = "exhaustion"
                    confidence = max(35, h4["confidence"] - 15)
                    h4_reg = "exhaustion"
                # If no prev direction available, keep current h4 direction as-is
            else:
                new_direction = "bullish" if h4["direction"] == "bearish" else "bearish"
                h30_confirms_reversal = (h30.get("direction") == new_direction)
                h1_supports_reversal = (h1.get("direction") == new_direction) or h1_reg == "exhaustion"

                if h30_confirms_reversal or h1_supports_reversal:
                    # Block bearish reversal in macro bull
                    if new_direction == "bearish" and (macro.get("is_bull") or macro.get("is_bull_pullback")):
                        alignment_rule = "规则3d: 趋势衰竭但宏观看涨禁止做空"
                        confidence = max(30, h4["confidence"] - 20)
                    else:
                        alignment_rule = "规则3d: 趋势衰竭反转预警（ADX高但正在衰减）"
                        h4["direction"] = new_direction
                        h4["regime"] = "exhaustion"  # override for downstream action advice routing
                        _dir_override_log.append(f"P3d: {orig_h4_direction} → {new_direction} (exhaustion reversal)")
                        h4["signal_type"] = "exhaustion_reversal"
                        confidence = max(35, h4["confidence"] - 15)
                        h4_reg = "exhaustion"  # override for downstream logic
                else:
                    # ADX declining but no reversal confirmation yet — warn but don't flip
                    alignment_rule = "规则3d: 趋势衰减中，等待反转确认"
                    confidence = max(30, h4["confidence"] - 10)

    # Priority 4: 4h exhaustion → reversal signal
    elif h4_reg == "exhaustion":
        # Check if smaller TFs confirm or contradict
        if h1_reg == "trending" and h1["direction"] == h4["direction"]:
            alignment_rule = "规则4b: 4h衰竭但1h趋势延续，可能是回调"
            # Don't flip direction — treat as pullback within trend
            confidence = max(40, h4["confidence"] - 10)
        else:
            # Hysteresis: if already reversed in previous cycle, maintain direction.
            prev_4h_signal_type = (previous_regimes or {}).get("4h_signal_type")
            is_already_reversed = prev_4h_signal_type == "exhaustion_reversal"

            if is_already_reversed:
                # Already reversed — maintain without re-evaluating to prevent whipsaw.
                # CRITICAL: also lock direction to previous reversed value.
                #
                # SAFETY CHECK: if locked direction equals current base direction,
                # the underlying trend has flipped to match the reversal prediction
                # (trend caught up). This VALIDATES the reversal — keep the direction
                # as-is, DON'T flip back.
                # Only recalculate when hysteresis conditions change (regime switch).
                prev_4h_direction = (previous_regimes or {}).get("4h_direction")
                if prev_4h_direction:
                    h4["direction"] = prev_4h_direction
                    alignment_rule = "规则4: 衰竭反转已锁定（防止来回切换）"
                    h4["signal_type"] = "exhaustion_reversal"
                    confidence = max(35, h4["confidence"] - 15)
                # If no prev direction available, keep current h4 direction as-is
            else:
                # ─── Exhaustion reversal logic ───
                # Flip direction: when 4h exhaustion + 1h not continuing the old trend,
                # flip the direction to the opposite of the original 4h direction.
                # This is a clear, unambiguous definition: new_direction = flip(orig_h4_direction).
                new_direction = "bullish" if orig_h4_direction == "bearish" else "bearish"

                # Block bearish reversal in macro bull or bull pullback — likely just a pullback, not a reversal
                if new_direction == "bearish" and (macro.get("is_bull") or macro.get("is_bull_pullback")):
                    alignment_rule = "规则4: 4h衰竭，但宏观看涨禁止做空"
                    confidence = max(30, h4["confidence"] - 20)
                    h4["direction"] = "bullish"  # Keep original bullish direction
                    _dir_override_log.append(f"P4: {orig_h4_direction} → bullish (macro bull blocked)")
                    h4["signal_type"] = "macro_blocked"  # Macro bullish blocked bearish signal
                else:
                    # Cooling check: prevent repeated reversal signals in volatile markets.
                    # Require: (a) at least 1h since last reversal signal of SAME direction, AND
                    # (b) 30m confirms the new direction (not flip-flopping).
                    recent_reversals = market_ctx.get("recent_reversals", [])
                    cooldown_seconds = 3600  # 1h minimum between reversal signals
                    now_time = time.time()
                    last_reversal_same_dir = None
                    for rr in recent_reversals:
                        if rr["direction"] == new_direction:
                            last_reversal_same_dir = rr
                            break

                    cooling_active = False
                    if last_reversal_same_dir:
                        elapsed = now_time - last_reversal_same_dir["created_at"]
                        if elapsed < cooldown_seconds:
                            cooling_active = True

                    h30_confirms = (h30["direction"] == new_direction) if h30["direction"] else False

                    if cooling_active:
                        alignment_rule = "规则4: 4h衰竭，反转冷却中（1h内已发过同向信号）"
                        confidence = max(30, h4["confidence"] - 20)
                    elif h30_confirms:
                        alignment_rule = "规则4: 4h exhaustion 反向信号（轻仓试探）"
                        h4["direction"] = new_direction
                        _dir_override_log.append(f"P4: {orig_h4_direction} → {new_direction} (exhaustion reversal)")
                        h4["signal_type"] = "exhaustion_reversal"
                        confidence = max(35, h4["confidence"] - 15)
                    else:
                        alignment_rule = "规则4: 4h衰竭，30m未确认反转方向"
                        h4["direction"] = new_direction
                        _dir_override_log.append(f"P4: {orig_h4_direction} → {new_direction} (exhaustion unconfirmed)")
                        h4["signal_type"] = "exhaustion_reversal"
                        confidence = max(30, h4["confidence"] - 15)

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

    # If convergence warning detected but not already tagged as reversal, mark it
    if convergence_warning and not h4.get("signal_type"):
        h4["signal_type"] = "convergence_warning"

    # Action advice (replaces vague directions_match logic)
    # Gate: trend exhaustion filter blocks trend-following entries,
    # but exhaustion reversal signals (with 30m confirmation) are allowed through
    is_reversal_signal = (
        h4["regime"] == "exhaustion"
        and h4.get("direction") != orig_h4_direction
        and h4.get("signal_type") == "exhaustion_reversal"
    )
    # Trend relay: 4h exhaustion but smaller TFs forming in original direction —
    # allow light entry through the exhaustion gate.
    is_trend_relay = (
        h4["regime"] == "exhaustion"
        and (
            (h1["regime"] == "forming" and h1["direction"] == orig_h4_direction)
            or (h30["regime"] == "forming" and h30["direction"] == orig_h4_direction)
        )
    )
    if trend_decay_level == "EXHAUSTED" and not is_reversal_signal and not is_trend_relay:
        action = "趋势末端禁止入场"
        side = "观望"
        reason = exhaustion_reason
        target = None
        stop = None
    elif h4["regime"] == "trending" and h4["adx"] >= h4_adx_threshold * 2.0:
        # ADX ≥ 50 = extreme overbought/oversold, trend is exhausted.
        # Do NOT open new trend-following positions — wait for pullback or reversal.
        dir_label = "涨" if h4["direction"] == "bullish" else "跌"
        action = f"趋势过热禁止追单（ADX={h4['adx']:.0f}）"
        side = "观望"
        reason = f"ADX={h4['adx']:.0f}处于极端高位，趋势已过度展开，追{dir_label}风险极高，等待回调或反转确认"
        # Dedup: if 3+ consecutive exhaustion_block signals with same direction already
        # exist, suppress generating another one. The market_ctx carries the count from
        # main.py; when suppressed, keep signal_type=None (state snapshot only) to
        # avoid polluting verification stats with redundant identical warnings.
        eb_count = market_context.get("exhaustion_block_consecutive", 0) if market_context else 0
        if eb_count >= 3:
            h4["signal_type"] = "none"  # suppress redundant signal
        else:
            h4["signal_type"] = "trend_exhaustion_block"
        # Flag for position manager: existing positions should tighten stops,
        # block adds, and consider reducing exposure.
        trend_decay_level = "DECAYING"
        exhaustion_reason = reason
    elif h4["direction"] == "bullish" and h4["regime"] == "trending":
        if h1["direction"] == "bullish":
            action = "做多"
            side = "多"
            reason = "4h+1h 趋势一致，动量稳定"
            h4["signal_type"] = "trend_following"
        else:
            # 1h not aligned in a 4h bullish trend = likely pullback opportunity.
            # Only allow pullback entry when macro is not bearish.
            if macro.get("is_bear"):
                action = "谨慎持有"
                side = "多"
                reason = "4h 看涨但 1h 方向不一致，且宏观偏空，谨慎观望"
            else:
                # Bull or neutral macro: treat 1h misalignment as pullback.
                # Require at least one trigger condition:
                # (a) 30m price structure shows higher lows (止跌)
                # (b) 30m entry position in lower 40% (回调到位)
                # (c) Last 30m candle is bullish reversal (close > open, lower shadow > body×2)
                h30_ps = h30.get("price_structure", {})
                h30_entry = h30.get("entry_position", {})
                h30_pct = h30_entry.get("percentile", 50)
                h30_in_low = h30_pct < 40
                h30_higher_lows = h30_ps.get("higher_lows", False)

                h30_bullish_type = h30_ps.get("type") == "bullish"

                fib_entry = h30["entry_position"]["range_low"] + (
                    h30["entry_position"]["range_high"] - h30["entry_position"]["range_low"]
                ) * 0.382

                trigger_ok = h30_in_low or h30_higher_lows or h30_bullish_type

                if trigger_ok:
                    triggers = []
                    if h30_in_low:
                        triggers.append(f"30m位置低位({h30_pct:.0f}%)")
                    if h30_higher_lows:
                        triggers.append("30m低点抬高")
                    if h30_bullish_type:
                        triggers.append("30m结构转多")
                    action = "回调入场（轻仓做多）"
                    side = "多"
                    reason = (
                        f"4h看涨趋势中（ADX={h4['adx']:.0f}），1h短期偏离，"
                        f"{'+'.join(triggers)}，视为回调入场机会，"
                        f"建议限价单 {fib_entry:.0f} 附近入场"
                    )
                    h4["signal_subtype"] = "pullback_entry"
                    h4["signal_type"] = "trend_pullback"
                else:
                    # Pullback not confirmed — wait for price to come down or show reversal
                    action = "谨慎持有（回调未确认）"
                    side = "多"
                    reason = (
                        f"4h看涨趋势中（ADX={h4['adx']:.0f}），1h短期偏离，"
                        f"但30m无止跌信号（位置{h30_pct:.0f}%，无低点抬高），"
                        f"等待回调至 {fib_entry:.0f} 附近或出现止跌结构再入场"
                    )
    elif h4["direction"] == "bearish" and h4["regime"] == "trending":
        if h1["direction"] == "bearish":
            if macro.get("is_bull") or macro.get("is_bull_pullback"):
                # Block short in macro bull or bull pullback — bearish signal in a bull market
                # is likely just a shallow pullback that gets bought up quickly.
                pullback_info = f"（回撤深度 {macro.get('pullback_depth_pct', 0):.1f}%）" if macro.get("is_bull_pullback") else ""
                action = "观望"
                side = "观望"
                reason = f"宏观看涨/回调中，禁止做空 {pullback_info}"
            else:
                action = "做空"
                side = "空"
                reason = "4h+1h 趋势一致看跌"
                h4["signal_type"] = "trend_following"
        else:
            if macro.get("is_bull") or macro.get("is_bull_pullback"):
                action = "观望"
                side = "观望"
                reason = "宏观看涨/回调中，1h方向不一致但禁止做空"
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
        # Trend relay: 4h exhaustion but smaller TFs forming same direction as original trend.
        # This catches "old trend dying, new trend being born" in the same direction —
        # not a reversal, but a continuation via smaller timeframe接力.
        # Checked BEFORE reversal logic — if small TFs confirm the original direction,
        # the reversal assumption is invalid.
        elif ((h1_reg == "forming" and h1["direction"] == orig_h4_direction)
              or (h30.get("regime") == "forming" and h30["direction"] == orig_h4_direction)):
            relay_label = "涨" if orig_h4_direction == "bullish" else "跌"
            action = f"趋势接力（轻仓试{relay_label}）"
            side = "多" if orig_h4_direction == "bullish" else ("空" if orig_h4_direction == "bearish" else "观望")
            reason = (
                f"4h趋势衰竭（ADX={h4['adx']:.0f}）但小周期同向形成中，"
                f"1h={'看涨' if h1['direction']=='bullish' else '看跌' if h1['direction'] else '未确认'}，"
                f"30m={'看涨' if h30['direction']=='bullish' else '看跌' if h30['direction'] else '未确认'}，"
                f"可轻仓试探趋势延续"
            )
        else:
            # Direction was flipped in multi-TF rules → reversal signal.
            # Cooldown already handled in alignment rules: when in cooldown,
            # h4["signal_type"] is NOT set to "exhaustion_reversal", so we
            # naturally fall through to the "trend exhaustion" branch below.
            flipped_dir = h4["direction"]
            is_reversal = h4.get("signal_type") == "exhaustion_reversal"
            h30_confirms = (h30["direction"] == flipped_dir) if h30["direction"] else False
            if h30_confirms and is_reversal:
                action = f"衰竭反转早期信号（轻仓试{'涨' if flipped_dir == 'bullish' else '跌'}）"
                side = "多" if flipped_dir == "bullish" else ("空" if flipped_dir == "bearish" else "观望")
                reason = f"4h趋势衰竭（ADX={h4['adx']:.0f}动量减弱），30m已确认{'看涨' if flipped_dir == 'bullish' else '看跌'}方向，可轻仓试探反转"
            elif is_reversal:
                action = f"衰竭反转待确认（30m未确认）"
                side = "观望"
                reason = f"4h趋势衰竭（ADX={h4['adx']:.0f}），反转方向已确定为{'看涨' if flipped_dir == 'bullish' else '看跌'}，等待30m确认"
            else:
                action = "趋势衰竭"
                side = "观望"
                reason = f"趋势正在衰竭（ADX={h4['adx']:.0f}动量减弱），建议获利了结，等待新趋势确认"
    elif h4["regime"] == "low_vol_trend":
        # Low vol trend needs 1h confirmation
        if h1_reg == "trending" and h1["direction"] == h4["direction"]:
            action = "低波动趋势确认"
            side = "多" if h4["direction"] == "bullish" else ("空" if h4["direction"] == "bearish" else "观望")
            reason = f"低波动趋势延续（ADX={h4['adx']:.0f}），1h趋势确认同向，可入场"
            h4["signal_type"] = "low_vol_trend"
        else:
            action = "低波动趋势未确认"
            side = "观望"
            reason = f"4h低波动趋势但1h{'方向相反' if h1.get('direction') and h1['direction'] != h4['direction'] else '未形成趋势'}，等待确认"
    elif h4.get("signal_type") in ("ranging_breakout", "macro_bias_long") and ranging_breakout:
        # Ranging breakout or macro bias: light position from consolidation
        # Block bearish signals in macro bull
        if ranging_breakout["direction"] == "bearish" and (macro.get("is_bull") or macro.get("is_bull_pullback")):
            action = "观望"
            side = "观望"
            reason = f"宏观看涨趋势中，禁止做空（{ranging_breakout['reason']}）"
        else:
            direction_label = "涨" if ranging_breakout["direction"] == "bullish" else "跌"
            signal_label = "盘整突破" if ranging_breakout.get("signal_subtype") == "ranging_breakout" else "宏观 bias"
            action = f"{signal_label}（轻仓试{direction_label}）"
            side = "多" if ranging_breakout["direction"] == "bullish" else ("空" if ranging_breakout["direction"] == "bearish" else "观望")
            reason = ranging_breakout["reason"]
    elif h4["regime"] == "high_volatility":
        action = "高波动预警"
        side = "观望"
        reason = f"市场高波动（ATR%={h4['atr_pct']:.2f}），建议观望或减仓"
    elif h4["regime"] == "low_volatility":
        action = "低波动蓄力"
        side = "观望"
        reason = f"市场低波动蓄力（ATR%={h4['atr_pct']:.2f}），关注突破方向"
    elif h4["regime"] == "forming":
        # Forming state: normally wait, but if 1h trending with quality filter
        # passed, allow a light entry (async cycle mode)
        if h4_forming_quality:
            direction_label = "涨" if h4["direction"] == "bullish" else "跌"
            action = f"形成态早期信号（轻仓试{direction_label}）"
            side = "多" if h4["direction"] == "bullish" else ("空" if h4["direction"] == "bearish" else "观望")
            reason = f"4h趋势正在形成（{'看涨' if h4['direction'] == 'bullish' else '看跌'}），ADX={h4['adx']:.0f}，1h趋势已确认同向，可轻仓试探"
            h4["signal_type"] = "trend_forming_early"  # forming signals, not trend_following
        else:
            action = "等待确认"
            side = "观望"  # 形成态不开仓
            reason = f"趋势正在形成（{'看涨' if h4['direction'] == 'bullish' else '看跌'}），ADX={h4['adx']:.0f}未达阈值，暂不开仓"
    else:
        action = "观望"
        side = "观望"
        reason = "市场处于盘整，等待突破"

    # ─── Decay Entry Block ──────────────────────────────────────────────────
    # Trend is weakening — do NOT open new positions in the decaying direction.
    # Existing positions are handled by the position manager (decay reduction logic).
    # New entries during decay are logically inconsistent: if the trend is ending,
    # entering in the same direction bets against our own decay signal.
    if trend_decay_level == "DECAYING" and side != "观望":
        direction_label = "涨" if h4["direction"] == "bullish" else "跌"
        action = f"观望（{direction_label}趋势衰减中）"
        side = "观望"
        reason = exhaustion_reason or f"{direction_label}趋势正在衰减，等待方向确认"

    # ─── Stage 1 stop/target override ───
    # Stage1 signals use 30m ATR for stop (signal timeframe matches risk).
    # Target ensures RR ≥ 1.2 — either 4h conserv target or stop×1.2.
    _stage1_stop = None
    _stage1_target = None

    # ─── Stage 1 Fallback: when main signal says 观望 but Stage 1 fires ───
    # This makes the two-stage system actionable: even if multi-TF alignment
    # says wait, a Stage 1 warning can trigger a light entry (10-15%).
    # But NOT during DECAYING/EXHAUSTED — trend decay overrides Stage 1.
    # And NOT in high_volatility/exhaustion — these regimes have too much noise.
    # And Stage 1 direction must align with macro — no fighting the macro trend.
    _stage1_regime_ok = h4["regime"] not in ("high_volatility", "exhaustion", "low_volatility")
    _stage1_dir = stage_1_warning["direction"] if stage_1_warning else None
    _stage1_macro_ok = True
    if _stage1_dir:
        if _stage1_dir == "bearish" and (macro.get("is_bull") or macro.get("is_bull_pullback")):
            _stage1_macro_ok = False
        if _stage1_dir == "bullish" and (macro.get("is_bear") or macro.get("is_bear_pullback")):
            _stage1_macro_ok = False

    if side == "观望" and trend_decay_level == "NONE" and stage_1_warning and _stage1_regime_ok and _stage1_macro_ok:
        s1 = stage_1_warning
        direction_label = "涨" if s1["direction"] == "bullish" else "跌"
        action = f"Stage1预警（轻仓试{direction_label}）"
        side = "多" if s1["direction"] == "bullish" else ("空" if s1["direction"] == "bearish" else "观望")
        reason = s1["reason"]
        # Tag signal type for evolution tracking
        if not h4.get("signal_type"):
            h4["signal_type"] = f"stage1_{s1['type']}"
        # Stage1 stop: use 30m ATR but floored by 1h ATR — avoids stops
        # that are too tight for the 4h/1h structure being traded.
        # If stage_1_warning provides a pre-computed stop (e.g., P3 breakout
        # with structure-based stop), use it directly.
        if "stop" in s1:
            _stage1_stop = s1["stop"]
            _stage1_target = round(current_price + abs(current_price - s1["stop"]) * 1.5, 1) if side == "多" else round(current_price - abs(current_price - s1["stop"]) * 1.5, 1)
        else:
            s1_stop_dist = max(h30["atr"] * 1.5, h1["atr"] * 0.8)
            s1_target_dist = s1_stop_dist * 1.2  # force RR ≥ 1.2
            if side == "多":
                _stage1_stop = round(current_price - s1_stop_dist, 1)
                _stage1_target = round(current_price + s1_target_dist, 1)
            elif side == "空":
                _stage1_stop = round(current_price + s1_stop_dist, 1)
                _stage1_target = round(current_price - s1_target_dist, 1)

    # ─── Aggressive Signals: when everything says 观望, check 5 light strategies ───
    # Initialize aggressive override vars early (used later in order signal generation)
    _aggressive_stop = None
    _aggressive_target = None
    _aggressive_entry_type = None
    _aggressive_position_pct = None
    _aggressive_entry_note = None
    _aggressive_leverage = None
    _funding_rate_decay = False  # set by funding rate filter if rate is in decay exemption zone

    # Aggressive strategies: light position, ONLY in ranging/low_volatility regimes.
    # NOT during DECAYING/EXHAUSTED — trend decay overrides all entries.
    # Regime whitelist: only allow aggressive signals when market is consolidating
    # with enough directional energy (ADX > 15). Block in high_volatility,
    # exhaustion, forming — these regimes have too much noise or false breakouts.
    _aggressive_regime_ok = h4["regime"] in ("ranging", "low_volatility") and h4["adx"] > 15

    if side == "观望" and trend_decay_level == "NONE" and _aggressive_regime_ok:
        aggressive = _check_aggressive_signals(
            h4, h1, h30, all_candles_4h, all_thresholds, market_ctx, macro,
            pre_fetched_candles=pre_fetched_klines,
        )
        if aggressive:
            action = aggressive["action"]
            side = aggressive["side"]
            reason = aggressive["reason"]
            h4["signal_type"] = aggressive["signal_type"]
            confidence = aggressive.get("confidence", 35)
            # Store for downstream order signal generation
            _aggressive_entry_type = aggressive.get("entry_type", "market")
            _aggressive_position_pct = aggressive.get("position_pct", 10)
            _aggressive_stop = aggressive.get("stop")
            _aggressive_target = aggressive.get("target")
            _aggressive_entry_note = aggressive.get("entry_note", "")
            _aggressive_leverage = aggressive.get("leverage")

    # Funding rate filter (adaptive, not all-or-nothing)
    # Funding rates on BTC are settled every 8h. Around settlement time, rates
    # can briefly spike above the block threshold due to arbitrage activity —
    # this is transient noise, not a sustained high-cost environment.
    #
    # Strategy: use a "decay exemption" zone. If funding rate is above the
    # block threshold but within 50% above it, allow opening with reduced
    # leverage instead of full block. Only fully block when rate is severely
    # elevated (>1.5x the block threshold).
    oi_warning = False
    oi_divergence = False
    funding_rate_pct = funding_rate * 100  # convert to percentage

    block_thr = settings.funding_rate_block_threshold
    warn_thr = settings.funding_rate_warn_threshold
    exemption_thr = block_thr * 1.5  # decay exemption ceiling

    if funding_rate > exemption_thr:
        # Severely elevated — fully block long
        if side == "多":
            side = "观望"
            action = "观望"
            h4["direction"] = None
            reason = f"资金费率极高 ({funding_rate_pct:.3f}%)，做多风险过大"
    elif funding_rate > block_thr:
        # Decay exemption zone: allow long but with leverage reduction
        # Mark it so the order_signal can apply reduced leverage downstream
        _funding_rate_decay = True
    elif funding_rate > warn_thr:
        pass
    elif funding_rate < -exemption_thr:
        # Severely elevated negative — fully block short
        if side == "空":
            side = "观望"
            action = "观望"
            h4["direction"] = None
            reason = f"资金费率极低 ({funding_rate_pct:.3f}%)，做空风险过大"
    elif funding_rate < -block_thr:
        # Decay exemption zone: allow short but with leverage reduction
        _funding_rate_decay = True
    elif funding_rate < -warn_thr:
        pass

    # OI divergence: OI declining + price moving in signal direction
    # Adds price direction check to distinguish normal profit-taking from
    # genuine divergence (OI down + price going against the signal).
    oi_divergence = False
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

    # Aggressive signal overrides: use pre-computed stop/target/entry
    # Default stop/target from forecast (overridden below for aggressive signals)
    target_raw = h4["forecast"]["target_conserv"]
    stop_distance = h4["forecast"].get("stop_distance", h4["atr"] * 1.5)
    # Minimum stop distance floor: 0.15% of price (~$117 at $78k) prevents
    # zero/tiny stops when ATR collapses during extreme low volatility
    min_stop_distance = current_price * 0.0015
    stop_distance = max(stop_distance, min_stop_distance)
    # target_conserv is already direction-aware (below current in downtrend,
    # above in uptrend), so use it directly for both long and short.
    target = target_raw
    stop = (
        round(current_price - stop_distance, 1)
        if side == "多"
        else round(current_price + stop_distance, 1)
    )

    # Apply stop/target overrides (mutually exclusive in practice:
    # Stage 1 and aggressive signals never both trigger because Stage 1
    # sets side first, blocking aggressive. But order matters for safety.)
    # Priority: Stage 1 > Aggressive > default forecast
    if _stage1_stop is not None:
        stop = _stage1_stop
    elif _aggressive_stop is not None:
        stop = _aggressive_stop
    if _stage1_target is not None:
        target = _stage1_target
    elif _aggressive_target is not None:
        target = _aggressive_target

    # Final guard: ensure stop maintains minimum distance from entry price
    # (catches aggressive/stage1 overrides that could produce stops too close)
    stop_distance_actual = abs(stop - current_price)
    if stop_distance_actual < min_stop_distance:
        if side == "多":
            stop = round(current_price - min_stop_distance, 1)
        else:
            stop = round(current_price + min_stop_distance, 1)

    # Entry price: based on timing and direction

    # Initialize current_rr for safety check
    current_rr = None

    # Entry timing: align with trend direction
    entry_pct = h30["entry_position"]["percentile"]
    range_narrow = h30["entry_position"].get("range_too_narrow", False)
    range_size = h30["entry_position"].get("range_size", 0)
    # Range too wide (> 2% of price): percentile "high/low" is meaningless
    # in a wide swing — a 60% position in a $2000 range is still $1200 from top.
    range_too_wide = range_size > current_price * 0.02

    if range_narrow:
        # Range too narrow — percentile unreliable, always use limit at mid-range
        if side == "多":
            timing = "good"
            timing_label = "区间过窄（中位限价）"
            timing_reason = f"30m区间仅 {range_size:.0f} 点，百分位不可靠，用区间中位价挂限价单"
            entry_action = "limit"
        elif side == "空":
            timing = "good"
            timing_label = "区间过窄（中位限价）"
            timing_reason = f"30m区间仅 {range_size:.0f} 点，百分位不可靠，用区间中位价挂限价单"
            entry_action = "limit"
        else:
            timing = "good"
            timing_label = "观望中"
            timing_reason = "当前无交易信号"
            entry_action = "wait"
    elif range_too_wide:
        # Range too wide — percentile "high/low" labels are meaningless
        # in a wide swing. Use limit at mid-range to avoid chasing.
        if side == "多":
            timing = "good"
            timing_label = "区间过宽（中位限价）"
            timing_reason = f"30m区间达 {range_size:.0f} 点（>{current_price*0.02:.0f}），百分位追高/追低不可靠，用区间中位价挂限价单"
            entry_action = "limit"
        elif side == "空":
            timing = "good"
            timing_label = "区间过宽（中位限价）"
            timing_reason = f"30m区间达 {range_size:.0f} 点（>{current_price*0.02:.0f}），百分位追高/追低不可靠，用区间中位价挂限价单"
            entry_action = "limit"
        else:
            timing = "good"
            timing_label = "观望中"
            timing_reason = "当前无交易信号"
            entry_action = "wait"
    elif side == "多":
        if entry_pct > 60:
            timing = "high"
            timing_label = "接近区间高位"
            timing_reason = f"价格在区间 {entry_pct:.0f}% 位置，追高风险，建议等待回调"
            entry_action = "limit"
        elif entry_pct < 30:
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
        if entry_pct < 40:
            timing = "low"
            timing_label = "接近区间低位"
            timing_reason = f"价格在区间 {entry_pct:.0f}% 位置，追空风险，建议等待反弹"
            entry_action = "limit"
        elif entry_pct > 70:
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

    # Aggressive signals: only override timing if side is still actionable
    if _aggressive_entry_type is not None and side != "观望":
        entry_action = _aggressive_entry_type
        timing = "good"
        timing_label = "激进信号"
        timing_reason = _aggressive_entry_note or "激进信号，市价入场"

    # ─── Position State Machine ───
    pos_state = _analyze_position_state(h4, h1, h30, history_rows or [], position, all_thresholds)

    # Track whether trend is fresh (not yet established across 3 consecutive cycles).
    # Used by position manager to block re-entry on established trends.
    _trend_fresh = (
        _is_fresh_trend(history_rows or [])
        if h4.get("regime") == "trending"
        else True
    )

    # 具体下单信号
    atr4 = h4["atr"]
    atr1 = h1["atr"]
    atr30 = h30["atr"]

    # Safety: if side ended up as "观望", clear funding decay flag to prevent
    # any stale state from leaking into downstream logic.
    if side == "观望":
        _funding_rate_decay = False

    # Fixed leverage: 20x for all signals
    base_leverage = 20
    leverage_str = f"{int(base_leverage)}x"

    # Entry price: based on timing and direction
    if side == "观望":
        entry_price = None
        entry_note = reason or "趋势不明确，建议观望"
        target = None
        stop = None
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
        # Price past target: signal opportunity already gone — check FIRST
        _price_past_target_long = target and current_price >= target * 0.999
        if _price_past_target_long:
            side = "观望"
            entry_action = "wait"
            entry_note = f"价格已涨至目标 {target:.0f} 以上，做多机会已错过"
        # Stage 1 signals: if market moved past limit entry, switch to market price
        elif stage_1_warning and current_rr is not None and current_rr < 1.0:
            entry_price = round(current_price, 1)
            entry_action = "market"
            entry_note = "Stage1信号，现价市价做多（限价位已过）"
        # R/R uses actual planned entry_price (not current_price)
        risk = abs(entry_price - stop)
        reward = abs(target - entry_price)
        rr = round(reward / max(risk, 1), 2)
        # Safety: if current RR < 1.0, market moved too far — force wait
        # But Stage 1 signals are exploratory — switch to market entry instead of blocking.
        if current_rr is not None and current_rr < 1.0 and not stage_1_warning:
            side = "观望"
            entry_action = "wait"
            entry_note = f"现价盈亏比过低（{current_rr:.2f}），价格已偏离入场位，等待回调至 {entry_price:.0f} 附近再入场（目标 {target:.0f}，止损 {stop:.0f}）"
            base_pct = round(max(1.0, 2.0 / max(rr, 1)), 1)
            leverage_num = int(base_leverage)
            # Funding rate decay exemption: reduce leverage when funding is
            # elevated but not severely enough to fully block.
            if _funding_rate_decay:
                leverage_num = max(5, leverage_num // 2)  # halve leverage, floor 5x
            risk_price_pct = risk / max(entry_price, 1) * 100
            max_safe_pct = 2.0 / max(leverage_num * risk_price_pct / 100, 0.01)
            position_pct = round(min(base_pct, max_safe_pct), 1)
            order_signal = {
                "side": "观望",
                "entry_type": entry_action,
                "action": pos_state.get("action", ""),
                "entry_price": entry_price,
                "planned_entry_price": entry_price,
                "entry_note": entry_note,
                "target": target,
                "stop": stop,
                "risk": risk,
                "reward": reward,
                "rr_ratio": rr,
                "current_rr": current_rr,
                "position_pct": position_pct,
                "leverage": leverage_str,
                "decay": False,
                "confidence": _order_confidence(rr, confidence),
            }
        else:
            base_pct = round(max(1.0, 2.0 / max(rr, 1)), 1)
            leverage_num = int(base_leverage)
            # Funding rate decay exemption: reduce leverage when funding is
            # elevated but not severely enough to fully block.
            if _funding_rate_decay:
                leverage_num = max(5, leverage_num // 2)  # halve leverage, floor 5x
            risk_price_pct = risk / max(entry_price, 1) * 100
            max_safe_pct = 2.0 / max(leverage_num * risk_price_pct / 100, 0.01)
            position_pct = round(min(base_pct, max_safe_pct), 1)
            order_signal = {
                "side": "做多",
                "entry_type": entry_action,
                "action": pos_state.get("action", ""),
                "entry_price": entry_price,
                "planned_entry_price": entry_price,
                "entry_note": entry_note,
                "target": target,
                "stop": stop,
                "risk": risk,
                "reward": reward,
                "rr_ratio": rr,
                "current_rr": current_rr,
                "position_pct": position_pct,
                "leverage": leverage_str,
                "decay": False,
                "confidence": _order_confidence(rr, confidence),
            }
    elif side == "空":
        # Price past target: signal opportunity already gone — check FIRST
        _price_past_target_short = target and current_price <= target * 1.001
        if _price_past_target_short:
            side = "观望"
            entry_action = "wait"
            entry_note = f"价格已跌至目标 {target:.0f} 以下，做空机会已错过"
        # Stage 1 signals: if market moved past limit entry, switch to market price
        elif stage_1_warning and current_rr is not None and current_rr < 1.0:
            entry_price = round(current_price, 1)
            entry_action = "market"
            entry_note = "Stage1信号，现价市价做空（限价位已过）"
        risk = abs(entry_price - stop)
        reward = abs(target - entry_price)
        rr = round(reward / max(risk, 1), 2)
        # Safety: if current RR < 1.0, market moved too far — force wait
        # But Stage 1 signals are exploratory — switch to market entry instead of blocking.
        if current_rr is not None and current_rr < 1.0 and not stage_1_warning:
            side = "观望"
            entry_action = "wait"
            entry_note = f"现价盈亏比过低（{current_rr:.2f}），价格已偏离入场位，等待反弹至 {entry_price:.0f} 附近再入场（目标 {target:.0f}，止损 {stop:.0f}）"
            base_pct = round(max(1.0, 2.0 / max(rr, 1)), 1)
            leverage_num = int(base_leverage)
            # Funding rate decay exemption: reduce leverage when funding is
            # elevated but not severely enough to fully block.
            if _funding_rate_decay:
                leverage_num = max(5, leverage_num // 2)  # halve leverage, floor 5x
            risk_price_pct = risk / max(entry_price, 1) * 100
            max_safe_pct = 2.0 / max(leverage_num * risk_price_pct / 100, 0.01)
            position_pct = round(min(base_pct, max_safe_pct), 1)
            order_signal = {
                "side": "观望",
                "entry_type": entry_action,
                "action": pos_state.get("action", ""),
                "entry_price": entry_price,
                "planned_entry_price": entry_price,
                "entry_note": entry_note,
                "target": target,
                "stop": stop,
                "risk": risk,
                "reward": reward,
                "rr_ratio": rr,
                "current_rr": current_rr,
                "position_pct": position_pct,
                "leverage": leverage_str,
                "decay": False,
                "confidence": _order_confidence(rr, confidence),
            }
        else:
            base_pct = round(max(1.0, 2.0 / max(rr, 1)), 1)
            leverage_num = int(base_leverage)
            # Funding rate decay exemption: reduce leverage when funding is
            # elevated but not severely enough to fully block.
            if _funding_rate_decay:
                leverage_num = max(5, leverage_num // 2)  # halve leverage, floor 5x
            risk_price_pct = risk / max(entry_price, 1) * 100
            max_safe_pct = 2.0 / max(leverage_num * risk_price_pct / 100, 0.01)
            position_pct = round(min(base_pct, max_safe_pct), 1)
            order_signal = {
                "side": "做空",
                "entry_type": entry_action,
                "action": pos_state.get("action", ""),
                "entry_price": entry_price,
                "planned_entry_price": entry_price,
                "entry_note": entry_note,
                "target": target,
                "stop": stop,
                "risk": risk,
                "reward": reward,
                "rr_ratio": rr,
                "current_rr": current_rr,
                "position_pct": position_pct,
                "leverage": leverage_str,
                "decay": False,
                "confidence": _order_confidence(rr, confidence),
            }
    else:
        order_signal = {
            "side": "观望",
            "entry_price": None,
            "planned_entry_price": None,
            "entry_note": entry_note,
            "target": None,
            "stop": None,
            "risk": None,
            "reward": None,
            "rr_ratio": None,
            "position_pct": None,
            "leverage": None,
            "decay": False,
            "confidence": "低",
        }

    # ─── Price Past Target: market already moved, signal is stale ───
    # Catches cases where order_signal side was set to "做多"/"做空" but
    # price has already moved past the target (e.g., forming state direct assignment).
    if order_signal.get("side") in ("做多", "做空"):
        _buffer = 0.001  # 0.1% buffer for price noise
        if order_signal["side"] == "做空" and current_price <= target * (1 + _buffer):
            order_signal = {
                "side": "观望",
                "entry_price": None,
                "planned_entry_price": None,
                "entry_note": f"价格已跌至目标 {target:.0f} 附近，做空机会已错过",
                "target": None, "stop": None,
                "risk": None, "reward": None, "rr_ratio": None,
                "position_pct": None, "leverage": None, "decay": False,
                "confidence": "低",
            }
            action = "观望"
            reason = "价格已到达目标位，做空机会已错过"
        elif order_signal["side"] == "做多" and current_price >= target * (1 - _buffer):
            order_signal = {
                "side": "观望",
                "entry_price": None,
                "planned_entry_price": None,
                "entry_note": f"价格已涨至目标 {target:.0f} 附近，做多机会已错过",
                "target": None, "stop": None,
                "risk": None, "reward": None, "rr_ratio": None,
                "position_pct": None, "leverage": None, "decay": False,
                "confidence": "低",
            }
            action = "观望"
            reason = "价格已到达目标位，做多机会已错过"

    # ─── Macro Bias Long: very light position (5-10%) ───
    if h4.get("signal_type") == "macro_bias_long" and order_signal.get("position_pct"):
        order_signal["position_pct"] = min(order_signal["position_pct"], 10)
        order_signal["entry_note"] += "（宏观看涨轻仓试多，上限10%）"
        order_signal["leverage"] = "20x"

    # ─── Pullback Entry: light position (max 20%) ───
    if h4.get("signal_subtype") == "pullback_entry" and order_signal.get("position_pct"):
        order_signal["position_pct"] = min(order_signal["position_pct"], 20)
        order_signal["entry_note"] += "（回调入场轻仓，上限20%）"
        order_signal["leverage"] = "20x"

    # ─── Ranging Breakout: reduce position size to 25% of normal ───
    if h4.get("signal_type") == "ranging_breakout" and order_signal.get("position_pct"):
        order_signal["position_pct"] = round(order_signal["position_pct"] * 0.25, 1)
        order_signal["entry_note"] += "（盘整突破轻仓，正常仓位的25%）"
        order_signal["leverage"] = "20x"

    # ─── Stage 1 Warning: use fixed small position (10-15%) ───
    if stage_1_warning and order_signal.get("position_pct"):
        target_pct = stage_1_warning.get("position_pct_hint", 10)
        # Only adjust if current position is larger than Stage 1 suggestion
        if order_signal["position_pct"] > target_pct:
            order_signal["position_pct"] = target_pct
            order_signal["entry_note"] += f"（Stage1预警轻仓，建议{target_pct}%）"
        order_signal["leverage"] = "20x"

    # ─── Aggressive Signals: use pre-defined position size ───
    if _aggressive_position_pct is not None and order_signal.get("position_pct"):
        order_signal["position_pct"] = _aggressive_position_pct
        if _aggressive_entry_note:
            order_signal["entry_note"] = _aggressive_entry_note
        if _aggressive_leverage is not None:
            order_signal["leverage"] = f"{_aggressive_leverage}x"

    # ─── OI Divergence: halve position size ───
    if oi_divergence and order_signal.get("position_pct"):
        order_signal["position_pct"] = round(order_signal["position_pct"] * 0.5, 1)
        order_signal["entry_note"] += "（OI背离，仓位减半）"

    if h4["direction"] == "bullish":
        trend_label = "上涨"
    elif h4["direction"] == "bearish":
        trend_label = "下跌"
    else:
        trend_label = "无明确"
    summary = f"{trend_label}趋势（{h4['strength']}），已持续约 {h4['duration_hours']} 小时"

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

    # Ensure h4 has a proper signal_type — prevent default trend_following leak.
    # Must run BEFORE the safety loop below, since h4 == results["4h"].
    if not h4.get("signal_type") or h4["signal_type"] == "none":
        if h4["regime"] == "ranging" or h4["direction"] is None:
            h4["signal_type"] = "none"
        elif h4["regime"] == "forming":
            h4["signal_type"] = "trend_forming_early"
        elif h4["regime"] == "trending":
            h4["signal_type"] = "trend_following"
        elif h4["regime"] == "low_vol_trend":
            h4["signal_type"] = "low_vol_trend"
        elif h4["regime"] == "exhaustion":
            h4["signal_type"] = "trend_exhaustion"
        elif h4["regime"] == "high_volatility":
            h4["signal_type"] = "none"
        else:
            h4["signal_type"] = "none"

    # Safety: ensure all timeframes have signal_type (never None or missing)
    for tf in results:
        if "signal_type" not in results[tf] or results[tf]["signal_type"] is None:
            results[tf]["signal_type"] = "none"

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
                    "action": "做多" if side == "多" else ("谨慎持有" if side == "观望" else "建议减仓或离场"),
                    "reason": reason if side == "多" else ("观望等待方向确认" if side == "观望" else f"当前信号偏空（{reason}），多单建议减仓或设置 tighter 止损"),
                    "target": target if side == "多" else None,
                    "stop": stop if side == "多" else None,
                },
                "short_advice": {
                    "action": "做空" if side == "空" else ("谨慎持有" if side == "观望" else "建议减仓或离场"),
                    "reason": reason if side == "空" else ("观望等待方向确认" if side == "观望" else f"当前信号偏多（{reason}），空单建议减仓或设置 tighter 止损"),
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
                "open_interest_prev": market_ctx.get("open_interest_prev", 0),
                "oi_warning": oi_warning,
                "oi_divergence": oi_divergence,
                "circuit_breaker": circuit_breaker_reason,
                "vol_percentile_4h": h4.get("vol_percentile"),
                "vol_threshold": vol_threshold_info,
                "alignment_rule": alignment_rule,
                "trend_fresh": _trend_fresh,
                "suggested_leverage": f"{int(base_leverage)}x",
                "convergence_warning": convergence_warning,
                "ranging_breakout": ranging_breakout,
                "stage_1_warning": stage_1_warning,
                "aggressive_signal": h4.get("signal_type", "") if h4.get("signal_type", "").startswith("aggressive_") else None,
                "macro_context": macro,
                "trend_exhausted": {
                    "detected": trend_decay_level != "NONE" and not is_reversal_signal,
                    "level": trend_decay_level if trend_decay_level != "NONE" else None,
                    "direction": orig_h4_direction if (trend_decay_level != "NONE" and not is_reversal_signal) else None,
                    "reason": exhaustion_reason if (trend_decay_level != "NONE" and not is_reversal_signal) else None,
                },
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
                    "ema20": results[tf]["ema20"],
                    "atr_pct": results[tf].get("atr_pct"),
                    "di_spread": results[tf].get("di_spread"),
                    "di_spread_history": results[tf].get("di_spread_history", []),
                    "di_spread_converging": results[tf].get("di_spread_converging", False),
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

"""Simulated position manager: auto-open and auto-manage simulated positions.

Priority chain (strict):
  1. Stop loss (price hits stop)
  2. Take profit (price hits target)
  3. PnL trailing stop (dual strategy: <30% lock 50% profit, >=30% fixed 10% pullback)
     After trailing stop moves, immediate price check to prevent one-cycle delay
  4. Emergency exit (high volatility + opposite direction, ADX dead)
     ADX dead: absolute (<20 and drop≥5) OR relative (drop ≥30% from entry)
  4.5. Circuit breaker (flash crash, liquidation cascade, tighten stop if in profit)
  5. Signal reversal (verdict direction flipped against position)
  6. Trend exhaustion (verdict detects trend ending, immediate exit; skipped for exhaustion_reversal)
  7. ADX trailing stop (max_adx >= 30, current <= max_adx - max(8, max_adx*0.15); adaptive: 15/5 for aggressive)
  7.5. Max hold time (slow=120h, default=72h, fast=36h; exit when signal weakens + unprofitable; ≥5% PnL exempt)
  8. Exit signal (engine level='exit')
  9. Reduce signal (engine level='reduce', max 2 times -> then auto-exit)
      After reduce: stop remains at original level (not tightened)
      Reduce cooldown: half signal interval (faster response to escalating risk)
  10. Add signal (engine level='add', with trailing stop update)
  11. Hold (update max_adx)

Signal-type-aware behavior:
  - RR ratio: trend_following/pullback/low_vol_trend/exhaustion_reversal require 1.5;
    stage1/macro/ranging require 1.0
  - Auto-open blocked: convergence_warning, stage1_* (low-conviction warnings)
  - ADX trailing: aggressive_adx7_early uses threshold 15/drop 5 (vs 30/8 default)
  - PnL trailing grace: fast strategies 5 min, default 15 min
  - Max hold time: slow (low_vol_trend, macro_bias_long) 96h, default 48h,
    fast (breakout, reversal, aggressive) 24h
  - Trend exhaustion exit: skipped for exhaustion_reversal positions

Features:
- Multiple reduce (max 2), auto-exit on 3rd reduce
  - After reduce: stop tightened 50% toward entry (normal) or entry ±0.2% (DECAYING)
  - Reduce cooldown: half signal interval (faster response to escalating risk)
- Signal cooldown: after close, wait 1 signal cycle before re-entry (same direction);
  wait 2 cycles for opposite direction re-entry
- Decay protection: when signal engine detects DECAYING trend, existing positions
  reduce 50%; stop tightened to entry ±0.2% (near breakeven)
- PnL trailing stop: dual strategy — PnL < 30% locks 50% profit, PnL >= 30% uses fixed 10% pullback (grace: 5-15 min);
  immediate price check after trailing prevents one-cycle delay
- Dynamic leverage: uses signal engine's order_signal.leverage (funding-rate adjusted),
  fallback: trending=20x, breakout=10x, forming=10x, exhaustion=10x, low_vol_trend=10x
- Pyramid add-on: triggered by engine 'add' signal, max 3 adds
  - add_count is lifetime counter, NOT reset on reduce (prevents add-reduce cycling)
- ADX trailing stop: exit when ADX drops >= 8 from peak (peak >= 30)
- Dynamic stop loss: ATR × k_vol × k_adx from signal engine
"""

import time

from app.config import settings
from app.evolution import get_active_thresholds

# Hook registries for auto-trader callbacks
# Each hook: fn(conn, position_dict, ...) — set by auto_trader.register_callbacks()
_auto_open_hooks = []
_auto_close_hooks = []
_auto_reduce_hooks = []
_auto_add_hooks = []


def _register_auto_hooks(open_hooks, close_hooks, reduce_hooks, add_hooks):
    """Internal: called by auto_trader.register_callbacks()."""
    global _auto_open_hooks, _auto_close_hooks, _auto_reduce_hooks, _auto_add_hooks
    _auto_open_hooks = open_hooks or []
    _auto_close_hooks = close_hooks or []
    _auto_reduce_hooks = reduce_hooks or []
    _auto_add_hooks = add_hooks or []


REOPEN_COOLDOWN_SECONDS = max(settings.signal_interval_seconds, 300)  # min 5 min after any close before re-entry
OPPOSITE_DIRECTION_COOLDOWN = 2 * REOPEN_COOLDOWN_SECONDS  # extra cooldown for opposite direction
MAX_REDUCE_COUNT = 2
MAX_ADD_COUNT = 3  # max pyramid adds


def calculate_pnl(position: dict, close_price: float) -> float:
    """Calculate PnL in USD: BTC_quantity × price_change."""
    entry = position["entry_price"]
    position_btc = position.get("position_size")  # BTC quantity
    if position_btc:
        if position["side"] == "long":
            return position_btc * (close_price - entry)
        else:
            return position_btc * (entry - close_price)
    # Fallback: legacy percentage
    leverage = position.get("leverage") or 5
    if position["side"] == "long":
        return (close_price - entry) / max(entry, 1) * 100 * leverage
    else:
        return (entry - close_price) / max(entry, 1) * 100 * leverage


def close_simulated_position(conn, position: dict, close_price: float, reason: str):
    """Close a simulated position and record the reason and PnL.

    Uses atomic UPDATE WHERE status='open' to prevent race conditions
    between the monitoring thread and main signal cycle.
    """
    # Guard: skip if already closed (double-close protection)
    if position.get("status") != "open":
        return

    pnl = calculate_pnl(position, close_price)
    now = time.time()
    cursor = conn.execute(
        "UPDATE positions SET status='closed', pnl=?, close_reason=?, "
        "close_price=?, closed_at=?, updated_at=? WHERE id=? AND status='open'",
        (round(pnl, 2), reason, round(close_price, 1), now, now, position["id"]),
    )
    if cursor.rowcount == 0:
        return  # Already closed by another thread (race condition)
    conn.commit()

    # Also update in-memory dict for callers that track status
    position["status"] = "closed"
    position["close_price"] = close_price
    position["close_reason"] = reason
    position["pnl"] = round(pnl, 2)
    position["closed_at"] = now

    # Fire close hooks (auto-trader mirror)
    for hook in _auto_close_hooks:
        try:
            hook(conn, position, close_price, reason)
        except Exception:
            pass  # hook failure must not break position management


def _check_pnl_trailing_stop(conn, sim_pos: dict, current_price: float) -> bool:
    """PnL-based trailing stop with dual strategy:

    Leveraged PnL < 30% → proportional lock (锁 50% profit)
    Leveraged PnL >= 30% → fixed 10% pullback (从最高盈利回撤10%止损)

    Examples at 20x leverage:
      PnL 10% → stop at 5% profit  (price 0.25%)
      PnL 20% → stop at 10% profit (price 0.50%)
      PnL 40% → stop at 30% profit (price 1.50%) via fixed pullback
    """
    entry = sim_pos.get("entry_price")
    stop = sim_pos.get("stop")
    side = sim_pos.get("side")
    leverage = sim_pos.get("leverage") or 20
    if not entry or not stop:
        return False

    # Grace period: newly opened positions are exempt from PnL trailing
    # to avoid premature stops on initial price noise.
    # Fast strategies (breakout, reversal, aggressive) get shorter grace.
    # EXCEPTION: if PnL already >= 5%, skip grace — significant profit
    # means the trade direction is confirmed, trailing should activate.
    created_at = sim_pos.get("created_at") or 0
    signal_type = sim_pos.get("signal_type") or ""

    price_move = (current_price - entry) if side == "long" else (entry - current_price)
    pnl_pct = price_move / entry * 100 * leverage

    if created_at > 0 and pnl_pct < 5:
        if signal_type in ("ranging_breakout", "exhaustion_reversal", "aggressive_adx7_early"):
            grace_seconds = 300  # 5 min for fast strategies
        else:
            grace_seconds = 900  # 15 min default
        if (time.time() - created_at) < grace_seconds:
            return False

    if pnl_pct < 5:
        return False

    # Dual trailing strategy:
    # PnL < 30%: proportional lock — trail = pnl_pct * 0.5
    # PnL >= 30%: fixed pullback — trail = pnl_pct - 10 (locks 10% pullback room)
    # NOTE: at the transition (29%→30%), the fixed pullback formula would
    # produce a wider stop (20% vs 14.5%). The "favorable direction only" guard
    # below prevents stop from moving backward, so stop stays at the tighter
    # level until PnL grows enough that the new stop is tighter than current.
    if pnl_pct >= 30:
        # Fixed 10% pullback: stop locks at (pnl_pct - 10%) profit level
        # At 30% pnl → stop at 20%; at 40% pnl → stop at 30%; at 60% pnl → stop at 50%
        locked_pnl_pct = pnl_pct - 10
    else:
        # Proportional lock: keep 50% of profit
        locked_pnl_pct = pnl_pct * 0.5

    # Convert locked PnL % to price distance from entry
    trail_pct = max(locked_pnl_pct / leverage / 100, 0.0015)

    # PnL trailing stop: both directions move stop toward the profit side.
    #   LONG:  stop moves UP to entry + trail (above entry, locks profit on pullback)
    #   SHORT: stop moves DOWN to entry - trail (below entry, locks profit on bounce)
    if side == "long":
        new_stop = round(entry * (1 + trail_pct), 1)
    else:
        new_stop = round(entry * (1 - trail_pct), 1)

    # Only move stop in favorable direction (closer to locked profit)
    if side == "long" and new_stop <= stop:
        return False
    if side == "short" and new_stop >= stop:
        return False

    conn.execute(
        "UPDATE positions SET stop=? WHERE id=?",
        (new_stop, sim_pos["id"]),
    )
    conn.commit()

    # Immediate check: if current price already crosses the new stop,
    # trigger close now rather than waiting for next signal cycle.
    # This prevents a one-cycle delay window after the stop is tightened.
    if side == "long" and current_price <= new_stop:
        close_simulated_position(conn, sim_pos, current_price, "stop_loss")
        return True
    if side == "short" and current_price >= new_stop:
        close_simulated_position(conn, sim_pos, current_price, "stop_loss")
        return True

    return True


def _check_price_based_exit(conn, sim_pos: dict, current_price: float) -> bool:
    """Priority 1-2: Check if stop-loss or take-profit has been hit."""
    stop = sim_pos.get("stop")
    target = sim_pos.get("target")
    side = sim_pos["side"]

    if stop is not None:
        if side == "long" and current_price <= stop:
            close_simulated_position(conn, sim_pos, current_price, "stop_loss")
            return True
        if side == "short" and current_price >= stop:
            close_simulated_position(conn, sim_pos, current_price, "stop_loss")
            return True

    if target is not None:
        if side == "long" and current_price >= target:
            close_simulated_position(conn, sim_pos, current_price, "take_profit")
            return True
        if side == "short" and current_price <= target:
            close_simulated_position(conn, sim_pos, current_price, "take_profit")
            return True

    return False


def _check_emergency_exit(conn, sim_pos: dict, verdict: dict, current_price: float) -> bool:
    """Priority 3: Emergency exit - high volatility + opposite or ADX dead."""
    side = sim_pos["side"]
    regime = verdict.get("regime", "")
    direction = verdict.get("direction")
    adx_4h = verdict.get("timeframes", {}).get("4h", {}).get("adx", 0)

    if regime == "high_volatility" and direction is not None:
        opposite = (side == "long" and direction == "bearish") or (side == "short" and direction == "bullish")
        if opposite:
            close_simulated_position(conn, sim_pos, current_price, "emergency_high_vol")
            return True

    # ADX dead: require BOTH absolute low AND significant drop from entry.
    # Prevents healthy ADX fluctuations from prematurely closing positions
    # opened during low-volatility trends.
    #
    # Supplemental: also trigger on severe relative drop (≥ 30%) even if
    # absolute threshold hasn't been breached. Entry ADX=35 dropping to 25
    # doesn't meet the <20 threshold but represents a 29% collapse.
    entry_adx = sim_pos.get("entry_adx") or adx_4h
    adx_exit_threshold = 20
    adx_drop_from_entry = 5
    adx_drop_ratio = 0.30  # 30% relative drop

    if adx_4h:
        is_abs_low = adx_4h < adx_exit_threshold and adx_4h < (entry_adx - adx_drop_from_entry)
        is_relative_drop = entry_adx > 0 and adx_4h < entry_adx * (1 - adx_drop_ratio)
        if is_abs_low or is_relative_drop:
            reason = "emergency_relative_drop" if is_relative_drop and not is_abs_low else "emergency_adx_dead"
            close_simulated_position(conn, sim_pos, current_price, reason)
            return True

    return False


def _check_signal_reversal(conn, sim_pos: dict, verdict: dict, current_price: float) -> bool:
    """Priority 4: Signal reversal - engine explicitly says exit."""
    side = sim_pos["side"]
    regime = verdict.get("regime", "")

    # Check position-specific state from signal engine
    pos_state = verdict.get("hold_long" if side == "long" else "hold_short", {})
    level = pos_state.get("level", "")
    reason = pos_state.get("reason", "")

    # Exit signal from engine (requires explicit exit, not just direction flip)
    if level == "exit":
        close_simulated_position(conn, sim_pos, current_price, f"signal_reversal: {reason}")
        return True

    # Strong reversal: regime changed to opposing trend with exhaustion
    direction = verdict.get("direction")
    if regime == "exhaustion" and direction is not None:
        opposite = (side == "long" and direction == "bearish") or (side == "short" and direction == "bullish")
        if opposite:
            close_simulated_position(conn, sim_pos, current_price, "signal_reversal_exhaustion")
            return True

    return False


def _check_trend_exhaustion(conn, sim_pos: dict, verdict: dict, current_price: float, now: float) -> bool:
    """Reduce position when verdict detects trend exhaustion matching position side.

    Acts as an early warning — fires even when regime != "exhaustion",
    catching ADX ceiling, DI convergence, and price structure reversal.

    Grace period: newly opened positions (< 10 min) are exempt to avoid
    immediate closure from transient exhaustion signals.

    Exception: exhaustion_reversal positions are NOT subject to this exit.
    These positions are opened on the premise that the original trend is
    exhausting, so an exhaustion detection in the reversal direction is
    actually confirming the trade thesis, not invalidating it.
    """
    created_at = sim_pos.get("created_at") or 0
    if created_at > 0 and (now - created_at) < 600:  # 10-minute grace
        return False

    # Skip for reversal positions — exhaustion is the trade thesis, not an exit.
    signal_type = sim_pos.get("signal_type") or ""
    if signal_type == "exhaustion_reversal":
        return False

    market_ctx = verdict.get("market_context", {})
    exhausted = market_ctx.get("trend_exhausted", {})
    if not exhausted.get("detected"):
        return False

    # Only force-close on EXHAUSTED level. DECAYING means trend is weakening
    # but still valid — let the position run with normal trailing logic.
    level = exhausted.get("level", "EXHAUSTED")
    if level == "DECAYING":
        return False

    exhausted_dir = exhausted.get("direction")
    pos_side = sim_pos.get("side")

    # Same direction exhaustion = time to exit
    if (pos_side == "short" and exhausted_dir == "bearish") or \
       (pos_side == "long" and exhausted_dir == "bullish"):
        reason = exhausted.get("reason", "趋势末端，动能衰竭")
        close_simulated_position(conn, sim_pos, current_price, f"trend_exhaustion: {reason}")
        return True

    return False


def _check_circuit_breaker(conn, sim_pos: dict, verdict: dict, current_price: float) -> bool:
    """P3.2: When circuit breaker fires, immediately exit position.

    Circuit breaker reasons: flash crash (candle > ATR×3), liquidation cascade,
    DI spread too low, ADX freeze.
    """
    market_ctx = verdict.get("market_context", {})
    cb_reason = market_ctx.get("circuit_breaker")
    if not cb_reason:
        return False

    close_simulated_position(conn, sim_pos, current_price, f"circuit_breaker: {cb_reason}")
    return True


def _check_adx_trailing_stop(conn, sim_pos: dict, verdict: dict, current_price: float) -> bool:
    """ADX trailing stop: exit when ADX drops from peak + price crosses EMA20.

    Uses adaptive thresholds based on signal type and peak ADX level:
    - aggressive_adx7_early: max_adx >= 15, drop >= 5 (lower ADX range)
    - Normal signals: max_adx >= 30, drop >= max(8, max_adx * 0.15)
      Relative drop prevents premature exit from strong trends (e.g. 50→42
      is still a strong trend at 42, whereas 30→22 is significant decay).
    """
    signal_type = sim_pos.get("signal_type") or ""
    is_aggressive = signal_type == "aggressive_adx7_early"

    max_adx = sim_pos.get("max_adx") or 0

    # Adaptive thresholds
    if is_aggressive:
        adx_peak_threshold = 15
        adx_drop_points = 5
    else:
        adx_peak_threshold = 30
        # Relative drop: 8 points minimum, or 15% of peak ADX (whichever is larger)
        adx_drop_points = max(8, round(max_adx * 0.15))

    if max_adx < adx_peak_threshold:
        return False

    tf_4h = verdict.get("timeframes", {}).get("4h", {})
    adx_4h = tf_4h.get("adx", 0)

    # ADX drop condition
    if adx_4h > max_adx - adx_drop_points:
        return False

    # Price-based confirmation: check if price crossed EMA20 against position.
    ema20_4h = tf_4h.get("ema20")
    side = sim_pos["side"]

    if ema20_4h:
        if side == "long" and current_price < ema20_4h:
            close_simulated_position(conn, sim_pos, current_price, "adx_trailing_stop")
            return True
        if side == "short" and current_price > ema20_4h:
            close_simulated_position(conn, sim_pos, current_price, "adx_trailing_stop")
            return True
    else:
        # Fallback: use 4h price structure range midpoint (less accurate).
        # Add 1% buffer to avoid false triggers from proxy deviation.
        range_high = tf_4h.get("entry_position", {}).get("range_high", current_price)
        range_low = tf_4h.get("entry_position", {}).get("range_low", current_price)
        ema20_proxy = (range_high + range_low) / 2
        buffer = current_price * 0.01
        if side == "long" and current_price < ema20_proxy - buffer:
            close_simulated_position(conn, sim_pos, current_price, "adx_trailing_stop")
            return True
        if side == "short" and current_price > ema20_proxy + buffer:
            close_simulated_position(conn, sim_pos, current_price, "adx_trailing_stop")
            return True

    return False


def _manage_open_position(conn, sim_pos: dict, verdict: dict, current_price: float, now: float):
    """Priority 5-8: Signal-based management (exit / reduce / add / hold).

    Multiple reduces: after MAX_REDUCE_COUNT, auto-exit on next reduce.
    After reduce: stop remains at original level.
    Pyramid add-on: up to MAX_ADD_COUNT, leverage decreases each time.

    add_count semantics: lifetime counter for the position, NOT reset on reduce.
    A position that adds 3 times then reduces twice will have add_count=3,
    reduce_count=2. Further adds are blocked (add_count >= MAX_ADD_COUNT).
    This prevents infinite add-reduce cycles on a single position.
    """
    side = sim_pos["side"]
    pos_state = verdict.get("hold_long" if side == "long" else "hold_short", {})
    level = pos_state.get("level", "")

    # Safety fallback: unexpected level values (e.g. "open", "wait" from no-position
    # branch leaking due to future logic changes) should default to hold
    if level not in ("exit", "reduce", "add", "hold", ""):
        level = "hold"

    adx_4h = verdict.get("timeframes", {}).get("4h", {}).get("adx")
    reduce_count = sim_pos.get("reduce_count") or 0
    add_count = sim_pos.get("add_count") or 0

    # Exit signals (priority 5)
    if level == "exit":
        close_simulated_position(conn, sim_pos, current_price, "signal_exit")
        return

    # Reduce signals (priority 6)
    if level == "reduce":
        # Cooldown: half signal interval for reduce (was full interval).
        # Reduce is a risk-reduction action — shorter cooldown means
        # faster response to escalating market risk (e.g. volatility spike).
        last_sig = sim_pos.get("last_signal_time") or 0
        if now - last_sig < settings.signal_interval_seconds / 2:
            return

        new_reduce_count = reduce_count + 1
        if new_reduce_count > MAX_REDUCE_COUNT:
            close_simulated_position(conn, sim_pos, current_price, "reduce_max")
            return

        # Calculate partial pnl for the reduced portion
        reduce_pct = 0.30
        reduce_type = pos_state.get("reduce_type", "30pct")
        try:
            reduce_pct = int(reduce_type.replace("pct", "")) / 100
        except (ValueError, AttributeError):
            pass

        old_size = sim_pos.get("position_size") or 0
        reduce_size = old_size * reduce_pct
        realized_pnl = reduce_size * (current_price - sim_pos["entry_price"]) if sim_pos["side"] == "long" else reduce_size * (sim_pos["entry_price"] - current_price)
        new_size = old_size - reduce_size

        # Update position: reduce size, record cumulative realized pnl
        cum_pnl = (sim_pos.get("realized_pnl") or 0) + realized_pnl
        new_action_state = f"reduce_{new_reduce_count}"
        conn.execute(
            "UPDATE positions SET action_state=?, reduce_count=?, "
            "last_signal_time=?, position_size=?, realized_pnl=? WHERE id=?",
            (
                new_action_state,
                new_reduce_count, now,
                round(new_size, 6),
                round(cum_pnl, 2),
                sim_pos["id"],
            ),
        )
        conn.commit()

        # Record reduce action
        conn.execute(
            "INSERT INTO position_action_state (position_id, action, adx_4h, price, position_size, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sim_pos["id"], f"reduce_{new_reduce_count}", adx_4h, round(current_price, 1), round(reduce_size, 6), now),
        )
        conn.commit()

        # Fire reduce hooks
        for hook in _auto_reduce_hooks:
            try:
                hook(conn, sim_pos, reduce_size)
            except Exception:
                pass

        return

    # Add signals - pyramid add-on (priority 7)
    # Block adds during DECAYING — trend is weakening, no new exposure
    order_signal = verdict.get("order_signal", {})
    exhausted = verdict.get("market_context", {}).get("trend_exhausted", {})
    is_decay = exhausted.get("level") == "DECAYING"

    if level == "add" and is_decay:
        # Skip add, fall through to hold with stop tightening
        level = "hold"

    # Decay protection: when signal engine detects DECAYING trend + existing position
    # = trend weakening, reduce position to protect profits.
    # Only fires once per position (action_state tracks whether already decay-reduced).
    action_state = sim_pos.get("action_state") or ""
    if level == "hold" and is_decay and reduce_count == 0 and not action_state.startswith("decay_reduce"):
        order_side = order_signal.get("side", "")
        pos_is_short = side == "short"
        # When order_signal.side is "观望" (e.g. ADX ≥ 50 overheat),
        # use trend_exhausted.direction as fallback for same-dir check.
        effective_side = order_side
        if effective_side == "观望" and exhausted.get("direction"):
            effective_side = "做空" if exhausted["direction"] == "bearish" else "做多"
        decay_same_dir = (pos_is_short and effective_side == "做空") or (not pos_is_short and effective_side == "做多")
        if decay_same_dir:
            created_at = sim_pos.get("created_at") or 0
            grace_ok = (now - created_at) >= 600  # 10 min grace (aligned with trend_exhaustion)
            if grace_ok:
                entry = sim_pos["entry_price"]
                in_profit = (current_price < entry and pos_is_short) or (current_price > entry and not pos_is_short)

                reduce_size = (sim_pos.get("position_size") or 0) * 0.5
                realized_pnl = reduce_size * (entry - current_price) if pos_is_short else reduce_size * (current_price - entry)
                new_size = (sim_pos.get("position_size") or 0) - reduce_size

                if in_profit:
                    # Reduce to lock profits, then tighten stop to near breakeven
                    action_state = "decay_reduce"
                else:
                    # Not in profit: reduce to cut risk, tighten stop
                    action_state = "decay_reduce_loss"

                conn.execute(
                    "UPDATE positions SET action_state=?, reduce_count=?, "
                    "position_size=?, realized_pnl=?, last_signal_time=? WHERE id=?",
                    (action_state, 1, round(new_size, 6), round(realized_pnl, 2), now, sim_pos["id"]),
                )
                conn.commit()
                conn.execute(
                    "INSERT INTO position_action_state (position_id, action, adx_4h, price, position_size, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (sim_pos["id"], "decay_reduce_loss" if not in_profit else "decay_reduce",
                     adx_4h, round(current_price, 1), round(reduce_size, 6), now),
                )
                conn.commit()
                for hook in _auto_reduce_hooks:
                    try:
                        hook(conn, sim_pos, reduce_size)
                    except Exception:
                        pass

                return

    if level == "add" and add_count < MAX_ADD_COUNT:
        new_add_count = add_count + 1
        new_action_state = f"add_{new_add_count}"

        # Calculate add size using same 2% risk model
        risk_pct = 0.02
        stop = sim_pos.get("stop")
        stop_distance = abs(current_price - stop) if stop else current_price * 0.02
        leverage = sim_pos.get("leverage", 20)

        # Current available balance for add
        # Include realized pnl from open positions (partial reduces)
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as closed_pnl FROM positions "
            "WHERE is_simulated=1 AND status='closed'"
        ).fetchone()
        closed_pnl = row["closed_pnl"] or 0
        row2 = conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) as open_rpnl FROM positions "
            "WHERE is_simulated=1 AND status='open'"
        ).fetchone()
        open_realized = row2["open_rpnl"] or 0
        current_balance = settings.sim_initial_balance + closed_pnl + open_realized
        open_pos_margin = (sim_pos.get("position_size", 0) * sim_pos["entry_price"]) / leverage
        available = current_balance - open_pos_margin

        risk_amount = max(available, 0) * risk_pct
        add_btc = risk_amount / max(stop_distance, 1)

        # Pyramid: each add is smaller (70% of previous add)
        if new_add_count >= 2:
            add_btc *= 0.7 ** (new_add_count - 1)

        # Update weighted average entry price and adjust stop accordingly.
        # Without this, the add-on's risk distance is unbounded — the original
        # stop was sized for the first batch's entry, not the new average.
        # We keep the original stop distance (in price points) applied to the
        # new average entry, so total risk stays proportional.
        old_size = sim_pos.get("position_size") or 0
        old_entry = sim_pos["entry_price"]
        new_total_size = old_size + add_btc
        if new_total_size > 0:
            new_entry_price = (old_entry * old_size + current_price * add_btc) / new_total_size
        else:
            new_entry_price = current_price

        # Adjust stop to maintain original risk distance from new average entry
        old_stop_distance = abs(old_entry - stop) if stop else current_price * 0.02
        pos_side = sim_pos.get("side", "long")
        if pos_side == "long":
            new_stop = round(new_entry_price - old_stop_distance, 1)
        else:
            new_stop = round(new_entry_price + old_stop_distance, 1)

        conn.execute(
            "UPDATE positions SET action_state=?, max_adx=?, "
            "last_signal_time=?, add_count=?, position_size=?, "
            "entry_price=?, stop=? WHERE id=?",
            (
                new_action_state,
                max(sim_pos.get("max_adx") or 0, adx_4h or 0),
                now,
                new_add_count,
                round(new_total_size, 6),
                round(new_entry_price, 1),
                new_stop,
                sim_pos["id"],
            ),
        )
        conn.commit()

        # Record add action with size
        position_id = sim_pos["id"]
        conn.execute(
            "INSERT INTO position_action_state (position_id, action, adx_4h, price, position_size, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (position_id, new_action_state, adx_4h, round(current_price, 1), round(add_btc, 6), now),
        )
        conn.commit()

        # Skip engine advice stop sync after add: the stop computed from
        # weighted-average entry (L710-716) maintains the original risk
        # distance in price points. Engine's advice.stop is based on
        # current_price + ATR, which may diverge significantly from the
        # actual position's risk profile after a pyramid add.

        # Fire add hooks
        for hook in _auto_add_hooks:
            try:
                hook(conn, sim_pos, add_btc)
            except Exception:
                pass

        return

    # Hold - update max_adx and track price extremes for add-on detection
    if level == "hold":
        if adx_4h:
            current_max = sim_pos.get("max_adx") or 0
            if adx_4h > current_max:
                conn.execute(
                    "UPDATE positions SET max_adx=? WHERE id=?",
                    (adx_4h, sim_pos["id"]),
                )
                conn.commit()

        # Track max_price (for longs) and min_price (for shorts) so that
        # _is_price_extreme can detect genuine breakouts, not just
        # minor moves above/below the original entry price.
        pos_side = sim_pos.get("side")
        max_p = sim_pos.get("max_price")
        min_p = sim_pos.get("min_price")
        updates = {}
        if pos_side == "long" and (max_p is None or current_price > max_p):
            updates["max_price"] = round(current_price, 1)
        if pos_side == "short" and (min_p is None or current_price < min_p):
            updates["min_price"] = round(current_price, 1)
        if updates:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            conn.execute(
                f"UPDATE positions SET {set_clause} WHERE id=?",
                tuple(updates.values()) + (sim_pos["id"],),
            )
            conn.commit()



def _try_auto_open(conn, verdict: dict, current_price: float, now: float):
    """Auto-open a simulated position if signal conditions are met.

    Relaxed multi-timeframe:
    - 4h trending + bullish/ bearish, 1h not opposite (allow neutral/pullback)
    - Breakout regime allowed (with lower leverage)

    Cooldown:
    - After any close: wait 1 signal cycle before re-entry (same direction)
    - Opposite direction: wait 2 signal cycles

    Dynamic leverage:
    - Uses signal engine's order_signal.leverage (funding-rate adjusted)
    - Fallback: trending=20x, breakout=10x, forming=10x, exhaustion=10x, low_vol_trend=10x

    Position sizing (2% risk model):
    - risk_amount = available_balance * 2%
    - position_btc = risk_amount / stop_distance
    - margin = position_btc * entry_price / leverage
    """
    order = verdict.get("order_signal", {})
    order_side = order.get("side", "")

    if order_side not in ("做多", "做空"):
        return

    side = "long" if order_side == "做多" else "short"
    regime = verdict.get("regime", "")

    # Block opening on established trends (3+ consecutive trending cycles).
    # The signal engine still generates directional orders for established trends
    # (for dashboard display), but the position manager should not auto-open.
    # Exception: if 4h ADX is very strong (>=40), the trend is still accelerating.
    # But established trends open with HALF risk (1% vs 2%) — price is chasing, not pulling back.
    trend_fresh = verdict.get("market_context", {}).get("trend_fresh", True)
    adx_4h = verdict.get("timeframes", {}).get("4h", {}).get("adx", 0)
    established_trend = not trend_fresh and adx_4h >= 40
    if not trend_fresh and not established_trend:
        return

    # Block opening on low-conviction signal types.
    # convergence_warning: DI converging, direction uncertain — not actionable.
    # stage1_*: early warning stage, not yet a confirmed trade setup.
    signal_type = verdict.get("timeframes", {}).get("4h", {}).get("signal_type") or ""
    if signal_type in ("convergence_warning",) or signal_type.startswith("stage1_"):
        return

    # Cooldown check after any close, with consecutive stop-loss escalation
    last_closed = conn.execute(
        "SELECT side, MAX(closed_at) as last_close FROM positions WHERE is_simulated=1 AND status='closed'"
    ).fetchone()
    if last_closed and last_closed["last_close"] and last_closed["side"]:
        last_close_time = last_closed["last_close"]
        last_close_side = last_closed["side"]
        elapsed = now - last_close_time

        # Count consecutive stop losses, distinguishing same-direction vs whipsaw
        same_dir_sl = 0  # same direction = directional judgment wrong
        any_dir_sl = 0  # total consecutive SL (any direction) = market whipsaw
        rows = conn.execute(
            "SELECT side, close_reason FROM positions WHERE is_simulated=1 AND status='closed' "
            "ORDER BY closed_at DESC LIMIT 10"
        ).fetchall()
        for r in rows:
            if r["close_reason"] != "stop_loss":
                break
            any_dir_sl += 1
            if r["side"] == side:
                same_dir_sl += 1
            elif same_dir_sl > 0:
                # Alternating SL: break the same-direction streak but keep total count
                break

        cooldown = REOPEN_COOLDOWN_SECONDS
        if side != last_close_side:
            cooldown = OPPOSITE_DIRECTION_COOLDOWN

        # Escalate cooldown after consecutive stop losses:
        # Same-direction SL = direction wrong, escalate aggressively
        # Alternating SL (whipsaw) = market chop, moderate escalation
        if same_dir_sl >= 1:
            cooldown = min(cooldown * (2.0 ** same_dir_sl), 1800)  # 2x, 4x, 8x... cap 30min
        elif any_dir_sl >= 1:
            cooldown = min(cooldown * (1.5 ** any_dir_sl), 1800)  # 1.5x, 2.25x, 3.4x...

        if elapsed < cooldown:
            return

    # Price improvement check: same-direction re-entry must be at a better price.
    # Long: buy lower than last long entry. Short: sell higher than last short entry.
    # Prevents chasing — each re-entry must offer a better entry point.
    last_same_dir = conn.execute(
        "SELECT entry_price FROM positions WHERE is_simulated=1 AND status='closed' "
        "AND side = ? ORDER BY closed_at DESC LIMIT 1",
        (side,),
    ).fetchone()
    if last_same_dir and last_same_dir["entry_price"]:
        last_entry = last_same_dir["entry_price"]
        if side == "long" and current_price >= last_entry:
            return  # chasing longs — price not better than last entry
        if side == "short" and current_price <= last_entry:
            return  # chasing shorts — price not better than last entry

    # Block opening when market context shows trend exhaustion
    exhausted = verdict.get("market_context", {}).get("trend_exhausted", {})
    if exhausted.get("detected"):
        return

    # Dynamic leverage: prefer signal engine's suggestion, fallback by regime
    order_leverage = order.get("leverage", "")
    if isinstance(order_leverage, str) and order_leverage.endswith("x"):
        try:
            leverage = int(order_leverage.rstrip("x"))
        except ValueError:
            leverage = None
    if leverage is None:
        leverage = 20  # trending default
        if regime == "breakout":
            leverage = 10
        elif regime == "forming":
            leverage = 10
        elif regime == "exhaustion":
            leverage = 10
        elif regime == "low_vol_trend":
            leverage = 10  # low vol = lower leverage, less slippage risk

    # aggressive_adx7_early: noisy early signal, cap at 5x
    _sig_type = verdict.get("timeframes", {}).get("4h", {}).get("signal_type") or ""
    if _sig_type == "aggressive_adx7_early":
        leverage = min(leverage, 5)

    target = order.get("target")
    stop = order.get("stop")

    # Reject if stop is missing, equals entry, or too close (< 0.15% of price)
    # This prevents positions from being stopped out immediately
    entry_ref = order.get("entry_price") or current_price
    if stop is None:
        return  # no stop defined, unsafe to open
    stop_distance = abs(entry_ref - stop)
    min_stop = current_price * 0.0015  # 0.15% floor
    if stop_distance < min_stop:
        return  # stop too close to entry, would trigger immediately

    entry_adx = verdict.get("timeframes", {}).get("4h", {}).get("adx")
    signal_type = verdict.get("timeframes", {}).get("4h", {}).get("signal_type") or ""

    # Verify risk:reward ratio — dynamic by signal type.
    # Trend-following signals (trend_following, trend_pullback, low_vol_trend)
    # and high-conviction reversal/confirmation signals use evolution rr_ratio (default 1.5).
    # All other signals (stage1/macro/ranging) use 1.0 floor.
    if target and stop:
        entry = order.get("entry_price") or current_price
        risk = abs(entry - stop)
        reward = abs(target - entry)
        evol = get_active_thresholds()

        # Use signal_type for RR classification — more reliable than string parsing
        signal_type = verdict.get("timeframes", {}).get("4h", {}).get("signal_type") or ""
        is_trend_following = signal_type in (
            "trend_following", "trend_pullback", "low_vol_trend",
            "exhaustion_reversal",  # reversal signals require good RR too
        )
        rr_min = evol.get("rr_ratio", 1.5) if is_trend_following else 1.0

        if risk > 0 and reward / risk < rr_min:
            return  # RR too low, skip

    strength = verdict.get("strength", "")
    direction = verdict.get("direction", "")
    confidence = verdict.get("confidence", 0)
    alignment_rule = verdict.get("alignment_rule", "") or ""

    # Dynamic risk model: adjust risk_pct by signal quality
    if confidence >= 70 and strength == "强" and "4h trending" in alignment_rule:
        risk_pct = 0.05  # 5% — 极强趋势
    elif confidence >= 60 and strength in ("强", "中等"):
        risk_pct = 0.03  # 3% — 高确信
    else:
        risk_pct = 0.02  # 2% — 默认

    # Half-confirmation low_vol_trend: reduce risk to 1% (half of default)
    order_action = order.get("action", "")
    if "1h形成中" in order_action:
        risk_pct = 0.01

    # Established trend with strong ADX: chase risk, only 1%
    if established_trend:
        risk_pct = 0.01

    entry_reason = f"{regime} {direction} {strength} (conf:{confidence}, lev:{leverage}x, risk:{int(risk_pct*100)}%)"
    if established_trend:
        entry_reason += " [趋势追单]"

    # Calculate position size in BTC: dynamic risk model
    initial_balance = settings.sim_initial_balance

    # Check available balance: current_balance = initial + realized_pnl
    # Shared account: include ALL positions (manual + simulated)
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) as closed_pnl FROM positions "
        "WHERE status='closed'"
    ).fetchone()
    closed_pnl = row["closed_pnl"] or 0
    current_balance = initial_balance + closed_pnl

    # Subtract ALL open position margins (manual + simulated)
    # Use SUM to account for multiple concurrent positions
    row = conn.execute(
        "SELECT COALESCE(SUM(position_size * entry_price / COALESCE(leverage, 1)), 0) as total_margin "
        "FROM positions WHERE status='open' AND position_size IS NOT NULL AND entry_price IS NOT NULL"
    ).fetchone()
    margin_used = row["total_margin"] or 0
    available_balance = current_balance - margin_used

    if available_balance < 100:
        return  # Insufficient balance to open position (min 100 U)

    stop_distance = abs(current_price - stop) if stop else current_price * 0.02

    # Dynamic risk model:
    # risk_amount = balance * risk_pct  (max loss when stop hits)
    # position_btc = risk_amount / stop_distance  (BTC qty that loses exactly risk_amount)
    # margin = position_btc * entry_price / leverage  (capital required)
    risk_amount = available_balance * risk_pct
    position_btc = risk_amount / max(stop_distance, 1)

    # Cap position by available balance
    max_notional = available_balance * leverage
    max_btc = max_notional / current_price
    if position_btc > max_btc:
        position_btc = max_btc

    # Final DB-level guard: prevent duplicate positions from concurrent calls
    existing = conn.execute(
        "SELECT id FROM positions WHERE status='open' AND is_simulated=1 LIMIT 1"
    ).fetchone()
    if existing:
        return

    conn.execute(
        "INSERT INTO positions (side, entry_price, target, stop, leverage, "
        "status, created_at, updated_at, is_simulated, action_state, "
        "entry_adx, max_adx, entry_reason, reduce_count, add_count, position_size, signal_type, "
        "max_price, min_price) VALUES "
        "(?, ?, ?, ?, ?, 'open', ?, ?, 1, 'open', ?, ?, ?, 0, 0, ?, ?, ?, ?)",
        (
            side,
            round(current_price, 1),
            round(target, 1) if target else None,
            round(stop, 1) if stop else None,
            leverage,
            now,
            now,
            entry_adx,
            entry_adx,
            entry_reason,
            round(position_btc, 6),
            signal_type or None,
            round(current_price, 1),  # max_price = entry_price at open
            round(current_price, 1),  # min_price = entry_price at open
        ),
    )
    conn.commit()

    position_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO position_action_state (position_id, action, adx_4h, price, position_size, created_at) "
        "VALUES (?, 'open', ?, ?, ?, ?)",
        (position_id, entry_adx, round(current_price, 1), round(position_btc, 6), now),
    )
    conn.commit()

    # Fire open hooks (auto-trader mirror)
    new_pos = {
        "id": position_id, "side": side, "entry_price": round(current_price, 1),
        "position_size": round(position_btc, 6), "leverage": leverage,
        "signal_type": signal_type or None,
    }
    for hook in _auto_open_hooks:
        try:
            hook(conn, new_pos, verdict, current_price)
        except Exception:
            pass  # hook failure must not break position management


def manage_simulated_position(conn, data: dict, current_price: float):
    """Main entry point: manage simulated positions on each signal cycle.

    Strict priority chain:
      1. Stop loss
      2. Take profit
      3. PnL trailing stop (dual strategy: <30% lock 50% profit, >=30% fixed 10% pullback)
      4. Emergency exit (high volatility + opposite, ADX dead)
      4.5. Circuit breaker (flash crash, liquidation cascade, tighten stop if in profit)
      5. Signal reversal
      7. Trend exhaustion (early warning, regime may not be "exhaustion")
      8. ADX trailing stop
      8.5. Max hold time (48h without strong signal conviction)
      9. Exit signal
      10. Reduce signal (max 2, then auto-exit)
      11. Add signal (pyramid add, max 3)
      12. Hold
    """
    now = time.time()
    verdict = data.get("verdict", {})

    sim_pos_row = conn.execute(
        "SELECT * FROM positions WHERE status='open' AND is_simulated=1 ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    if sim_pos_row:
        sim_pos = dict(sim_pos_row)

        if _check_price_based_exit(conn, sim_pos, current_price):
            return

        pnl_changed = _check_pnl_trailing_stop(conn, sim_pos, current_price)
        if pnl_changed:
            # Re-sync in-memory state after DB update to stop/trailing values
            refreshed = conn.execute(
                "SELECT * FROM positions WHERE id = ?", (sim_pos["id"],)
            ).fetchone()
            if refreshed:
                sim_pos = dict(refreshed)

        if _check_emergency_exit(conn, sim_pos, verdict, current_price):
            return

        # P3.2: circuit breaker fired → protect existing position
        if _check_circuit_breaker(conn, sim_pos, verdict, current_price):
            return

        if _check_signal_reversal(conn, sim_pos, verdict, current_price):
            return

        # Trend exhaustion (priority 6): exit immediately when trend ends,
        # before waiting for ADX to drop below trailing threshold
        if _check_trend_exhaustion(conn, sim_pos, verdict, current_price, now):
            return

        if _check_adx_trailing_stop(conn, sim_pos, verdict, current_price):
            return

        # Priority 8.5: Max holding time protection.
        # Positions held beyond their expected duration without strong signal
        # conviction → exit. Prevents stale positions from lingering through
        # changed market conditions.
        #
        # Timeouts are sized for 4h cycle: a full trend wave (forming → trending
        # → exhaustion) typically runs 5-10 days (120-240h), so 48h is too short.
        #
        # Profit override: if leveraged PnL ≥ 5%, skip the timeout even with
        # weak signal — let profits run, only cut "stale + unprofitable" positions.
        created_at = sim_pos.get("created_at") or 0
        signal_type = sim_pos.get("signal_type") or ""
        holding_hours = (now - created_at) / 3600

        if signal_type in ("low_vol_trend", "macro_bias_long"):
            max_hours = 120  # 5 days for slow strategies (was 96h)
            strong_confidence = 50
            strong_strength = ("强", "中等")
        elif signal_type in ("ranging_breakout", "exhaustion_reversal", "aggressive_adx7_early"):
            max_hours = 36  # 1.5 days for fast strategies (was 24h)
            strong_confidence = 60
            strong_strength = ("强",)
        else:
            max_hours = 72  # 3 days for default trend-following (was 48h)
            strong_confidence = 50
            strong_strength = ("强", "中等")

        if holding_hours >= max_hours:
            # Profit override: if position ever reached >= 5% leveraged profit,
            # don't force-exit. Let trailing stop protect it.
            # Use price extreme (min_price for short, max_price for long) to
            # avoid penalizing momentary price fluctuations.
            entry = sim_pos.get("entry_price") or 0
            leverage = sim_pos.get("leverage") or 20
            side = sim_pos.get("side")
            if entry > 0:
                # Calculate best-ever pnl from recorded price extreme
                if side == "long":
                    best_price = sim_pos.get("max_price") or current_price
                    best_pnl = (best_price - entry) / entry * 100 * leverage
                else:
                    best_price = sim_pos.get("min_price") or current_price
                    best_pnl = (entry - best_price) / entry * 100 * leverage

                if best_pnl >= 5:
                    # Position was profitable at some point — let it run
                    pass
                else:
                    confidence = verdict.get("confidence", 0)
                    strength = verdict.get("strength", "")
                    weak_signal = confidence < strong_confidence or strength not in strong_strength
                    if weak_signal:
                        close_simulated_position(
                            conn, sim_pos, current_price,
                            f"max_hold_time: {holding_hours:.1f}h, conf={confidence}, strength={strength}, best_pnl={best_pnl:.1f}%",
                        )
                        return

        _manage_open_position(conn, sim_pos, verdict, current_price, now)
    else:
        _try_auto_open(conn, verdict, current_price, now)

"""Simulated position manager: auto-open and auto-manage simulated positions.

Priority chain (strict):
  1. Stop loss (price hits stop)
  2. Take profit (price hits target)
  3. Emergency exit (high volatility + opposite direction, ADX < adx_exit)
  4. Signal reversal (verdict direction flipped against position)
  5. ADX trailing stop (max_adx >= 30, current <= max_adx - 8)
  6. Exit signal (engine level='exit')
  7. Reduce signal (engine level='reduce', max 2 times -> then auto-exit)
  8. Add signal (engine level='add', with trailing stop update)
  9. Hold (update max_adx)

Features:
- Multiple reduce (max 2), auto-exit on 3rd reduce
- Signal cooldown: cooldown only applies to opposite direction; same-direction
  re-entry allowed immediately (trend continuation)
- Trailing stop: after reduce, move stop to entry price (breakeven)
- Dynamic leverage: uses signal engine's order_signal.leverage (funding-rate adjusted),
  fallback: trending=20x, breakout=10x, forming=10x, exhaustion=10x
- Pyramid add-on: triggered by engine 'add' signal, max 3 adds
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


SIGNAL_COOLDOWN_SECONDS = 2 * settings.signal_interval_seconds  # opposite direction cooldown
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
    """Close a simulated position and record the reason and PnL."""
    # Guard: skip if already closed (double-close protection)
    if position.get("status") != "open":
        return

    # Double-check against DB to avoid in-memory stale state
    db_row = conn.execute(
        "SELECT id, status FROM positions WHERE id = ?", (position["id"],)
    ).fetchone()
    if not db_row or db_row["status"] != "open":
        return

    pnl = calculate_pnl(position, close_price)
    now = time.time()
    conn.execute(
        "UPDATE positions SET status='closed', pnl=?, close_reason=?, "
        "close_price=?, closed_at=?, updated_at=? WHERE id=?",
        (round(pnl, 2), reason, round(close_price, 1), now, now, position["id"]),
    )
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


def _update_trailing_stop(conn, position: dict):
    """After reduce, move stop to entry price (breakeven protection)."""
    conn.execute(
        "UPDATE positions SET stop=? WHERE id=?",
        (position["entry_price"], position["id"]),
    )
    conn.commit()


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
    entry_adx = sim_pos.get("entry_adx") or adx_4h
    adx_exit_threshold = 20
    adx_drop_from_entry = 5  # must have dropped meaningfully from entry
    if adx_4h and adx_4h < adx_exit_threshold and adx_4h < (entry_adx - adx_drop_from_entry):
        close_simulated_position(conn, sim_pos, current_price, "emergency_adx_dead")
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
    """
    created_at = sim_pos.get("created_at") or 0
    if created_at > 0 and (now - created_at) < 600:  # 10-minute grace
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


def _check_adx_trailing_stop(conn, sim_pos: dict, verdict: dict, current_price: float) -> bool:
    """ADX trailing stop: max_adx >= 30 and current ADX <= max_adx - 8 AND price crosses EMA20.

    ADX alone can trigger false exits during healthy retracements.
    Adding EMA20 cross confirmation prevents premature exits.
    """
    max_adx = sim_pos.get("max_adx") or 0
    if max_adx < 30:
        return False

    tf_4h = verdict.get("timeframes", {}).get("4h", {})
    adx_4h = tf_4h.get("adx", 0)

    # ADX drop condition
    if adx_4h > max_adx - 8:
        return False

    # Price-based confirmation: check if price crossed EMA20 against position.
    # Use 4h EMA20 directly (computed in _analyze_single_timeframe and passed
    # through verdict.timeframes['4h'].ema20).
    tf_4h = verdict.get("timeframes", {}).get("4h", {})
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
        # Fallback: use 4h price structure range midpoint (less accurate)
        range_high = tf_4h.get("entry_position", {}).get("range_high", current_price)
        range_low = tf_4h.get("entry_position", {}).get("range_low", current_price)
        ema20_proxy = (range_high + range_low) / 2
        if side == "long" and current_price < ema20_proxy:
            close_simulated_position(conn, sim_pos, current_price, "adx_trailing_stop")
            return True
        if side == "short" and current_price > ema20_proxy:
            close_simulated_position(conn, sim_pos, current_price, "adx_trailing_stop")
            return True

    return False


def _manage_open_position(conn, sim_pos: dict, verdict: dict, current_price: float, now: float):
    """Priority 5-8: Signal-based management (exit / reduce / add / hold).

    Multiple reduces: after MAX_REDUCE_COUNT, auto-exit on next reduce.
    After reduce: trailing stop moved to breakeven.
    Pyramid add-on: up to MAX_ADD_COUNT, leverage decreases each time.
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
        last_sig = sim_pos.get("last_signal_time") or 0
        if now - last_sig < settings.signal_interval_seconds:
            return

        new_reduce_count = reduce_count + 1
        if new_reduce_count >= MAX_REDUCE_COUNT:
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
        conn.execute(
            "UPDATE positions SET action_state='reduced', reduce_count=?, "
            "last_signal_time=?, position_size=?, realized_pnl=?, "
            "realized_pnl_record=? WHERE id=?",
            (
                new_reduce_count, now,
                round(new_size, 6),
                round(cum_pnl, 2),
                round(cum_pnl, 2),
                sim_pos["id"],
            ),
        )
        conn.commit()
        _update_trailing_stop(conn, sim_pos)

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
    is_decay = order_signal.get("decay", False)

    if level == "add" and is_decay:
        # Skip add, fall through to hold with stop tightening
        level = "hold"

    if level == "add" and add_count < MAX_ADD_COUNT:
        new_add_count = add_count + 1
        new_action_state = f"add_{new_add_count}"

        # Calculate add size using same 2% risk model
        risk_pct = 0.02
        stop = sim_pos.get("stop")
        stop_distance = abs(current_price - stop) if stop else current_price * 0.02
        leverage = sim_pos.get("leverage", 20)

        # Current available balance for add
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as closed_pnl FROM positions "
            "WHERE is_simulated=1 AND status='closed'"
        ).fetchone()
        closed_pnl = row["closed_pnl"] or 0
        current_balance = settings.sim_initial_balance + closed_pnl
        open_pos_margin = (sim_pos.get("position_size", 0) * sim_pos["entry_price"]) / leverage
        available = current_balance - open_pos_margin

        risk_amount = max(available, 0) * risk_pct
        add_btc = risk_amount / max(stop_distance, 1)

        # Pyramid: each add is smaller (70% of previous add)
        if new_add_count >= 2:
            add_btc *= 0.7 ** (new_add_count - 1)

        # Update weighted average entry price
        old_size = sim_pos.get("position_size") or 0
        old_entry = sim_pos["entry_price"]
        new_total_size = old_size + add_btc
        if new_total_size > 0:
            new_entry_price = (old_entry * old_size + current_price * add_btc) / new_total_size
        else:
            new_entry_price = current_price

        conn.execute(
            "UPDATE positions SET action_state=?, max_adx=?, "
            "last_signal_time=?, add_count=?, position_size=?, "
            "entry_price=? WHERE id=?",
            (
                new_action_state,
                max(sim_pos.get("max_adx") or 0, adx_4h or 0),
                now,
                new_add_count,
                round(new_total_size, 6),
                round(new_entry_price, 1),
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

        # Fire add hooks
        for hook in _auto_add_hooks:
            try:
                hook(conn, sim_pos, add_btc)
            except Exception:
                pass

        return

    # Hold - just update max_adx
    if level == "hold" and adx_4h:
        current_max = sim_pos.get("max_adx") or 0
        if adx_4h > current_max:
            conn.execute(
                "UPDATE positions SET max_adx=? WHERE id=?",
                (adx_4h, sim_pos["id"]),
            )
            conn.commit()

        # DECAYING: tighten stop toward breakeven (50% of profit distance),
        # but ONLY when position is in profit (>= 1% from entry).
        # Minimum 5-minute interval to avoid micro-adjustments.
        if is_decay:
            DECAY_TIGHTEN_INTERVAL = 5 * 60  # 5 minutes
            last_tightened = sim_pos.get("last_decay_tightened") or sim_pos.get("updated_at") or 0
            if now - last_tightened >= DECAY_TIGHTEN_INTERVAL:
                current_stop = sim_pos.get("stop")
                entry = sim_pos["entry_price"]
                pos_side = sim_pos["side"]
                if current_stop and entry and entry > 0:
                    if pos_side == "long":
                        pnl_pct = (current_price - entry) / entry
                        new_stop = current_stop + (current_price - current_stop) * 0.5
                        new_stop = min(new_stop, entry)
                    elif pos_side == "short":
                        pnl_pct = (entry - current_price) / entry
                        new_stop = current_stop - (current_stop - current_price) * 0.5
                        new_stop = max(new_stop, entry)
                    else:
                        pnl_pct = 0
                        new_stop = current_stop
                    if pnl_pct >= 0.01 and abs(new_stop - current_stop) > 1:
                        conn.execute(
                            "UPDATE positions SET stop=?, last_decay_tightened=? WHERE id=?",
                            (round(new_stop, 1), now, sim_pos["id"]),
                        )
                        conn.commit()


def _try_auto_open(conn, verdict: dict, current_price: float, now: float):
    """Auto-open a simulated position if signal conditions are met.

    Relaxed multi-timeframe:
    - 4h trending + bullish/ bearish, 1h not opposite (allow neutral/pullback)
    - Breakout regime allowed (with lower leverage)

    Cooldown:
    - Opposite direction signal after close: wait cooldown
    - Same direction (trend continuation): allow immediate re-entry

    Dynamic leverage:
    - Uses signal engine's order_signal.leverage (funding-rate adjusted)
    - Fallback: trending=20x, breakout=10x, forming=10x, exhaustion=10x

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

    # Cooldown check: only blocks opposite direction re-entry
    # Same direction (trend continuation) allowed immediately
    last_closed = conn.execute(
        "SELECT side, MAX(closed_at) as last_close FROM positions WHERE is_simulated=1 AND status='closed'"
    ).fetchone()
    if last_closed and last_closed["last_close"] and last_closed["side"]:
        last_close_time = last_closed["last_close"]
        last_close_side = last_closed["side"]
        # Opposite direction: enforce cooldown
        if side != last_close_side:
            if now - last_close_time < SIGNAL_COOLDOWN_SECONDS:
                return

    # Dynamic leverage: prefer signal engine's suggestion, fallback by regime
    order_leverage = order.get("leverage", "")
    if order_leverage and order_leverage.endswith("x"):
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

    target = order.get("target")
    stop = order.get("stop")
    entry_adx = verdict.get("timeframes", {}).get("4h", {}).get("adx")
    signal_type = verdict.get("timeframes", {}).get("4h", {}).get("signal_type", "")

    # Verify risk:reward ratio — dynamic by signal type.
    # Trend signals (4h trending + aligned) use evolution rr_ratio (default 1.5).
    # All other signals (aggressive/stage1/macro/ranging etc.) use 1.0 floor —
    # 0.5 was too low, allowing poor-expectancy signals to open.
    # DECAYING entries require RR ≥ 1.2 regardless of signal type.
    is_decay = order.get("decay", False)
    if target and stop:
        entry = order.get("entry_price") or current_price
        risk = abs(entry - stop)
        reward = abs(target - entry)
        evol = get_active_thresholds()

        alignment_rule = verdict.get("alignment_rule", "") or ""
        is_trending = "trending" in alignment_rule and "4h" in alignment_rule
        rr_min = evol.get("rr_ratio", 1.5) if is_trending else 1.0

        if is_decay:
            rr_min = max(rr_min, 1.2)  # decay entries require RR ≥ 1.2 (light 1% risk)

        if risk > 0 and reward / risk < rr_min:
            return  # RR too low, skip

    strength = verdict.get("strength", "")
    direction = verdict.get("direction", "")
    confidence = verdict.get("confidence", 0)
    alignment_rule = verdict.get("alignment_rule", "") or ""

    # Dynamic risk model: adjust risk_pct by signal quality
    if is_decay:
        risk_pct = 0.01  # 1% — decay entry, reduced risk
    elif confidence >= 70 and strength == "强" and "4h trending" in alignment_rule:
        risk_pct = 0.05  # 5% — 极强趋势
    elif confidence >= 60 and strength in ("强", "中等"):
        risk_pct = 0.03  # 3% — 高确信
    else:
        risk_pct = 0.02  # 2% — 默认

    entry_reason = f"{regime} {direction} {strength} (conf:{confidence}, lev:{leverage}x, risk:{int(risk_pct*100)}%)"

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

    # Subtract ALL open position margin
    open_pos = conn.execute(
        "SELECT position_size, entry_price, leverage FROM positions "
        "WHERE status='open' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if open_pos and open_pos["position_size"] and open_pos["entry_price"]:
        margin_used = (open_pos["position_size"] * open_pos["entry_price"]) / (open_pos["leverage"] or 1)
    else:
        margin_used = 0
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
        "entry_adx, max_adx, entry_reason, reduce_count, add_count, position_size) VALUES "
        "(?, ?, ?, ?, ?, 'open', ?, ?, 1, 'open', ?, ?, ?, 0, 0, ?)",
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
      3. Emergency exit (high volatility + opposite, ADX dead)
      4. Signal reversal
      5. ADX trailing stop
      6. Trend exhaustion (early warning, regime may not be "exhaustion")
      7. Exit signal
      8. Reduce signal (max 2, then auto-exit)
      9. Add signal (pyramid add, max 3)
      10. Hold
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

        if sim_pos.get("status") != "open":
            return

        if _check_emergency_exit(conn, sim_pos, verdict, current_price):
            return

        if sim_pos.get("status") != "open":
            return

        if _check_signal_reversal(conn, sim_pos, verdict, current_price):
            return

        if sim_pos.get("status") != "open":
            return

        # ADX trailing stop (new)
        if _check_adx_trailing_stop(conn, sim_pos, verdict, current_price):
            return

        if sim_pos.get("status") != "open":
            return

        # Trend exhaustion: early warning even when regime != "exhaustion"
        if _check_trend_exhaustion(conn, sim_pos, verdict, current_price, now):
            return

        _manage_open_position(conn, sim_pos, verdict, current_price, now)
    else:
        _try_auto_open(conn, verdict, current_price, now)

"""Auto-trader: callback hooks to mirror simulated positions to real execution.

When AUTO_TRADE_ENABLED is true, every simulated position action (open/close/reduce/add)
triggers a corresponding real order via TradeRouter.

Architecture:
  auto_trader.register_callbacks() → pushes hooks into simulated_position_manager
  simulated_position_manager fires hooks after each DB write.
  TradeRouter routes to Hyperliquid or simulated based on TRADE_BACKEND config.
"""

from __future__ import annotations

import time
import logging

from app.config import settings
from app.trade_executor import get_router
from app.database import record_hl_order, update_hl_order_price

_logger = logging.getLogger(__name__)


_RETRYABLE_ERRORS = (
    "timeout", "timed out", "connection", "refused",
    "ssl", "certificate", "unreachable",
    "market_open succeeded",  # OID extraction failed, but market fill may have happened
)


def _is_retryable_error(error: str) -> bool:
    """Determine if an HL open failure is a network issue worth retrying."""
    if not error:
        return False
    lower = error.lower()
    return any(kw in lower for kw in _RETRYABLE_ERRORS)


def _hl_mirror_open(conn, position: dict, verdict: dict, price: float):
    """Mirror a simulated open via TradeRouter.

    On failure: distinguishes between retryable (network) and hard errors
    (insufficient balance, rejected order). Hard errors delete the position
    to prevent stale retries. Retryable errors leave the position record
    intact for _hl_position_reconcile to retry.
    """
    if not settings.auto_trade_enabled:
        return

    side = position.get("side", "")
    btc_size = position.get("position_size") or 0
    leverage = int(position.get("leverage", 20))

    if not btc_size or btc_size <= 0:
        _logger.warning("HL mirror open skipped: invalid position_size=%s", btc_size)
        return

    tp = position.get("target")
    sl = position.get("stop")
    router = get_router()
    _logger.info("HL mirror OPEN: side=%s size=%s leverage=%s tp=%s sl=%s", side, btc_size, leverage, tp, sl)
    result = router.open(side, round(btc_size, 5), leverage, tp, sl)

    if result.get("ok"):
        order_id = result.get("order_id")
        if not order_id:
            _logger.error("HL mirror open: order_id is None despite ok=True — likely _extract_oid failure")
            return
        tp_oid = result.get("tp_oid")
        sl_oid = result.get("sl_oid")
        updates = ["hl_enabled=1", "hl_sz=?"]
        values: list = [round(btc_size, 5)]
        values.append(str(order_id))
        updates.append("hl_entry_oid=?")
        if tp_oid:
            values.append(str(tp_oid))
            updates.append("hl_tp_oid=?")
        if sl_oid:
            values.append(str(sl_oid))
            updates.append("hl_sl_oid=?")
        values.append(position["id"])
        conn.execute(
            f"UPDATE positions SET {', '.join(updates)} WHERE id=?",
            tuple(values),
        )
        conn.commit()

        # Record HL order mappings
        record_hl_order(position["id"], order_id, "open", size=round(btc_size, 5), price=price)
        if tp_oid:
            record_hl_order(position["id"], tp_oid, "tp", size=round(btc_size, 5), price=tp)
        if sl_oid:
            record_hl_order(position["id"], sl_oid, "sl", size=round(btc_size, 5), price=sl)
    else:
        error = result.get("error", "")
        retryable = _is_retryable_error(str(error))
        if retryable:
            _logger.warning("HL mirror open: retryable error (will retry via reconcile): %s", error)
            # Leave position record intact — _hl_position_reconcile will retry
        else:
            _logger.error("HL mirror open: hard failure (deleting position): %s", error)
            conn.execute("DELETE FROM positions WHERE id=?", (position["id"],))
            conn.execute("DELETE FROM position_action_state WHERE position_id=?", (position["id"],))
            conn.commit()


def _hl_mirror_close(conn, position: dict, close_price: float, reason: str):
    """Post-close safety net hook.

    HL close was already executed inside close_simulated_position() before the
    DB was updated. Under normal operation this hook does nothing.

    Safety net: if HL still has a position (partial failure, legacy caller, or
    external close), force-close it and correct the DB PnL with the actual HL
    fill price.
    """
    if not settings.auto_trade_enabled:
        return

    # Quick safety net: if HL still has a position, try to close it.
    # This should NOT happen under the new HL-first close flow, but guards
    # against legacy callers or partial failures.
    router = get_router()
    try:
        hl_pos, hl_err = router.get_position_state()
        if hl_pos and hl_pos.get("size"):
            _logger.warning("HL mirror close hook: unexpected HL position detected (size=%s), forcing close",
                            hl_pos.get("size"))
            result = router.close(round(hl_pos.get("size"), 5))
            if result.get("ok"):
                fill_price = result.get("fill_price")
                if fill_price and fill_price > 0:
                    entry = position.get("entry_price") or 0
                    side = position.get("side", "")
                    btc_size = hl_pos.get("size")
                    if entry > 0:
                        if side == "long":
                            hl_pnl = btc_size * (fill_price - entry)
                        else:
                            hl_pnl = btc_size * (entry - fill_price)
                        hl_pnl = round(hl_pnl, 2)
                        # Add accumulated realized PnL from partial reduces
                        realized = position.get("realized_pnl") or 0
                        if realized != 0:
                            hl_pnl = round(hl_pnl + realized, 2)
                    conn.execute(
                        "UPDATE positions SET pnl=?, close_price=?, hl_close_oid=? WHERE id=?",
                        (hl_pnl, round(fill_price, 1), str(result.get("order_id")), position["id"]),
                    )
                    conn.commit()
                    _logger.info("HL mirror close hook (safety net): fill=%s pnl=%s", fill_price, hl_pnl)
                    record_hl_order(position["id"], result.get("order_id"), "close",
                                    size=btc_size, price=fill_price)
        else:
            pass  # No HL position — expected state after HL-first close
    except Exception:
        pass  # Safety net is best-effort


def _hl_mirror_reduce(conn, position: dict, reduce_size: float):
    """Mirror a reduce action via TradeRouter.

    After a partial close on HL, the existing TP/SL orders still reference
    the original position size. We must resize them to match the new
    position, otherwise they would over-execute if triggered.
    """
    if not settings.auto_trade_enabled:
        return

    if not reduce_size or reduce_size <= 0:
        _logger.warning("HL mirror reduce skipped: invalid reduce_size=%s", reduce_size)
        return

    router = get_router()
    _logger.info("HL mirror REDUCE: size=%s", reduce_size)
    result = router.reduce(round(reduce_size, 5))

    if not result.get("ok"):
        _logger.error("HL mirror reduce FAILED: %s", result.get("error"))
        return

    # Record reduce order mapping
    reduce_oid = result.get("order_id")
    if reduce_oid:
        record_hl_order(position["id"], reduce_oid, "reduce",
                        size=round(reduce_size, 5), price=result.get("fill_price"))

    # Update hl_sz and resize TP/SL orders to match the new position size.
    # Hooks receive the position dict BEFORE the DB refresh in caller,
    # so we must re-query to get the actual updated position_size, stop, target.
    row = conn.execute(
        "SELECT position_size, hl_tp_oid, hl_sl_oid, side, stop, target FROM positions "
        "WHERE id=? AND hl_enabled=1",
        (position["id"],),
    ).fetchone()
    if row and row["position_size"]:
        new_sz = round(row["position_size"], 5)
        side = row["side"] or ""
        conn.execute(
            "UPDATE positions SET hl_sz=? WHERE id=? AND hl_enabled=1",
            (new_sz, position["id"]),
        )
        conn.commit()

        # Correct realized_pnl with actual HL fill price
        fill_price = result.get("fill_price")
        if fill_price and fill_price > 0:
            entry = position.get("entry_price") or 0
            if entry > 0:
                if side == "long":
                    actual_pnl = reduce_size * (fill_price - entry)
                else:
                    actual_pnl = reduce_size * (entry - fill_price)
                # Get current realized_pnl from DB and replace with actual
                current_pnl_row = conn.execute(
                    "SELECT realized_pnl FROM positions WHERE id=?",
                    (position["id"],),
                ).fetchone()
                current_pnl = current_pnl_row["realized_pnl"] if current_pnl_row else 0
                # We need to know the old simulated pnl to correct it
                # The reduce already added a computed realized_pnl; replace with actual
                old_realized = current_pnl - actual_pnl  # remove the just-added simulated pnl
                corrected = round(old_realized + actual_pnl, 2)
                conn.execute(
                    "UPDATE positions SET realized_pnl=? WHERE id=?",
                    (corrected, position["id"]),
                )
                conn.commit()
                _logger.info("HL reduce PnL corrected: fill=%s actual=%s",
                             fill_price, actual_pnl)

        # Resize TP order at the existing target price
        tp_oid = row.get("hl_tp_oid")
        tp_price = row.get("target")
        if tp_oid and tp_oid != "None" and tp_price:
            tp_result = router.set_take_profit(str(tp_oid), tp_price, new_sz, side)
            if tp_result.get("ok"):
                new_tp_oid = tp_result.get("order_id")
                if new_tp_oid and new_tp_oid != tp_oid:
                    conn.execute(
                        "UPDATE positions SET hl_tp_oid=? WHERE id=? AND hl_enabled=1",
                        (str(new_tp_oid), position["id"]),
                    )
                    conn.commit()
                    record_hl_order(position["id"], new_tp_oid, "tp", size=new_sz, price=tp_price)
                else:
                    # oid unchanged — modify succeeded in-place, record price update
                    update_hl_order_price(str(tp_oid), tp_price, new_size=new_sz)
                _logger.info("HL REDUCE TP resize: price=%s size=%s oid=%s", tp_price, new_sz, new_tp_oid)
            else:
                _logger.warning("HL REDUCE TP resize failed: %s", tp_result.get("error"))

        # Resize SL order at the current stop price (trailing may have moved it)
        sl_oid = row.get("hl_sl_oid")
        sl_price = row.get("stop")
        if sl_oid and sl_oid != "None" and sl_price:
            sl_result = router.set_stop_loss(str(sl_oid), sl_price, new_sz, side)
            if sl_result.get("ok"):
                new_sl_oid = sl_result.get("order_id")
                if new_sl_oid and new_sl_oid != sl_oid:
                    conn.execute(
                        "UPDATE positions SET hl_sl_oid=? WHERE id=? AND hl_enabled=1",
                        (str(new_sl_oid), position["id"]),
                    )
                    conn.commit()
                    record_hl_order(position["id"], new_sl_oid, "sl", size=new_sz, price=sl_price)
                else:
                    update_hl_order_price(str(sl_oid), sl_price, new_size=new_sz)
                _logger.info("HL REDUCE SL resize: price=%s size=%s oid=%s", sl_price, new_sz, new_sl_oid)
            else:
                _logger.warning("HL REDUCE SL resize failed: %s", sl_result.get("error"))


def _hl_mirror_add(conn, position: dict, add_size: float):
    """Mirror an add (pyramid) action via TradeRouter.

    After adding to position on HL, we resize TP/SL orders to match the
    new total position size. HL's TP/SL orders don't auto-resize when
    the position grows via market_open.
    """
    if not settings.auto_trade_enabled:
        return

    if not add_size or add_size <= 0:
        _logger.warning("HL mirror add skipped: invalid add_size=%s", add_size)
        return

    side = position.get("side", "")
    leverage = int(position.get("leverage", 20))

    router = get_router()
    _logger.info("HL mirror ADD: side=%s size=%s leverage=%s", side, add_size, leverage)
    result = router.add(side, round(add_size, 5), leverage)

    if not result.get("ok"):
        _logger.error("HL mirror add FAILED: %s", result.get("error"))
        return

    # Record add order mapping
    add_oid = result.get("order_id")
    if add_oid:
        record_hl_order(position["id"], add_oid, "add", size=round(add_size, 5))

    # Update hl_sz and resize TP/SL orders to match the new position size.
    # Hooks receive the position dict BEFORE the DB refresh in caller,
    # so we must re-query to get the actual updated position_size, stop, target.
    row = conn.execute(
        "SELECT position_size, hl_tp_oid, hl_sl_oid, side, stop, target FROM positions "
        "WHERE id=? AND hl_enabled=1",
        (position["id"],),
    ).fetchone()
    if row and row["position_size"]:
        new_sz = round(row["position_size"], 5)
        side = row["side"] or ""
        conn.execute(
            "UPDATE positions SET hl_sz=? WHERE id=? AND hl_enabled=1",
            (new_sz, position["id"]),
        )
        conn.commit()

        # Resize TP order at the existing target price
        tp_oid = row.get("hl_tp_oid")
        tp_price = row.get("target")
        if tp_oid and tp_oid != "None" and tp_price:
            tp_result = router.set_take_profit(str(tp_oid), tp_price, new_sz, side)
            if tp_result.get("ok"):
                new_tp_oid = tp_result.get("order_id")
                if new_tp_oid and new_tp_oid != tp_oid:
                    conn.execute(
                        "UPDATE positions SET hl_tp_oid=? WHERE id=? AND hl_enabled=1",
                        (str(new_tp_oid), position["id"]),
                    )
                    conn.commit()
                    record_hl_order(position["id"], new_tp_oid, "tp", size=new_sz, price=tp_price)
                else:
                    update_hl_order_price(str(tp_oid), tp_price, new_size=new_sz)
                _logger.info("HL ADD TP resize: price=%s size=%s oid=%s", tp_price, new_sz, new_tp_oid)
            else:
                _logger.warning("HL ADD TP resize failed: %s", tp_result.get("error"))

        # Resize SL order at the current stop price (adjusted for avg entry)
        sl_oid = row.get("hl_sl_oid")
        sl_price = row.get("stop")
        if sl_oid and sl_oid != "None" and sl_price:
            sl_result = router.set_stop_loss(str(sl_oid), sl_price, new_sz, side)
            if sl_result.get("ok"):
                new_sl_oid = sl_result.get("order_id")
                if new_sl_oid and new_sl_oid != sl_oid:
                    conn.execute(
                        "UPDATE positions SET hl_sl_oid=? WHERE id=? AND hl_enabled=1",
                        (str(new_sl_oid), position["id"]),
                    )
                    conn.commit()
                    record_hl_order(position["id"], new_sl_oid, "sl", size=new_sz, price=sl_price)
                else:
                    update_hl_order_price(str(sl_oid), sl_price, new_size=new_sz)
                _logger.info("HL ADD SL resize: price=%s size=%s oid=%s", sl_price, new_sz, new_sl_oid)
            else:
                _logger.warning("HL ADD SL resize failed: %s", sl_result.get("error"))


def _hl_mirror_stop_update(conn, position: dict, new_stop_price: float):
    """Mirror a trailing stop price update to Hyperliquid.

    When the simulated position's stop price moves via PnL trailing,
    this updates the corresponding SL trigger order on Hyperliquid.
    """
    if not settings.auto_trade_enabled:
        return

    row = conn.execute(
        "SELECT hl_sl_oid, position_size, side FROM positions WHERE id=? AND hl_enabled=1",
        (position["id"],),
    ).fetchone()
    _logger.info("[HL mirror SL UPDATE] query result: row_exists=%s hl_sl_oid=%s",
                 row is not None, str(row["hl_sl_oid"]) if row else "N/A")
    if not row or not row["hl_sl_oid"] or row["hl_sl_oid"] == "None":
        return

    sz = round(row["position_size"], 5) if row["position_size"] else 0
    side = row["side"] or ""
    if not sz:
        _logger.warning("HL SL mirror skipped: no size")
        return

    router = get_router()
    oid = str(row["hl_sl_oid"])
    _logger.info("HL mirror SL UPDATE: trigger=%s size=%s side=%s", new_stop_price, sz, side)
    result = router.set_stop_loss(oid, new_stop_price, sz, side)

    if result.get("ok"):
        new_oid = result.get("order_id")
        if new_oid and new_oid != oid:
            conn.execute(
                "UPDATE positions SET hl_sl_oid=? WHERE id=? AND hl_enabled=1",
                (str(new_oid), position["id"]),
            )
            conn.commit()
            record_hl_order(position["id"], new_oid, "sl", size=sz, price=new_stop_price)
        else:
            # oid unchanged — modify succeeded in-place, record price update
            update_hl_order_price(oid, new_stop_price, new_size=sz)
        _logger.info("HL SL updated: oid=%s price=%s", new_oid, new_stop_price)
    else:
        _logger.warning("HL SL mirror update FAILED: %s", result.get("error"))


def register_callbacks():
    """Register TradeRouter mirror hooks into the simulated position manager."""
    from app.simulated_position_manager import _register_auto_hooks

    _register_auto_hooks(
        open_hooks=[_hl_mirror_open],
        close_hooks=[_hl_mirror_close],
        reduce_hooks=[_hl_mirror_reduce],
        add_hooks=[_hl_mirror_add],
        stop_update_hooks=[_hl_mirror_stop_update],
    )

    _logger.info("Auto-trader callbacks registered (enabled=%s, backend=%s)",
                 settings.auto_trade_enabled, settings.trade_backend)


def _hl_startup_sync():
    """Startup sync: reconcile HL position state with DB.

    Handles 4 drift scenarios:
      1. DB open + HL no position → DB was closed externally or crashed before commit.
         Mark DB as closed using current price (PnL will be approximate).
      2. DB open + HL has position → normal, no action needed.
      3. DB closed + HL has position → orphan HL exposure. Force close it.
      4. DB closed + HL no position → normal, no action needed.

    Only runs when Hyperliquid backend is enabled. Non-blocking: all errors
    logged but never abort startup.
    """
    if settings.trade_backend != "hyperliquid" or not settings.auto_trade_enabled:
        return

    from app.simulated_position_manager import close_simulated_position
    from app.database import get_connection

    try:
        router = get_router()
        hl_pos, hl_err = router.get_position_state()
        hl_has_position = bool(hl_pos and hl_pos.get("size"))

        conn = get_connection()
        db_open = conn.execute(
            "SELECT * FROM positions WHERE status='open' AND is_simulated=1 "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        if db_open:
            if hl_has_position:
                # Scenario 2: both open — check sizes match
                db_size = db_open.get("position_size") or 0
                hl_size = hl_pos.get("size", 0)
                if abs(db_size - hl_size) > 0.001:
                    _logger.warning("Startup sync: size mismatch DB=%s vs HL=%s, updating DB",
                                    db_size, hl_size)
                    conn.execute(
                        "UPDATE positions SET position_size=?, hl_sz=? WHERE id=?",
                        (round(hl_size, 5), round(hl_size, 5), db_open["id"]),
                    )
                    conn.commit()
                _logger.info("Startup sync: DB open + HL open (size=%s), OK", hl_size)
            else:
                # Scenario 1: DB says open but HL has no position.
                # Query fills to find the actual closing trade for accurate PnL.
                from app.hyperliquid_viewer import get_viewer
                entry = dict(db_open).get("entry_price", 0)
                side = dict(db_open).get("side", "")
                entry_time = dict(db_open).get("created_at", 0) or 0
                position_size = dict(db_open).get("position_size", 0) or dict(db_open).get("hl_sz", 0)
                close_side = "short" if side == "long" else "long"
                close_price = None
                close_pnl = None
                close_reason = "startup_sync: hl_position_not_found"

                try:
                    viewer = get_viewer()
                    # Fills are sorted newest-first. The close fill must appear
                    # after (i.e., before in the list) the open fill. Since there
                    # are rarely many intervening trades, a small window suffices.
                    for page in range(1, 4):
                        fills_data = viewer.get_fills(page=page, size=10)
                        fills = fills_data.get("fills", [])
                        if not fills:
                            break
                        for f in fills:
                            # get_fills returns "timestamp", not "time"
                            fill_time = f.get("timestamp") or 0
                            if fill_time and fill_time < entry_time * 1000:
                                break  # passed entry time, no more candidates
                            if f.get("side", "") != close_side:
                                continue
                            close_price = f.get("price") or 0
                            close_pnl = f.get("closed_pnl")
                            if close_price and close_price > 0:
                                close_reason = "startup_sync: hl_fill_found"
                                _logger.info("Startup sync: found closing fill: price=%s pnl=%s", close_price, close_pnl)
                                break
                        if close_price and close_price > 0:
                            break
                except Exception:
                    _logger.exception("Startup sync: failed to query fills")

                if not close_price or close_price <= 0:
                    close_price = entry  # fallback to entry price

                if close_pnl is not None:
                    pnl = close_pnl
                elif position_size > 0 and entry > 0:
                    pnl = position_size * (entry - close_price) if side == "short" else position_size * (close_price - entry)
                else:
                    pnl = 0

                now = time.time()
                conn.execute(
                    "UPDATE positions SET status='closed', pnl=?, close_reason=?, "
                    "close_price=?, closed_at=?, updated_at=? WHERE id=? AND status='open'",
                    (round(pnl, 2), close_reason, round(close_price, 1), now, now, db_open["id"]),
                )
                conn.commit()
                _logger.warning("Startup sync: DB open but no HL position — closed id=%s price=%s pnl=%s",
                                db_open["id"], close_price, pnl)
        else:
            if hl_has_position:
                # Scenario 3: orphan HL exposure — force close it.
                hl_size = hl_pos.get("size", 0)
                _logger.error("Startup sync: orphan HL position detected (size=%s, side=%s), force closing",
                              hl_size, hl_pos.get("side"))
                result = router.close(round(hl_size, 5))
                if result.get("ok"):
                    _logger.info("Startup sync: orphan HL position closed, oid=%s fill=%s",
                                 result.get("order_id"), result.get("fill_price"))
                else:
                    _logger.error("Startup sync: orphan HL close failed: %s", result.get("error"))
            else:
                # Scenario 4: both closed — normal
                _logger.info("Startup sync: DB closed + HL closed, OK")

        conn.close()
    except Exception:
        _logger.exception("Startup sync failed (non-fatal)")
    finally:
        try:
            conn.close()
        except Exception:
            pass

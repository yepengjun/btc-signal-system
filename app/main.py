"""BTC Signal System — FastAPI main entry point."""

import copy
import logging
import threading
import time
from datetime import datetime

from typing import Optional

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.database import init_db, get_connection
from app.auth import create_session, verify_session, authenticate
from app.signal_engine import generate_verdict
from app.binance import fetch_klines, fetch_price, fetch_funding_rate, fetch_open_interest, fetch_all_market_data
from app.evolution import get_evolution_stats, verify_pending_signals, get_active_thresholds
from app.trade_executor import get_router
from app.hyperliquid_viewer import get_viewer
from app.simulated_position_manager import (
    manage_simulated_position, _check_price_based_exit,
    _check_pnl_trailing_stop, _register_auto_hooks,
)

# Configure app logger to output to stdout (uvicorn doesn't configure __name__ loggers)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# 信号响应缓存，避免每次请求都请求 Binance
_cache: dict = {"data": None, "ts": 0.0}
_rebuild_lock = threading.Lock()


def _get_cached_signals() -> Optional[dict]:
    """如果缓存未过期则返回缓存数据。"""
    if _cache["data"] and (time.time() - _cache["ts"]) < settings.signal_interval_seconds:
        return _cache["data"]
    return None


def _set_cached_signals(data: dict):
    _cache["data"] = data
    _cache["ts"] = time.time()


def _refresh_live_prices(data: dict, current_price: float = None):
    """从 Binance 拉取最新价格，更新实时字段。

    Args:
        data: signal response dict to update
        current_price: optional pre-fetched price to avoid redundant API call
    """
    if current_price is None:
        try:
            current_price = fetch_price(settings.binance_symbol)
        except Exception as e:
            logger.warning(f"fetch_price failed: {e}")
            return

    if not current_price:
        return

    try:
        # 更新 ticker 价格
        data["ticker"]["price"] = current_price
        data["verdict"]["price_at_signal"] = current_price

        # NOTE: funding_rate is NOT updated here — the cached verdict already has
        # funding_rate as percentage from the last full generate_verdict call.
        # _refresh_live_prices must NOT overwrite it with raw decimal, since the
        # signal engine expects raw decimal in market_context but the dashboard
        # expects percentage in the display. Funding rate has a 5-min cache TTL,
        # so the display value stays reasonably fresh.

        # 更新 OI 及变化率（缓存命中时也需要刷新）
        current_oi, prev_oi = fetch_open_interest(settings.binance_symbol)
        mc = data["verdict"].get("market_context", {})
        if current_oi and mc:
            mc["open_interest"] = current_oi
            mc["open_interest_prev"] = prev_oi

        # 更新 30m 时间框架价格
        if "30m" in data["timeframes"]:
            data["timeframes"]["30m"]["price"] = current_price

        # 更新入场时机的区间和百分位 — 拉最新 20 根 30m K线重新计算区间
        try:
            candles = fetch_klines(settings.binance_symbol, "30m", limit=20)
        except Exception as e:
            logger.warning(f"fetch_klines 30m failed: {e}")
            candles = None

        entry = data["verdict"].get("entry_timing", {})
        if entry and candles and len(candles) >= 20:
            highs = [c["high"] for c in candles[-20:]]
            lows = [c["low"] for c in candles[-20:]]
            range_low = min(lows)
            range_high = max(highs)
            entry["range_low"] = round(range_low, 1)
            entry["range_high"] = round(range_high, 1)
            if range_high > range_low:
                entry["percentile"] = round(
                    (current_price - range_low) / (range_high - range_low) * 100, 1
                )
            entry["short_move_pct"] = round(
                (current_price - highs[-1]) / max(highs[-1], 1) * 100, 3
            )

        # 更新下单信号的入场价和风险指标
        # entry_price = 现价（实时），止盈止损 = 固定结构位，盈亏比 = 按计划入场价算
        order = data["verdict"].get("order_signal", {})
        if order and order.get("side") not in (None, "观望"):
            stop = order.get("stop")
            target = order.get("target")
            # Use planned_entry_price if available (preserves original plan across cache hits)
            planned_entry = order.get("planned_entry_price") or order.get("entry_price")
            # Show current BTC price as entry_price
            order["entry_price"] = current_price
            if stop is not None and target is not None and planned_entry is not None:
                # R/R calculated based on planned entry price
                if order["side"] == "做多":
                    order["risk"] = round(abs(planned_entry - stop), 1)
                    order["reward"] = round(abs(target - planned_entry), 1)
                else:
                    order["risk"] = round(abs(stop - planned_entry), 1)
                    order["reward"] = round(abs(planned_entry - target), 1)
                if order.get("risk", 0) > 0:
                    order["rr_ratio"] = round(order["reward"] / order["risk"], 2)
                    if not order.get("position_pct"):
                        order["position_pct"] = round(2.0 / max(order["rr_ratio"], 1), 1)

        # Update advice target/stop based on current price (structural levels stay fixed,
        # but we recalculate for display consistency)
        advice = data["verdict"].get("advice", {})
        if advice.get("side") in ("多", "空"):
            adv_stop = advice.get("stop")
            adv_target = advice.get("target")
            if adv_stop is not None and adv_target is not None:
                if advice["side"] == "多":
                    advice["risk"] = round(abs(current_price - adv_stop), 1)
                    advice["reward"] = round(abs(adv_target - current_price), 1)
                else:
                    advice["risk"] = round(abs(adv_stop - current_price), 1)
                    advice["reward"] = round(abs(current_price - adv_target), 1)

        # Update hold_long/hold_short advice with current price context
        for key in ("hold_long", "hold_short"):
            hold = data["verdict"].get(key, {})
            if hold.get("stop") is not None:
                hold_stop = hold["stop"]
                hold_target = hold.get("target")
                side = key.replace("hold_", "")  # "long" or "short"
                if side == "long":
                    hold["risk"] = round(abs(current_price - hold_stop), 1)
                    if hold_target:
                        hold["reward"] = round(abs(hold_target - current_price), 1)
                else:
                    hold["risk"] = round(abs(hold_stop - current_price), 1)
                    if hold_target:
                        hold["reward"] = round(abs(current_price - hold_target), 1)
    except Exception as e:
        logger.warning(f"_refresh_live_prices update failed: {e}")


@app.on_event("startup")
async def startup():
    init_db()
    # 从 SQLite 加载历史 K 线到内存缓存，实现增量续接
    from app.binance import load_klines_from_db
    load_klines_from_db(settings.binance_symbol)
    # 后台预热缓存，避免首次访问慢
    import threading
    def _warm_cache():
        try:
            _build_signals_response(force_refresh=True)
        except Exception:
            pass  # 预热失败不影响正常启动
    threading.Thread(target=_warm_cache, daemon=True).start()

    # Register auto-trader callbacks (mirrors simulated positions to Hyperliquid)
    from app.auto_trader import register_callbacks
    register_callbacks()

    # Startup sync: reconcile Hyperliquid position state with DB.
    # Runs once at boot to catch drift from crashes, passive fills, etc.
    from app.auto_trader import _hl_startup_sync
    _hl_startup_sync()

    # 后台监控模拟仓位，独立于信号缓存（每 10 秒检查一次止盈/止损）
    def _monitor_sim_positions():
        while True:
            try:
                time.sleep(10)
                _check_sim_position_price_trigger()
            except Exception:
                pass  # 监控失败不影响主流程

    def _check_sim_position_price_trigger():
        """从 Binance 拉取实时价格，检查止盈/止损。

        不负责开新仓——开仓由信号周期（API 请求）生成新 verdict 后处理。
        止损后用旧信号立即开仓是危险行为。
        """
        current_price = None
        try:
            current_price = fetch_price(settings.binance_symbol)
        except Exception:
            pass
        if not current_price:
            try:
                current_price = fetch_price(settings.binance_symbol)
            except Exception:
                return  # 两次获取失败，跳过本次检查
        if not current_price:
            return

        conn = get_connection()
        try:
            sim_pos_row = conn.execute(
                "SELECT * FROM positions WHERE status='open' AND is_simulated=1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

            if sim_pos_row:
                pos = dict(sim_pos_row)
                # Check price-based exits (stop-loss / take-profit)
                exited = _check_price_based_exit(conn, pos, current_price)
                if not exited:
                    # Also run PnL trailing stop — this was previously only executed
                    # during the signal cycle (every 300s), missing profitable windows
                    _check_pnl_trailing_stop(conn, pos, current_price)
                    # Re-check status since trailing stop may have closed it
                    if pos.get("status") != "open":
                        conn.close()
                        return

                # Track price extremes for add-on detection (_is_price_extreme)
                # Only if position is still open (exits above may have closed it)
                if pos.get("status") != "open":
                    conn.close()
                    return

                side = pos.get("side")
                max_p = pos.get("max_price")
                min_p = pos.get("min_price")
                price_updates = {}
                if side == "long" and (max_p is None or current_price > max_p):
                    price_updates["max_price"] = round(current_price, 1)
                if side == "short" and (min_p is None or current_price < min_p):
                    price_updates["min_price"] = round(current_price, 1)
                if price_updates:
                    set_clause = ", ".join(f"{k}=?" for k in price_updates)
                    conn.execute(
                        f"UPDATE positions SET {set_clause} WHERE id=?",
                        tuple(price_updates.values()) + (pos["id"],),
                    )
                    conn.commit()
            # No open position: do NOT auto-open here. Wait for next signal cycle.
        finally:
            conn.close()

    threading.Thread(target=_monitor_sim_positions, daemon=True).start()

    # 后台 K 线同步任务 —— 每 15 秒从 Binance 拉取各周期 K 线写入数据库，
    # 保证信号验证等场景有最新数据可用，不依赖实时网络请求。
    def _sync_klines_to_db():
        while True:
            try:
                time.sleep(30)
                for tf in ("30m", "1h", "4h"):
                    fetch_klines(settings.binance_symbol, tf, limit=200)
            except Exception:
                pass  # 同步失败不影响主流程，下次周期自动重试

    threading.Thread(target=_sync_klines_to_db, daemon=True).start()

    def _sync_hl_position():
        """Reconcile Hyperliquid exchange position state with local DB.

        Detects passive TP/SL fills (HL executes the limit/stop order
        independently) or manual closes on the HL platform. When the HL
        exchange shows no position but local DB has an open HL position,
        we query fills to find the closing trade, extract the actual fill
        price and closedPnl, and update the local DB accordingly.
        """
        # Log on first run to confirm thread is alive
        if not hasattr(_sync_hl_position, "_logged"):
            logger.info("[HL sync] _sync_hl_position thread active (auto_trade=%s)", settings.auto_trade_enabled)
            _sync_hl_position._logged = True
        if not settings.auto_trade_enabled:
            return

        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM positions WHERE status='open' AND hl_enabled=1 "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return  # no open HL position to sync

            pos = dict(row)
            logger.info("[HL sync] checking position #%s (side=%s, size=%s)", pos["id"], pos.get("side"), pos.get("position_size"))

            # Check actual HL position state
            try:
                hl_pos, hl_err = get_router().get_position_state()
            except Exception:
                logger.exception("[HL sync] get_position_state failed")
                return  # HL query failed, try next cycle

            if hl_err is not None:
                logger.warning("[HL sync] HL API error: %s", hl_err)
                return  # HL API error — don't misinterpret as "no position"

            if hl_pos:
                # HL still has a position — check if size matches
                local_size = pos.get("position_size") or 0
                hl_size = hl_pos.get("size") or 0
                # Allow small float deviation (rounding differences)
                if abs(local_size - hl_size) > 0.0001:
                    # Size mismatch: HL was partially closed outside our flow,
                    # or an add/reduce failed to execute on HL.
                    # Update hl_sz and resize TP/SL to match exchange reality
                    logger.info(
                        "[HL sync] size mismatch: local=%s hl=%s → updating hl_sz + position_size + TP/SL",
                        local_size, hl_size,
                    )
                    conn.execute(
                        "UPDATE positions SET hl_sz=?, position_size=? WHERE id=? AND hl_enabled=1",
                        (round(hl_size, 5), round(hl_size, 5), pos["id"]),
                    )
                    conn.commit()

                    # Resize TP/SL orders to match actual HL size
                    side = pos.get("side", "")
                    tp_price = pos.get("target")
                    sl_price = pos.get("stop")
                    tp_oid = pos.get("hl_tp_oid")
                    sl_oid = pos.get("hl_sl_oid")
                    actual_sz = round(hl_size, 5)

                    if actual_sz > 0 and side:
                        router = get_router()

                        if tp_oid and tp_oid != "None" and tp_price:
                            tp_result = router.set_take_profit(str(tp_oid), tp_price, actual_sz, side)
                            if tp_result.get("ok"):
                                logger.info("[HL sync] TP resized: size=%s", actual_sz)

                        if sl_oid and sl_oid != "None" and sl_price:
                            sl_result = router.set_stop_loss(str(sl_oid), sl_price, actual_sz, side)
                            if sl_result.get("ok"):
                                logger.info("[HL sync] SL resized: size=%s", actual_sz)

                return

            # HL shows no position but local DB says open → passive close detected
            logger.info("[HL sync] HL has no position but local #%s is open — searching for closing fill (close_side=%s)", pos["id"], "short" if pos.get("side") == "long" else "long")
            side = pos.get("side", "")
            close_side = "short" if side == "long" else "long"  # opposite of position side
            entry_time = pos.get("created_at") or 0

            closing_fill = None
            viewer = get_viewer()
            for page in range(1, 4):  # search up to 30 fills
                fills_data = viewer.get_fills(page=page, size=10)
                fills = fills_data.get("fills", [])
                if not fills:
                    break

                for f in fills:
                    fill_time = f.get("timestamp") or 0
                    # Fill is before entry time — fills are sorted newest first
                    if fill_time and fill_time < entry_time * 1000:
                        break

                    fill_side = f.get("side", "")
                    if fill_side != close_side:
                        continue

                    closing_fill = f
                    break

                if closing_fill:
                    break

            if not closing_fill:
                logger.warning("[HL sync] no matching closing fill found (side=%s, entry_time=%s) — likely query error, skipping sync", close_side, entry_time)
                return

            actual_price = closing_fill.get("price") or 0
            closed_pnl = closing_fill.get("closed_pnl") or 0

            if actual_price <= 0:
                logger.warning("[HL sync] closing fill has invalid price=%s", actual_price)
                return

            # Determine close reason from context
            # If price hit stop → stop_loss, if price hit target → take_profit
            stop = pos.get("stop")
            target = pos.get("target")
            close_reason = "hl_passive_close"
            if stop is not None:
                if side == "long" and actual_price <= stop * 1.005:  # 0.5% tolerance
                    close_reason = "stop_loss"
                elif side == "short" and actual_price >= stop * 0.995:
                    close_reason = "stop_loss"
            if target is not None:
                if side == "long" and actual_price >= target * 0.995:
                    close_reason = "take_profit"
                elif side == "short" and actual_price <= target * 1.005:
                    close_reason = "take_profit"

            # Always compute PnL locally using the authoritative position size.
            # hl_sz reflects what's actually on HL (updated by sync on mismatch),
            # while position_size may be stale. Prefer hl_sz when available.
            entry = pos.get("entry_price") or 0
            btc_qty = pos.get("hl_sz") or pos.get("position_size") or 0
            if entry > 0 and btc_qty > 0:
                if side == "long":
                    pnl_to_record = btc_qty * (actual_price - entry)
                else:
                    pnl_to_record = btc_qty * (entry - actual_price)
            else:
                pnl_to_record = closed_pnl  # fallback to HL-reported

            now = time.time()
            conn.execute(
                "UPDATE positions SET status='closed', pnl=?, close_reason=?, "
                "close_price=?, closed_at=?, updated_at=? WHERE id=? AND status='open'",
                (round(pnl_to_record, 2), close_reason, round(actual_price, 1), now, now, pos["id"]),
            )
            if conn.execute("SELECT changes()").fetchone()[0] == 0:
                return  # already closed by another path
            conn.commit()

            logger.info(
                "[HL sync] passive close reconciled: side=%s price=%s pnl=%.2f reason=%s",
                side, actual_price, pnl_to_record, close_reason,
            )
        finally:
            conn.close()

    # Background HL position sync: detect passive TP/SL fills or manual HL closes
    # and reconcile local DB with exchange state.
    if settings.auto_trade_enabled:
        def _hl_position_sync():
            while True:
                try:
                    time.sleep(15)
                    _sync_hl_position()
                except Exception:
                    logger.exception("[HL sync] background sync cycle failed")

        threading.Thread(target=_hl_position_sync, daemon=True).start()
        logger.info("[HL sync] background threads started")

        # Background HL position reconciliation: detect simulated positions
        # that haven't been mirrored to HL yet (hl_enabled=0) and open them.
        def _hl_position_reconcile():
            """Reconcile HL simulated positions every 30s.

            Detects positions where the local simulated DB was created but
            the HL mirror failed to open (hl_enabled=0). This can happen
            if _hl_mirror_open crashed after the INSERT.
            """
            while True:
                try:
                    time.sleep(30)
                    conn = get_connection()
                    try:
                        rows = conn.execute(
                            "SELECT * FROM positions WHERE status='open' "
                            "AND is_simulated=1 AND hl_enabled=0 "
                            "ORDER BY created_at ASC LIMIT 5"
                        ).fetchall()
                    finally:
                        conn.close()

                    for row in rows:
                        pos = dict(row)
                        pid = pos["id"]
                        side = pos.get("side", "")
                        sz = pos.get("position_size")
                        leverage = pos.get("leverage", 20)
                        tp = pos.get("target")
                        sl = pos.get("stop")

                        if not sz or not side or side not in ("long", "short"):
                            continue

                        router = get_router()

                        # Guard: check if HL already has a position before opening.
                        # Can happen if _extract_oid failed but market_open actually filled.
                        try:
                            hl_pos, hl_err = router.get_position_state()
                            if hl_pos and hl_pos.get("size"):
                                hl_size = abs(float(hl_pos.get("size", 0)))
                                db_size = float(sz)
                                # If HL already has a position of similar size, just link it
                                if abs(hl_size - db_size) < 0.01:
                                    logger.info(
                                        "[HL reconcile] HL already has position (size=%s vs db=%s), linking without re-opening: id=%s",
                                        hl_size, db_size, pid,
                                    )
                                    conn.execute(
                                        "UPDATE positions SET hl_enabled=1, hl_sz=?, hl_entry_oid='reconcile_linked' "
                                        "WHERE id=? AND hl_enabled=0",
                                        (round(hl_size, 5), pid),
                                    )
                                    conn.commit()
                                    continue
                                else:
                                    logger.warning(
                                        "[HL reconcile] HL position size mismatch (hl=%s vs db=%s), may need manual review: id=%s",
                                        hl_size, db_size, pid,
                                    )
                        except Exception:
                            pass  # best-effort guard, proceed with open if check fails

                        logger.info(
                            "[HL reconcile] opening mirrored position: id=%s side=%s size=%s",
                            pid, side, sz,
                        )
                        result = router.open(side, round(float(sz), 5), int(leverage), tp, sl)

                        if result.get("ok"):
                            tp_oid = result.get("tp_oid")
                            sl_oid = result.get("sl_oid")
                            conn.execute(
                                "UPDATE positions SET hl_enabled=1, hl_sz=?, hl_entry_oid=? "
                                f", hl_tp_oid={'?' if tp_oid else 'NULL'}, hl_sl_oid={'?' if sl_oid else 'NULL'} "
                                "WHERE id=? AND hl_enabled=0",
                                (
                                    round(float(sz), 5),
                                    str(result.get("order_id")),
                                    str(tp_oid) if tp_oid else None,
                                    str(sl_oid) if sl_oid else None,
                                    pid,
                                ),
                            )
                            # Record action state (if not already recorded by initial open)
                            conn.execute(
                                "INSERT INTO position_action_state (position_id, action, price, position_size, created_at) "
                                "VALUES (?, 'open', ?, ?, ?)",
                                (pid, round(float(pos.get("entry_price") or 0), 1), round(float(sz), 5), time.time()),
                            )
                            conn.commit()
                            logger.info(
                                "[HL reconcile] position mirrored: id=%s entry_oid=%s",
                                pid, result.get("order_id"),
                            )
                        else:
                            from app.auto_trader import _is_retryable_error
                            error = result.get("error", "")
                            if _is_retryable_error(str(error)):
                                logger.warning(
                                    "[HL reconcile] retryable error (will retry next cycle): id=%s error=%s",
                                    pid, error,
                                )
                            else:
                                logger.error(
                                    "[HL reconcile] hard failure (deleting position): id=%s error=%s",
                                    pid, error,
                                )
                                conn.execute("DELETE FROM positions WHERE id=?", (pid,))
                                conn.execute("DELETE FROM position_action_state WHERE position_id=?", (pid,))
                                conn.commit()
                except Exception:
                    pass

        threading.Thread(target=_hl_position_reconcile, daemon=True).start()

        # Background HL TP/SL order reconciliation: detect missing or mismatched orders and recreate.
        def _hl_order_reconcile():
            """Reconcile HL TP/SL orders every 30s.

            Detects orders that failed to place on open, or were cancelled
            accidentally. Also handles passive fills where the oid was
            consumed but the position still exists in DB. Additionally checks
            that existing order sizes match hl_sz and recreates mismatches.
            """
            last_attempt: dict = {}  # position_id -> last retry timestamp
            while True:
                try:
                    time.sleep(30)
                    conn = get_connection()
                    try:
                        rows = conn.execute(
                            "SELECT * FROM positions WHERE status='open' AND hl_enabled=1"
                        ).fetchall()
                    finally:
                        conn.close()

                    for row in rows:
                        pos = dict(row)
                        pid = pos["id"]
                        now = time.time()
                        last = last_attempt.get(pid, 0)
                        if now - last < 60:
                            continue

                        sl_oid = pos.get("hl_sl_oid")
                        tp_oid = pos.get("hl_tp_oid")
                        sl_price = pos.get("stop")
                        tp_price = pos.get("target")
                        sz = pos.get("position_size") or pos.get("hl_sz")
                        side = pos.get("side", "")

                        need_sl = sl_price and (not sl_oid or sl_oid == "None")
                        need_tp = tp_price and (not tp_oid or tp_oid == "None")

                        router = get_router()

                        # Fetch open orders once per position
                        open_orders = []
                        try:
                            open_orders = router.get_open_orders()
                        except Exception:
                            pass

                        # Verify SL: exists AND size matches hl_sz
                        if sl_oid and sl_oid != "None":
                            sl_order = next((o for o in open_orders if str(o.get("oid")) == str(sl_oid)), None)
                            if not sl_order:
                                logger.warning(
                                    "[HL reconcile] SL oid=%s not found on exchange, recreating",
                                    sl_oid,
                                )
                                need_sl = True
                                conn.execute(
                                    "UPDATE positions SET hl_sl_oid=NULL WHERE id=?",
                                    (pid,),
                                )
                                conn.commit()
                            elif sz and sl_order.get("sz"):
                                order_sz = float(sl_order["sz"])
                                expected_sz = round(float(sz), 5)
                                if abs(order_sz - expected_sz) > 0.0001:
                                    logger.info(
                                        "[HL reconcile] SL size mismatch: order=%s expected=%s, canceling + recreating",
                                        order_sz, expected_sz,
                                    )
                                    router.cancel_order(str(sl_oid))
                                    need_sl = True
                                    conn.execute(
                                        "UPDATE positions SET hl_sl_oid=NULL WHERE id=?",
                                        (pid,),
                                    )
                                    conn.commit()

                        # Verify TP: exists AND size matches hl_sz
                        if tp_oid and tp_oid != "None":
                            tp_order = next((o for o in open_orders if str(o.get("oid")) == str(tp_oid)), None)
                            if not tp_order:
                                logger.warning(
                                    "[HL reconcile] TP oid=%s not found on exchange, recreating",
                                    tp_oid,
                                )
                                need_tp = True
                                conn.execute(
                                    "UPDATE positions SET hl_tp_oid=NULL WHERE id=?",
                                    (pid,),
                                )
                                conn.commit()
                            elif sz and tp_order.get("sz"):
                                order_sz = float(tp_order["sz"])
                                expected_sz = round(float(sz), 5)
                                if abs(order_sz - expected_sz) > 0.0001:
                                    logger.info(
                                        "[HL reconcile] TP size mismatch: order=%s expected=%s, canceling + recreating",
                                        order_sz, expected_sz,
                                    )
                                    router.cancel_order(str(tp_oid))
                                    need_tp = True
                                    conn.execute(
                                        "UPDATE positions SET hl_tp_oid=NULL WHERE id=?",
                                        (pid,),
                                    )
                                    conn.commit()

                        # Recreate missing orders
                        if need_sl and sl_price and sz:
                            result = router.set_stop_loss("", sl_price, round(float(sz), 5), side)
                            if result.get("ok"):
                                conn.execute(
                                    "UPDATE positions SET hl_sl_oid=? WHERE id=?",
                                    (str(result["order_id"]), pid),
                                )
                                conn.commit()
                                logger.info(
                                    "[HL reconcile] SL recreated: oid=%s price=%s",
                                    result["order_id"], sl_price,
                                )
                            else:
                                logger.warning(
                                    "[HL reconcile] SL recreate failed: %s",
                                    result.get("error"),
                                )

                        if need_tp and tp_price and sz:
                            result = router.set_take_profit("", tp_price, round(float(sz), 5), side)
                            if result.get("ok"):
                                conn.execute(
                                    "UPDATE positions SET hl_tp_oid=? WHERE id=?",
                                    (str(result["order_id"]), pid),
                                )
                                conn.commit()
                                logger.info(
                                    "[HL reconcile] TP recreated: oid=%s price=%s",
                                    result["order_id"], tp_price,
                                )
                            else:
                                logger.warning(
                                    "[HL reconcile] TP recreate failed: %s",
                                    result.get("error"),
                                )

                        if need_sl or need_tp:
                            last_attempt[pid] = now
                except Exception:
                    pass

        threading.Thread(target=_hl_order_reconcile, daemon=True).start()

        # Background HL close reconciliation: detect positions where local DB
        # was marked closed but HL position still exists (HL close failed).
        def _hl_close_reconcile():
            """Reconcile HL close state every 60s.

            Catches the asymmetric failure: local DB status='closed' but
            Hyperliquid position still open (e.g. HL close API timeout after
            DB commit, or network failure during _hl_mirror_close).
            """
            reconciled: set = set()  # position IDs already reconciled
            while True:
                try:
                    time.sleep(60)
                    conn = get_connection()
                    try:
                        # Check recently closed HL positions (last 10 minutes)
                        cutoff = time.time() - 600
                        rows = conn.execute(
                            "SELECT * FROM positions WHERE status='closed' "
                            "AND hl_enabled=1 AND closed_at >= ? "
                            "ORDER BY closed_at DESC LIMIT 10",
                            (cutoff,),
                        ).fetchall()
                    finally:
                        conn.close()

                    for row in rows:
                        pos = dict(row)
                        pid = pos["id"]
                        if pid in reconciled:
                            continue

                        # Check if HL still has an open position
                        try:
                            hl_pos, hl_err = get_router().get_position_state()
                        except Exception:
                            continue  # HL query failed, retry next cycle

                        if hl_pos and hl_pos.get("size") and hl_pos["size"] > 0:
                            # HL still has position → force close it
                            logger.warning(
                                "[HL close reconcile] local DB closed but HL position still open: id=%s size=%s → force closing",
                                pid, hl_pos["size"],
                            )
                            result = get_router().close(round(float(hl_pos["size"]), 5))
                            if result.get("ok"):
                                fill_price = result.get("fill_price")
                                if fill_price and fill_price > 0:
                                    entry = pos.get("entry_price") or 0
                                    side = pos.get("side", "")
                                    sz = float(hl_pos["size"])
                                    if entry > 0:
                                        actual_pnl = sz * (fill_price - entry) if side == "long" else sz * (entry - fill_price)
                                        conn = get_connection()
                                        try:
                                            conn.execute(
                                                "UPDATE positions SET pnl=?, close_price=? WHERE id=?",
                                                (round(actual_pnl, 2), round(fill_price, 1), pid),
                                            )
                                            conn.commit()
                                            logger.info(
                                                "[HL close reconcile] PnL updated: fill=%s pnl=%s (was %s)",
                                                fill_price, actual_pnl, pos.get("pnl"),
                                            )
                                        finally:
                                            conn.close()
                                logger.info("[HL close reconcile] force close succeeded: id=%s", pid)
                            else:
                                logger.warning("[HL close reconcile] force close failed: id=%s error=%s", pid, result.get("error"))

                        # Mark as reconciled regardless — if HL was already
                        # closed (passive TP/SL fill) no action needed.
                        reconciled.add(pid)
                except Exception:
                    pass

        threading.Thread(target=_hl_close_reconcile, daemon=True).start()


# ---------- Pages ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    token = request.cookies.get("session")
    username = verify_session(token) if token else None
    if username:
        return templates.TemplateResponse("dashboard.html", {"request": request, "username": username, "sim_initial_balance": settings.sim_initial_balance})
    return templates.TemplateResponse("login.html", {"request": request})


# ---------- Auth ----------

@app.post("/login")
async def login(request: Request, response: Response):
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")
    if authenticate(username, password):
        token = create_session(username)
        resp = RedirectResponse(url="/", status_code=302)
        resp.set_cookie("session", token, httponly=True, samesite="strict", max_age=86400 * 7)
        return resp
    return JSONResponse({"detail": "用户名或密码错误"}, status_code=401)


@app.post("/logout")
async def logout(response: Response):
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie("session")
    return resp


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cookie_header = ws.headers.get("cookie", "")
    token = None
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith("session="):
            token = part.split("=", 1)[1]
            break
    if not verify_session(token):
        await ws.close(code=1008)
        return
    import asyncio
    while True:
        try:
            now = time.time()
            # Full verdict refresh every SIGNAL_INTERVAL seconds
            data = _build_signals_response(force_refresh=True)
            await ws.send_json(data)
            await asyncio.sleep(settings.signal_interval_seconds)
        except (WebSocketDisconnect, RuntimeError):
            break
        except Exception as e:
            logger.warning(f"WS signal generation failed: {e}")
            # On failure, try to send cached data to keep connection alive
            try:
                stale = _get_cached_signals()
                if stale:
                    await ws.send_json(stale)
                await asyncio.sleep(settings.signal_interval_seconds)
            except (WebSocketDisconnect, RuntimeError):
                break


def require_auth(request: Request) -> Optional[str]:
    token = request.cookies.get("session")
    return verify_session(token) if token else None


def _build_signals_response(now: float = None, force_refresh: bool = False) -> dict:
    """Build full signals response with verdict + history + snapshots."""
    # Check cache (WebSocket can force refresh)
    if not force_refresh:
        cached = _get_cached_signals()
        if cached:
            return _build_cache_path_response(cached)

    # Cache miss or forced refresh: rebuild with lock to prevent concurrent rebuilds
    with _rebuild_lock:
        # Double-check after acquiring lock
        if not force_refresh:
            cached = _get_cached_signals()
            if cached:
                return _build_cache_path_response(cached)
        return _rebuild_signals_response(now)


def _build_cache_path_response(cached: dict) -> dict:
    """Cache hit path: deep-copy verdict and refresh live price."""
    conn = get_connection()
    try:
        # Deep-copy cached data, refresh price BEFORE position management
        data = copy.deepcopy(cached["data"])

        # Reuse verdict_history and m30_snapshots from cached data —
        # these don't change between cache hits (only written on full rebuild).
        if "verdict_history" not in data:
            history_rows_raw = conn.execute(
                "SELECT * FROM verdict_history ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            history_rows = []
            for r in history_rows_raw:
                history_rows.append({
                    "regime": r["regime"], "direction": r["direction"],
                    "strength": r["strength"], "confidence": r["confidence"],
                    "momentum": r["momentum"], "advice": r["advice"],
                    "price": r["price"], "adx_4h": r["adx_4h"], "adx_1h": r["adx_1h"],
                    "dir_4h": r["dir_4h"], "dir_1h": r["dir_1h"],
                    "created_at": r["created_at"],
                })
            verdict_history = []
            for r in history_rows:
                dt = datetime.fromtimestamp(r.get("created_at", time.time())).strftime("%m-%d %H:%M")
                verdict_history.append({**r, "time": dt})
            deduped = []
            for h in verdict_history:
                if not deduped or deduped[-1]["regime"] != h["regime"] or deduped[-1]["direction"] != h["direction"]:
                    deduped.append(h)
            data["verdict_history"] = deduped

        if "m30_snapshots" not in data:
            snapshots = conn.execute(
                "SELECT regime, direction, adx, plus_di, minus_di, price_at_signal, created_at "
                "FROM signals WHERE timeframe = '30m' ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            m30_snapshots = []
            for s in snapshots:
                dt = datetime.fromtimestamp(s["created_at"]).strftime("%H:%M:%S")
                m30_snapshots.append({
                    "ts": s["created_at"], "time": dt,
                    "price": s["price_at_signal"], "adx": s["adx"],
                    "plus_di": s["plus_di"], "minus_di": s["minus_di"],
                    "regime": s["regime"], "direction": s["direction"],
                })
            data["m30_snapshots"] = m30_snapshots

        # Get open position for state machine (manual only)
        open_position = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' AND is_simulated = 0 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        position_dict = dict(open_position) if open_position else None

        # Fetch live price first — manage_simulated_position needs current
        # price for accurate stop-loss/take-profit checks. Retry once on failure
        # to avoid using stale cached price for position operations.
        live_price = None
        try:
            live_price = fetch_price(settings.binance_symbol)
        except Exception:
            logger.warning("fetch_price failed, retrying once")
        if not live_price:
            try:
                live_price = fetch_price(settings.binance_symbol)
            except Exception:
                logger.warning("fetch_price retry failed, skipping position management")
        if live_price:
            data["ticker"]["price"] = live_price
            # Now manage positions with fresh price
            manage_simulated_position(conn, data, live_price)
        else:
            logger.warning("no live price available — skipping manage_simulated_position to avoid stale-price decisions")

        # Save planned entry price before _refresh_live_prices overwrites it
        _planned_entry = data["verdict"].get("order_signal", {}).get("entry_price")
        if _planned_entry is not None:
            data["verdict"]["order_signal"]["planned_entry_price"] = _planned_entry

        # Re-query simulated position since management may have changed it
        sim_row = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' AND is_simulated = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if sim_row:
            sim = dict(sim_row)
            orig = conn.execute(
                "SELECT position_size FROM position_action_state "
                "WHERE position_id = ? AND action = 'open' LIMIT 1",
                (sim["id"],),
            ).fetchone()
            if orig and orig["position_size"]:
                sim["original_size"] = orig["position_size"]
            data["simulated_position"] = sim
        else:
            data["simulated_position"] = None

        # Evolution uses same conn (avoids extra DB open/close)
        evolution_data = _build_evolution(conn)

        data["timestamp"] = datetime.now().isoformat()
        data["evolution"] = evolution_data

        # Reload per-TF base thresholds after evolution may have adjusted them
        ev_params = get_active_thresholds()
        tf_cfg = ev_params.get("tf_thresholds", {})
        vol_adj_factor = ev_params.get("vol_adjustment_factor", 0.1)
        for tf_key in ["30m", "1h", "4h"]:
            tf_base = tf_cfg.get(tf_key, {})
            base_t = tf_base.get("adx_trending_threshold", ev_params["adx_trending_threshold"])
            base_f = tf_base.get("adx_forming_threshold", ev_params["adx_forming_threshold"])
            tf_data = data.get("verdict", {}).get("timeframes", {}).get(tf_key, {})
            vol_pct = tf_data.get("vol_percentile", 50)
            t_adj = max(-5, min(5, (vol_pct - 50) * vol_adj_factor))
            f_adj = max(-3, min(3, (vol_pct - 50) * vol_adj_factor * 0.6))
            tf_data["base_trending"] = base_t
            tf_data["base_forming"] = base_f
            tf_data["trending_adj"] = round(t_adj, 1)
            tf_data["forming_adj"] = round(f_adj, 1)
            tf_data["effective_trending"] = round(min(base_t + t_adj, 34), 1)
            tf_data["effective_forming"] = round(min(base_f + f_adj, 28), 1)

        # 更新实时价格（复用已获取的 live_price，避免重复调用 fetch_price）
        _refresh_live_prices(data, data.get("ticker", {}).get("price"))
        data["_generated_at"] = time.time()
    finally:
        conn.close()
    return data


def _rebuild_signals_response(now: float = None) -> dict:
    """Cache miss: full signal generation from Binance."""
    conn = get_connection()
    try:
        # Query history rows and open position BEFORE generating verdict
        history_rows_raw = conn.execute(
            "SELECT * FROM verdict_history ORDER BY created_at DESC LIMIT 300"
        ).fetchall()
        history_rows = []
        for r in history_rows_raw:
            history_rows.append({
                "regime": r["regime"], "direction": r["direction"],
                "strength": r["strength"], "confidence": r["confidence"],
                "momentum": r["momentum"], "advice": r["advice"],
                "price": r["price"], "adx_4h": r["adx_4h"], "adx_1h": r["adx_1h"],
                "dir_4h": r["dir_4h"], "dir_1h": r["dir_1h"],
                "created_at": r["created_at"],
            })
    
        open_position = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' AND is_simulated = 0 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        position_dict = dict(open_position) if open_position else None
    
        if now is None:
            now = time.time()
    
        # Fetch all market data in parallel (klines + funding + OI)
        # Use different limits per TF — vol_percentile needs ~5-7 days of data
        market_data = fetch_all_market_data(
            settings.binance_symbol,
            timeframes=["30m", "1h", "4h"],
            limits={"30m": 200, "1h": 200, "4h": 100},
        )
        funding_rate = market_data.get("funding_rate", 0.0)
        current_oi = market_data.get("open_interest", 0.0)
        prev_oi = market_data.get("open_interest_prev", 0.0)

        # Fetch failure guard: 4h klines are mandatory for signal generation
        if not market_data.get("klines_4h"):
            logger.error("4h K线获取失败，信号生成中止")
            raise RuntimeError("4h K线获取失败，信号生成中止")

        # Recent price history for OI divergence detection
        price_candles = market_data.get("klines_4h", [])[:6]
        price_history = [c["close"] for c in price_candles] if price_candles else []
    
        # Exhaustion reversal cooling: check if a reversal signal was recently emitted
        last_reversal_row = conn.execute(
            "SELECT created_at, direction FROM signals "
            "WHERE signal_type = 'exhaustion_reversal' ORDER BY created_at DESC LIMIT 3"
        ).fetchall()
        recent_reversals = [
            {"created_at": r["created_at"], "direction": r["direction"]} for r in last_reversal_row
        ]
    
        # P3 ranging breakout / macro bias cooldown: track recent signals
        last_p3_row = conn.execute(
            "SELECT created_at, direction, signal_type FROM signals "
            "WHERE signal_type IN ('ranging_breakout', 'macro_bias_long') "
            "ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        recent_p3_signals = [
            {"created_at": r["created_at"], "direction": r["direction"], "signal_type": r["signal_type"]}
            for r in last_p3_row
        ]
    
        # Aggressive signal cooldowns: track recent aggressive signal timestamps
        from app.signal_engine import AGGRESSIVE_COOLDOWNS
        last_agg_row = conn.execute(
            "SELECT created_at, signal_type FROM signals "
            "WHERE signal_type LIKE 'aggressive_%' OR signal_type IN ('double_top', 'double_bottom') "
            "ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        now_ts = time.time()
        aggressive_cooldowns = {}
        for r in last_agg_row:
            stype = r["signal_type"]
            if stype in AGGRESSIVE_COOLDOWNS:
                elapsed = now_ts - r["created_at"]
                if elapsed < AGGRESSIVE_COOLDOWNS[stype]:
                    aggressive_cooldowns[stype] = True
    
        # Track last active aggressive signal (for sticky persistence):
        # double_top/double_bottom are persistent patterns; volume_spike is transient.
        last_aggressive = None
        for r in last_agg_row:
            stype = r["signal_type"]
            if stype in ("double_top", "double_bottom"):
                agg_key = "aggressive_pattern_breakout"
            elif stype.startswith("aggressive_"):
                agg_key = stype
            else:
                continue
            elapsed = now_ts - r["created_at"]
            if elapsed < AGGRESSIVE_COOLDOWNS.get(agg_key, 7200):
                is_persistent = agg_key in ("aggressive_pattern_breakout", "aggressive_squeeze_breakout")
                last_aggressive = f"{agg_key},{is_persistent}"
                break

        # Exhaustion block dedup: count consecutive same-direction trend_exhaustion_block
        # signals. When ADX > 50 and 3+ consecutive signals are identical, suppress
        # generating another one to avoid polluting verification stats.
        eb_rows = conn.execute(
            "SELECT direction, signal_type FROM signals WHERE timeframe = '4h' "
            "AND signal_type = 'trend_exhaustion_block' ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        eb_consecutive_same_dir = 0
        eb_last_dir = None
        for r in eb_rows:
            d = r["direction"]
            if d == eb_last_dir or eb_last_dir is None:
                eb_consecutive_same_dir += 1
                eb_last_dir = d
            else:
                break
    
        market_ctx = {
            "funding_rate": funding_rate,
            "open_interest": current_oi,
            "open_interest_prev": prev_oi,
            "price_history": price_history,
            "recent_reversals": recent_reversals,
            "recent_p3_signals": recent_p3_signals,
            "aggressive_cooldowns": aggressive_cooldowns,
            "last_aggressive_signal": last_aggressive,
            "exhaustion_block_consecutive": eb_consecutive_same_dir,
        }

        # Query previous per-TF regimes for hysteresis
        previous_regimes = {}
        for tf in ("30m", "1h", "4h"):
            row = conn.execute(
                "SELECT regime, direction, signal_type FROM signals WHERE timeframe = ? ORDER BY created_at DESC LIMIT 1",
                (tf,),
            ).fetchone()
            if row:
                previous_regimes[tf] = row["regime"]
                if row["direction"]:
                    previous_regimes[f"{tf}_direction"] = row["direction"]
                if row["signal_type"]:
                    previous_regimes[f"{tf}_signal_type"] = row["signal_type"]

        data = generate_verdict(
            history_rows=history_rows,
            position=position_dict,
            market_context=market_ctx,
            pre_fetched_klines={
                "30m": market_data.get("klines_30m", []),
                "1h": market_data.get("klines_1h", []),
                "4h": market_data.get("klines_4h", []),
            },
            previous_regimes=previous_regimes,
        )
        latest_signals = {}
        for tf in data["timeframes"]:
            row = conn.execute(
                "SELECT regime, direction, verdict, action FROM signals "
                "WHERE timeframe = ? ORDER BY created_at DESC LIMIT 1",
                (tf,),
            ).fetchone()
            if row:
                latest_signals[tf] = (row["regime"], row["direction"], row["verdict"], row["action"])
    
        signal_changed = False
        # Per-TF signal recording: always record the latest signal for each new candle.
        # If a signal for the current candle already exists, UPDATE it with fresh data.
        # This keeps snapshots fresh even when regime/direction doesn't change.
        CANDLE_SECONDS = {"30m": 1800, "1h": 3600, "4h": 14400}
        for tf, tf_data in data["timeframes"].items():
            tf_seconds = CANDLE_SECONDS.get(tf, 3600)
            candle_start = now - (now % tf_seconds)

            # Check if we already recorded a signal for this candle
            existing = conn.execute(
                "SELECT id FROM signals WHERE timeframe = ? AND created_at >= ? LIMIT 1",
                (tf, candle_start),
            ).fetchone()
            if existing:
                # Update the existing record with latest ADX/DI/price (real-time candle data)
                conn.execute(
                    "UPDATE signals SET regime=?, direction=?, adx=?, plus_di=?, minus_di=?, "
                    "confidence=?, strength=?, momentum=?, duration_hours=?, price_at_signal=?, "
                    "verdict=?, action=?, target=?, stop=?, signal_type=? WHERE timeframe=? AND created_at >= ?",
                    (
                        tf_data["regime"], tf_data["direction"], tf_data["adx"],
                        tf_data["plus_di"], tf_data["minus_di"], tf_data["confidence"],
                        tf_data.get("strength"), tf_data.get("momentum"),
                        tf_data.get("duration_hours"), tf_data["price"],
                        data["verdict"]["direction"],
                        data["verdict"]["advice"]["action"],
                        data["verdict"]["advice"]["target"],
                        data["verdict"]["advice"]["stop"],
                        tf_data.get("signal_type", "trend_following"),
                        tf, candle_start,
                    ),
                )
                continue

            # New candle -> insert fresh signal record
            signal_changed = True
            conn.execute(
                "INSERT INTO signals (timeframe, regime, direction, adx, plus_di, minus_di, "
                "confidence, strength, momentum, duration_hours, price_at_signal, "
                "verdict, action, target, stop, signal_type, created_at, verified) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    tf, tf_data["regime"], tf_data["direction"], tf_data["adx"],
                    tf_data["plus_di"], tf_data["minus_di"], tf_data["confidence"],
                    tf_data.get("strength"), tf_data.get("momentum"),
                    tf_data.get("duration_hours"), tf_data["price"],
                    data["verdict"]["direction"],
                    data["verdict"]["advice"]["action"],
                    data["verdict"]["advice"]["target"],
                    data["verdict"]["advice"]["stop"],
                    tf_data.get("signal_type", "trend_following"),
                    now,
                ),
            )
    
        if signal_changed:
            # Dedup verdict_history: only insert when verdict actually changed,
            # OR enough time has passed (heartbeat for same-state persistence).
            last_verdict = conn.execute(
                "SELECT regime, direction, strength, created_at FROM verdict_history ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            verdict_key = (data["verdict"]["regime"], data["verdict"]["direction"], data["verdict"]["strength"])
            min_interval = 1800   # 30 min minimum for changed verdicts
            heartbeat_interval = 7200  # 2 hour heartbeat for unchanged verdicts

            if last_verdict is None:
                should_insert = True
            elif (last_verdict["regime"], last_verdict["direction"], last_verdict["strength"]) == verdict_key:
                # Same verdict — only insert a heartbeat entry every 2 hours
                should_insert = (now - last_verdict["created_at"]) >= heartbeat_interval
            elif (now - last_verdict["created_at"]) < min_interval:
                should_insert = False  # too soon, skip
            else:
                should_insert = True

            if should_insert:
                conn.execute(
                    "INSERT INTO verdict_history (regime, direction, strength, confidence, "
                    "momentum, advice, price, adx_4h, adx_1h, dir_4h, dir_1h, created_at) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        data["verdict"]["regime"], data["verdict"]["direction"],
                        data["verdict"]["strength"], data["verdict"]["confidence"],
                        data["verdict"]["momentum"], data["verdict"]["advice"]["action"],
                        data["ticker"]["price"],
                        data["timeframes"]["4h"]["adx"],
                        data["timeframes"]["1h"]["adx"],
                        data["timeframes"]["4h"]["direction"],
                        data["timeframes"]["1h"]["direction"],
                        now,
                    ),
                )
        conn.commit()
    
        history_rows = conn.execute(
            "SELECT * FROM verdict_history ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        verdict_history = []
        for r in history_rows:
            dt = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d %H:%M")
            verdict_history.append({
                "time": dt, "regime": r["regime"], "direction": r["direction"],
                "strength": r["strength"], "confidence": r["confidence"],
                "momentum": r["momentum"], "advice": r["advice"],
                "price": r["price"], "adx_4h": r["adx_4h"], "adx_1h": r["adx_1h"],
                "dir_4h": r["dir_4h"], "dir_1h": r["dir_1h"],
            })
        deduped = []
        for h in verdict_history:
            if not deduped or deduped[-1]["regime"] != h["regime"] or deduped[-1]["direction"] != h["direction"]:
                deduped.append(h)
        verdict_history = deduped
    
        snapshots = conn.execute(
            "SELECT regime, direction, adx, plus_di, minus_di, price_at_signal, created_at "
            "FROM signals WHERE timeframe = '30m' ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        m30_snapshots = []
        for s in snapshots:
            dt = datetime.fromtimestamp(s["created_at"]).strftime("%H:%M:%S")
            m30_snapshots.append({
                "ts": s["created_at"], "time": dt,
                "price": s["price_at_signal"], "adx": s["adx"],
                "plus_di": s["plus_di"], "minus_di": s["minus_di"],
                "regime": s["regime"], "direction": s["direction"],
            })
    
        data["verdict_history"] = verdict_history
        data["m30_snapshots"] = m30_snapshots
        data["timestamp"] = datetime.now().isoformat()
        data["verdict"]["price_at_signal"] = data["ticker"]["price"]
        data["evolution"] = _build_evolution(conn)
    
        # Fetch fresh price BEFORE position management — K-line close price may be stale.
        # _refresh_live_prices fetches from Binance ticker and updates data["ticker"]["price"].
        _refresh_live_prices(data, None)  # None = force fetch from Binance

        # Manage simulated positions with fresh price
        manage_simulated_position(conn, data, data["ticker"]["price"])

        # Save planned entry price AFTER position management (order_signal may be updated)
        _planned_entry = data["verdict"].get("order_signal", {}).get("entry_price")
        if _planned_entry is not None:
            data["verdict"]["order_signal"]["planned_entry_price"] = _planned_entry

        # Re-query simulated position for response
        sim_row = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' AND is_simulated = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if sim_row:
            sim = dict(sim_row)
            # Attach original position size from action_state (for reduced positions)
            orig = conn.execute(
                "SELECT position_size FROM position_action_state "
                "WHERE position_id = ? AND action = 'open' LIMIT 1",
                (sim["id"],),
            ).fetchone()
            if orig and orig["position_size"]:
                sim["original_size"] = orig["position_size"]
            data["simulated_position"] = sim
        else:
            data["simulated_position"] = None

        # 写入缓存
        data["_generated_at"] = time.time()
        _set_cached_signals({"data": data})
    finally:
        conn.close()
    return data


def _build_evolution(conn=None) -> dict:
    """Build evolution stats with pending signal verification (only if pending)."""
    own_conn = False
    if conn is None:
        conn = get_connection()
        own_conn = True

    # Quick check: only fetch klines when there are actually pending signals
    pending = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE verified = 0"
    ).fetchone()
    if pending and pending["cnt"] > 0:
        verify_pending_signals(conn)

    result = get_evolution_stats()

    if own_conn:
        conn.close()
    return result


# ---------- API ----------

@app.get("/api/signals")
async def api_signals(request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    return _build_signals_response()


# ---------- API: Position actions (used by frontend for position management) ----------

@app.post("/api/position/{position_id}/close")
async def api_close_position(position_id: int, request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM positions WHERE id = ? AND status = 'open'",
        (position_id,),
    ).fetchone()
    if row is None:
        conn.close()
        return JSONResponse({"detail": "Position not found or already closed"}, status_code=404)

    hl_result = None
    if row["hl_enabled"]:
        router = get_router()
        hl_result = router.close(row.get("hl_sz") or 0)

    now = time.time()
    current_price = fetch_price(settings.binance_symbol) or row["entry_price"]
    updates: dict = {"status": "closed", "updated_at": now, "closed_at": now, "close_price": current_price}

    if row["is_simulated"]:
        # Calculate PnL for simulated positions
        entry = row["entry_price"]
        pct = (current_price - entry) / entry
        if row["side"] == "short":
            pct = -pct
        pnl = pct * (row["position_size"] * entry)  # notional * price change
        updates["pnl"] = round(pnl, 2)

    conn.execute(
        "UPDATE positions SET status=?, updated_at=?, closed_at=?, close_price=?, pnl=? WHERE id=?",
        (updates["status"], updates["updated_at"], updates["closed_at"], updates["close_price"], updates.get("pnl"), position_id),
    )
    conn.commit()
    conn.close()

    resp = {"ok": True}
    if row["hl_enabled"]:
        if hl_result and hl_result.get("ok"):
            resp["hyperliquid"] = True
            resp["oid"] = hl_result.get("order_id")
        else:
            resp["hyperliquid"] = False
            resp["error"] = hl_result.get("error", "Close failed") if hl_result else "Close failed"
    return resp


@app.post("/api/position/{position_id}/action")
async def api_record_action(position_id: int, request: Request):
    """Record a position action state change (add/reduce/exit)."""
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    body = await request.json()
    conn = get_connection()
    row = conn.execute("SELECT * FROM positions WHERE id = ? AND status = 'open'", (position_id,)).fetchone()
    if row is None:
        conn.close()
        return JSONResponse({"detail": "Position not found or already closed"}, status_code=404)

    action = body.get("action")  # add/reduce/exit
    adx_4h = body.get("adx_4h")
    price = body.get("price")
    position_size = body.get("position_size")  # BTC quantity for add/reduce
    now = time.time()

    conn.execute(
        "INSERT INTO position_action_state (position_id, action, adx_4h, price, position_size, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (position_id, action, adx_4h, price, position_size, now),
    )

    # Update position's action_state, and adjust size/pnl for reduce
    if "reduce" in action:
        conn.execute("UPDATE positions SET action_state = 'reduced' WHERE id = ?", (position_id,))
        # Adjust position size and record realized pnl
        current_size = row.get("position_size") or 0
        new_size = current_size - (position_size or 0)
        entry = row.get("entry_price") or 0
        realized_pnl = (price - entry) * position_size if row["side"] == "long" and entry and position_size else (entry - price) * position_size if row["side"] == "short" and entry and position_size else 0
        cum_pnl = (row.get("realized_pnl") or 0) + realized_pnl
        conn.execute(
            "UPDATE positions SET position_size=?, realized_pnl=? WHERE id=?",
            (round(new_size, 5), round(cum_pnl, 2), position_id),
        )
    elif action == "exit":
        conn.execute("UPDATE positions SET action_state = 'exited', status = 'closed', updated_at = ? WHERE id = ?", (now, position_id))
    else:
        conn.execute("UPDATE positions SET action_state = ? WHERE id = ?", (action, position_id))

    # Update max_adx if higher
    if adx_4h and (row["max_adx"] is None or adx_4h > row["max_adx"]):
        conn.execute("UPDATE positions SET max_adx = ? WHERE id = ?", (adx_4h, position_id))

    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/evolution")
async def api_evolution(request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    return _build_evolution()


@app.get("/api/position/simulated")
async def api_get_simulated_position(request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM positions WHERE status = 'open' AND is_simulated = 1 ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        sim = dict(row)
        orig = conn.execute(
            "SELECT position_size FROM position_action_state "
            "WHERE position_id = ? AND action = 'open' LIMIT 1",
            (sim["id"],),
        ).fetchone()
        if orig and orig["position_size"]:
            sim["original_size"] = orig["position_size"]
    else:
        sim = None
    conn.close()
    return sim


@app.get("/api/positions/simulated/history")
async def api_simulated_position_history(request: Request, page: int = 1, page_size: int = 10):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    conn = get_connection()

    offset = (page - 1) * page_size

    # Total count: open=1 row, closed=2 rows (open+close), plus reduce rows
    reduce_count = conn.execute(
        "SELECT COUNT(*) FROM position_action_state WHERE action LIKE 'reduce_%' OR action IN ('decay_reduce', 'decay_reduce_loss')"
    ).fetchone()[0]
    total = conn.execute("""
        SELECT COUNT(*) FROM positions
        WHERE is_simulated = 1 AND status = 'open'
    """).fetchone()[0] + conn.execute("""
        SELECT COUNT(*) * 2 FROM positions
        WHERE is_simulated = 1 AND status = 'closed'
    """).fetchone()[0] + reduce_count

    # Build all records via UNION ALL at DB level, then paginate
    # Each record type: open, close, reduce — unified columns for display
    all_records = []

    # Open positions (1 row each)
    open_rows = conn.execute("""
        SELECT id, side, entry_price, pnl, realized_pnl, created_at,
               closed_at, reduce_count, add_count, status, leverage, position_size,
               entry_reason, close_reason, close_price,
               'open' as action_type, NULL as action_detail
        FROM positions
        WHERE is_simulated = 1 AND status = 'open'
        ORDER BY created_at DESC
    """).fetchall()

    # Closed positions: open row + close row
    closed_rows = conn.execute("""
        SELECT id, side, entry_price, pnl, realized_pnl, created_at,
               closed_at, reduce_count, add_count, status, leverage, position_size,
               entry_reason, close_reason, close_price,
               'close' as action_type, close_reason as action_detail
        FROM positions
        WHERE is_simulated = 1 AND status = 'closed'
        ORDER BY closed_at DESC
    """).fetchall()

    # Reduce rows from all positions
    reduce_rows = conn.execute("""
        SELECT position_id as id, action as action_detail, price,
               position_size, created_at, adx_4h
        FROM position_action_state
        WHERE action LIKE 'reduce_%' OR action IN ('decay_reduce', 'decay_reduce_loss')
        ORDER BY created_at DESC
    """).fetchall()

    # Build in-memory records
    for r in open_rows:
        rd = dict(r)
        rd["action"] = "open"
        rd["action_type"] = "open"
        rd["duration_hours"] = None
        rd["reduce_rows"] = []
        all_records.append(rd)

    for r in closed_rows:
        rd = dict(r)
        rd["duration_hours"] = round((rd["closed_at"] - rd["created_at"]) / 3600, 1) if rd.get("closed_at") and rd.get("created_at") else None

        # Open row
        open_rec = dict(rd)
        open_rec["action"] = "open"
        open_rec["action_type"] = "open"
        open_rec["price"] = rd["entry_price"]
        all_records.append(open_rec)

        # Close row
        close_rec = dict(rd)
        close_rec["action"] = "close"
        close_rec["action_type"] = "close"
        close_rec["price"] = rd["close_price"]
        close_rec["created_at"] = rd["closed_at"]
        all_records.append(close_rec)

    # Pre-fetch all reduce rows by position_id (single query instead of N+1)
    reduce_by_pos = {}
    for rr in reduce_rows:
        pid = rr["id"]
        reduce_by_pos.setdefault(pid, []).append(dict(rr))

    # Inject reduce rows into their parent positions
    for rec in all_records:
        pid = rec["id"]
        if pid in reduce_by_pos:
            rec["reduce_rows"] = reduce_by_pos[pid]

    # Expand reduce rows into the flat list for display
    expanded = []
    for rec in all_records:
        expanded.append(rec)
        for red in rec.get("reduce_rows", []):
            entry = rec["entry_price"]
            side = rec["side"]
            reduced_amt = red["position_size"]
            pnl = round((red["price"] - entry) * reduced_amt if side == "long" else (entry - red["price"]) * reduced_amt, 2)
            expanded.append({
                "id": rec["id"],
                "side": rec["side"],
                "price": red["price"],
                "pnl": pnl,
                "realized_pnl": rec.get("realized_pnl", 0),
                "created_at": red["created_at"],
                "closed_at": rec.get("closed_at"),
                "status": rec["status"],
                "leverage": rec["leverage"],
                "position_size": reduced_amt,
                "entry_reason": rec.get("entry_reason"),
                "close_reason": rec.get("close_reason"),
                "close_price": rec.get("close_price"),
                "action": red["action_detail"],
                "action_type": "reduce",
                "duration_hours": None,
            })

    # Sort by display time descending
    expanded.sort(key=lambda r: r.get("created_at") or 0, reverse=True)

    # Apply pagination
    records = expanded[offset:offset + page_size]

    # Clean up internal reduce_rows before returning
    for rec in records:
        rec.pop("reduce_rows", None)

    conn.close()

    return {
        "positions": records,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


@app.get("/api/position/simulated/stats")
async def api_simulated_position_stats(request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    conn = get_connection()

    # Realized PnL: sum of all closed simulated positions + realized pnl from open positions (reduces)
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) as closed_pnl, COUNT(*) as total_trades "
        "FROM positions WHERE is_simulated = 1 AND status = 'closed'"
    ).fetchone()
    closed_pnl = row["closed_pnl"]
    total_trades = row["total_trades"]

    # Add realized pnl from open positions (from partial reduces)
    row = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0) as open_realized_pnl "
        "FROM positions WHERE is_simulated = 1 AND status = 'open'"
    ).fetchone()
    sim_realized_pnl = closed_pnl + (row["open_realized_pnl"] or 0)

    # When HL backend is active, use actual HL realized PnL from fills
    if settings.trade_backend == "hyperliquid" and settings.auto_trade_enabled:
        try:
            hl_realized_pnl = get_router().get_realized_pnl()
            realized_pnl = hl_realized_pnl
        except Exception:
            realized_pnl = sim_realized_pnl
    else:
        realized_pnl = sim_realized_pnl

    # Today's PnL: closed positions since start of today (UTC+8) + realized PnL from open positions
    import time as _time
    now = _time.time()
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    utc8_now = _dt.now(_tz.utc).astimezone(_tz(_td(hours=8)))
    today_start = _dt(utc8_now.year, utc8_now.month, utc8_now.day).replace(tzinfo=_tz(_td(hours=8))).timestamp()
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) as today_pnl, COUNT(*) as today_trades "
        "FROM positions WHERE is_simulated = 1 AND status = 'closed' AND closed_at >= ?",
        (today_start,),
    ).fetchone()
    today_pnl = row["today_pnl"]
    today_trades = row["today_trades"]

    # Add realized PnL from open positions (partial reduces not captured in closed query)
    row = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0) as open_realized_pnl "
        "FROM positions WHERE is_simulated = 1 AND status = 'open'"
    ).fetchone()
    today_pnl += (row["open_realized_pnl"] or 0)

    # Current balance = HL account value (if HL active) or initial + realized PnL
    initial_balance = settings.sim_initial_balance
    if settings.trade_backend == "hyperliquid" and settings.auto_trade_enabled:
        try:
            hl_value = get_router().get_account_value()
            if hl_value and hl_value > 0:
                current_balance = hl_value
            else:
                current_balance = initial_balance + realized_pnl
        except Exception:
            current_balance = initial_balance + realized_pnl
    else:
        current_balance = initial_balance + realized_pnl

    # Open position margin (notional / leverage)
    open_pos = conn.execute(
        "SELECT position_size, entry_price, leverage FROM positions "
        "WHERE is_simulated = 1 AND status = 'open' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if open_pos and open_pos["position_size"] and open_pos["entry_price"]:
        margin = (open_pos["position_size"] * open_pos["entry_price"]) / (open_pos["leverage"] or 1)
        available_balance = current_balance - margin
    else:
        margin = 0
        available_balance = current_balance

    # Win rate
    row = conn.execute(
        "SELECT COUNT(*) as wins FROM positions WHERE is_simulated = 1 AND status = 'closed' AND pnl > 0"
    ).fetchone()
    wins = row["wins"]
    win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0

    # 7-day equity curve: daily balance snapshot
    seven_days_ago = now - 7 * 86400
    # Build timeline: start from initial balance, add closed pnl day by day
    rows = conn.execute(
        "SELECT date(closed_at, 'unixepoch', '+8 hours') as day, COALESCE(SUM(pnl), 0) as day_pnl "
        "FROM positions WHERE is_simulated = 1 AND status = 'closed' AND closed_at >= ? "
        "GROUP BY day ORDER BY day",
        (seven_days_ago,),
    ).fetchall()

    # Include realized pnl from currently open positions (from reduces)
    open_realized = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0) as rpnl FROM positions "
        "WHERE is_simulated = 1 AND status = 'open'"
    ).fetchone()["rpnl"] or 0

    equity_curve = []
    cum_pnl = 0
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Merge open realized pnl into today's pnl to avoid duplicate date entries
    if open_realized != 0:
        # Check if today already has closed pnl entries
        if rows and rows[-1]["day"] == today_str:
            # Merge: add open realized to today's pnl
            rows[-1] = {"day": today_str, "day_pnl": rows[-1]["day_pnl"] + open_realized}
        else:
            # No closed entries today, add a new row
            rows = list(rows) + [{"day": today_str, "day_pnl": open_realized}]

    for r in rows:
        cum_pnl += r["day_pnl"]
        equity_curve.append({"date": r["day"], "balance": round(initial_balance + cum_pnl, 2)})

    conn.close()

    return {
        "initial_balance": initial_balance,
        "current_balance": round(current_balance, 2),
        "realized_pnl": round(realized_pnl, 2),
        "today_pnl": round(today_pnl, 2),
        "total_trades": total_trades,
        "today_trades": today_trades,
        "available_balance": round(available_balance, 2),
        "margin_used": round(margin, 2),
        "win_rate": win_rate,
        "wins": wins,
        "losses": total_trades - wins,
        "equity_curve": equity_curve,
    }


# ──────────────────────────────────────────────────────────────────────
# Hyperliquid Live Trading API Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/hyperliquid/portfolio")
async def api_hl_portfolio(request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        return viewer.get_portfolio()
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/hyperliquid/positions")
async def api_hl_positions(request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        return viewer.get_positions()
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/hyperliquid/orders")
async def api_hl_orders(request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        return viewer.get_open_orders()
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/hyperliquid/fills")
async def api_hl_fills(request: Request, page: int = 1, size: int = 50):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        return viewer.get_fills(page=page, size=size)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/hyperliquid/funding")
async def api_hl_funding(request: Request, hours: int = 24):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        return viewer.get_funding_history(hours=hours)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/hyperliquid/fees")
async def api_hl_fees(request: Request):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        return viewer.get_fee_stats()
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/hyperliquid/net_curve")
async def api_hl_net_curve(request: Request, period: str = "7d"):
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        return viewer.get_net_value_curve(period=period)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/hyperliquid/order_history")
async def api_hl_order_history(request: Request, page: int = 1, size: int = 10):
    """Order history: fills aggregated by order ID (oid).

    Data is served from local hl_fills cache (incrementally synced from HL API).
    Returns paginated list of orders + per-order fills for detail view.
    """
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        conn = get_connection()

        # Sync new fills from HL API (incremental, fast if nothing new)
        from app.hyperliquid_viewer import sync_fills_to_db
        sync_fills_to_db(conn, viewer, coin="BTC")

        # Aggregate fills by oid from local DB
        rows = conn.execute("""
            SELECT oid, side, dir,
                   SUM(size) as total_size,
                   SUM(price * size) / SUM(size) as avg_price,
                   SUM(closed_pnl) as total_pnl,
                   SUM(ABS(fee)) as total_fee,
                   COUNT(*) as fill_count,
                   MIN(timestamp_ms) as first_fill_time,
                   MAX(timestamp_ms) as last_fill_time
            FROM hl_fills
            WHERE coin = 'BTC'
            GROUP BY oid
            ORDER BY MAX(timestamp_ms) DESC
        """).fetchall()

        orders = []
        for r in rows:
            first_fill = datetime.fromtimestamp(r["first_fill_time"] / 1000).strftime("%m-%d %H:%M")
            last_fill = datetime.fromtimestamp(r["last_fill_time"] / 1000).strftime("%m-%d %H:%M")
            orders.append({
                "oid": r["oid"],
                "dir": r["dir"] or "",
                "side": r["side"],
                "fill_count": r["fill_count"],
                "total_size": round(r["total_size"], 5),
                "avg_price": round(r["avg_price"], 1),
                "total_pnl": round(r["total_pnl"], 2),
                "total_fee": round(r["total_fee"], 4),
                "first_fill_time": r["first_fill_time"],
                "last_fill_time": r["last_fill_time"],
                "first_fill_date": first_fill,
                "last_fill_date": last_fill,
            })

        total = len(orders)
        start = (page - 1) * size
        end = start + size
        page_orders = orders[start:end]

        conn.close()

        return {
            "orders": page_orders,
            "total": total,
            "page": page,
            "total_pages": max(1, (total + size - 1) // size),
        }
    except Exception as e:
        _logger.exception("HL order_history failed")
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/hyperliquid/pnl_curve")
async def api_hl_pnl_curve(request: Request, days: int = 0):
    """Perps PnL curve: cumulative closed PnL from fills over the last N days.

    Data is served from local hl_fills cache.
    days=0 means all available fills (no time filter).
    """
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        viewer = get_viewer()
        conn = get_connection()

        # Sync new fills from HL API
        from app.hyperliquid_viewer import sync_fills_to_db
        sync_fills_to_db(conn, viewer, coin="BTC")

        if days > 0:
            cutoff_ms = (time.time() - days * 86400) * 1000
            rows = conn.execute(
                "SELECT tid, oid, side, closed_pnl, ABS(fee) as fee, timestamp_ms "
                "FROM hl_fills WHERE coin = 'BTC' AND timestamp_ms >= ? "
                "ORDER BY timestamp_ms ASC",
                (cutoff_ms,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT tid, oid, side, closed_pnl, ABS(fee) as fee, timestamp_ms "
                "FROM hl_fills WHERE coin = 'BTC' "
                "ORDER BY timestamp_ms ASC"
            ).fetchall()

        # Build cumulative PnL curve
        cum_pnl = 0
        total_pnl = 0
        total_fees = 0
        curve = []
        fill_count = 0
        for r in rows:
            pnl = r["closed_pnl"]
            fee = r["fee"]
            total_pnl += pnl
            total_fees += fee
            if pnl != 0:
                cum_pnl += pnl
                fill_count += 1
                dt = datetime.fromtimestamp(r["timestamp_ms"] / 1000).strftime("%m-%d %H:%M")
                curve.append({"timestamp": r["timestamp_ms"], "date": dt, "cum_pnl": round(cum_pnl, 2)})

        # Ensure at least one point (start at 0)
        if curve and (curve[0]["cum_pnl"] != 0 or len(curve) > 1):
            first_ts = curve[0]["timestamp"]
            first_dt = curve[0]["date"]
            curve.insert(0, {"timestamp": first_ts, "date": first_dt, "cum_pnl": 0})
        elif not curve:
            now_ts = int(time.time() * 1000)
            now_dt = datetime.fromtimestamp(now_ts / 1000).strftime("%m-%d %H:%M")
            curve = [{"timestamp": now_ts, "date": now_dt, "cum_pnl": 0}]

        # Current unrealized PnL
        positions = viewer.get_positions()
        unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions if "error" not in p)

        # Net PnL = realized - fees + unrealized
        net_pnl = total_pnl - total_fees + unrealized_pnl

        conn.close()

        return {
            "curve": curve,
            "total_pnl": round(total_pnl, 2),
            "total_fees": round(total_fees, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "fill_count": fill_count,
        }
    except Exception as e:
        _logger.exception("HL pnl_curve failed")
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/api/position/{position_id}/hl_orders")
async def api_position_hl_orders(position_id: int, request: Request):
    """Get all HL orders mapped to a local position (open/add/close/reduce/tp/sl)."""
    if not require_auth(request):
        return JSONResponse({"detail": "未登录或登录已失效"}, status_code=401)
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM hl_order_mapping WHERE position_id = ? ORDER BY created_at",
            (position_id,),
        ).fetchall()
        orders = [dict(r) for r in rows]
        conn.close()
        return {"position_id": position_id, "orders": orders, "total": len(orders)}
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

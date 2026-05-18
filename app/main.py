"""BTC Signal System — FastAPI main entry point."""

import copy
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
from app.hyperliquid import place_order as hl_place_order, close_position as hl_close_position
from app.simulated_position_manager import (
    manage_simulated_position, _check_price_based_exit,
    _check_pnl_trailing_stop, _register_auto_hooks,
)

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
            import logging
            logging.getLogger("btc_signal").warning(f"fetch_price failed: {e}")
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
            import logging
            logging.getLogger("btc_signal").warning(f"fetch_klines 30m failed: {e}")
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
        import logging
        logging.getLogger("btc_signal").warning(f"_refresh_live_prices update failed: {e}")


@app.on_event("startup")
async def startup():
    init_db()
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
        current_price = fetch_price(settings.binance_symbol)
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
                if not _check_price_based_exit(conn, pos, current_price):
                    # Also run PnL trailing stop — this was previously only executed
                    # during the signal cycle (every 300s), missing profitable windows
                    _check_pnl_trailing_stop(conn, pos, current_price)

                # Track price extremes for add-on detection (_is_price_extreme)
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
        # 连续相同去重
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

        # Get open position for state machine (manual only)
        open_position = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' AND is_simulated = 0 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        position_dict = dict(open_position) if open_position else None

        # Deep-copy cached data, refresh price BEFORE position management
        data = copy.deepcopy(cached["data"])

        # Fetch live price first — manage_simulated_position needs current
        # price for accurate stop-loss/take-profit checks
        try:
            live_price = fetch_price(settings.binance_symbol)
            if live_price:
                data["ticker"]["price"] = live_price
        except Exception:
            pass  # fall back to cached price

        # Now manage positions with the freshest price available
        manage_simulated_position(conn, data, data["ticker"]["price"])

        # Save planned entry price before _refresh_live_prices overwrites it
        _planned_entry = data["verdict"].get("order_signal", {}).get("entry_price")
        if _planned_entry is not None:
            data["verdict"]["order_signal"]["planned_entry_price"] = _planned_entry

        # Re-query simulated position since management may have changed it
        sim_row = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' AND is_simulated = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        # Evolution uses same conn (avoids extra DB open/close)
        evolution_data = _build_evolution(conn)

        data["verdict_history"] = verdict_history
        data["m30_snapshots"] = m30_snapshots
        data["timestamp"] = datetime.now().isoformat()
        data["evolution"] = evolution_data
        data["simulated_position"] = dict(sim_row) if sim_row else None

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
    import logging
    logger = logging.getLogger('btc_signal')

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
    
        market_ctx = {
            "funding_rate": funding_rate,
            "open_interest": current_oi,
            "open_interest_prev": prev_oi,
            "price_history": price_history,
            "recent_reversals": recent_reversals,
            "recent_p3_signals": recent_p3_signals,
            "aggressive_cooldowns": aggressive_cooldowns,
            "last_aggressive_signal": last_aggressive,
        }

        # Query previous per-TF regimes for hysteresis
        previous_regimes = {}
        for tf in ("30m", "1h", "4h"):
            row = conn.execute(
                "SELECT regime, direction FROM signals WHERE timeframe = ? ORDER BY created_at DESC LIMIT 1",
                (tf,),
            ).fetchone()
            if row:
                previous_regimes[tf] = row["regime"]
                if row["direction"]:
                    previous_regimes[f"{tf}_direction"] = row["direction"]

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
        # Per-TF candle cooldown: prevent duplicate signals within same candle window.
        CANDLE_SECONDS = {"30m": 1800, "1h": 3600, "4h": 14400}
        for tf, tf_data in data["timeframes"].items():
            prev = latest_signals.get(tf)
            new_sig = (tf_data["regime"], tf_data["direction"], data["verdict"]["direction"], data["verdict"]["advice"]["action"])
            if prev is None or prev != new_sig:
                # Check if a signal was already created in the current candle period
                tf_seconds = CANDLE_SECONDS.get(tf, 3600)
                candle_start = now - (now % tf_seconds)
                existing = conn.execute(
                    "SELECT id FROM signals WHERE timeframe = ? AND created_at >= ? LIMIT 1",
                    (tf, candle_start),
                ).fetchone()
                if existing:
                    continue  # Skip: already have a signal for this candle
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
            # AND at least 30 minutes since last entry (prevents regime flip-flop).
            last_verdict = conn.execute(
                "SELECT regime, direction, strength, created_at FROM verdict_history ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            verdict_key = (data["verdict"]["regime"], data["verdict"]["direction"], data["verdict"]["strength"])
            min_interval = 1800  # 30 minutes between verdict entries
            if last_verdict is None:
                should_insert = True
            elif (last_verdict["regime"], last_verdict["direction"], last_verdict["strength"]) == verdict_key:
                should_insert = False  # exact same verdict
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
    
        # Manage simulated positions
        manage_simulated_position(conn, data, data["ticker"]["price"])

        # Save planned entry price BEFORE _refresh_live_prices overwrites entry_price
        _planned_entry = data["verdict"].get("order_signal", {}).get("entry_price")
        if _planned_entry is not None:
            data["verdict"]["order_signal"]["planned_entry_price"] = _planned_entry

        # 同步实时价格，确保 ticker.price 和 timeframes['30m'].price 一致
        _refresh_live_prices(data, data.get("ticker", {}).get("price"))
    
        # Query simulated position for response
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
        hl_result = hl_close_position(row.get("hl_sz"))

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
            resp["oid"] = hl_result.get("oid")
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
    now = time.time()

    conn.execute(
        "INSERT INTO position_action_state (position_id, action, adx_4h, price, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (position_id, action, adx_4h, price, now),
    )

    # Update position's action_state and max_adx
    if action == "reduce":
        conn.execute("UPDATE positions SET action_state = 'reduced' WHERE id = ?", (position_id,))
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

    # Each open position = 1 row (open). Each closed position = 2 rows (open + close).
    total = conn.execute("""
        SELECT COUNT(*) FROM positions
        WHERE is_simulated = 1 AND status = 'open'
    """).fetchone()[0] + conn.execute("""
        SELECT COUNT(*) * 2 FROM positions
        WHERE is_simulated = 1 AND status = 'closed'
    """).fetchone()[0]

    # Fetch open positions first
    open_rows = conn.execute("""
        SELECT id, side, entry_price, close_price, close_reason, entry_reason,
               leverage, position_size, pnl, realized_pnl, created_at, closed_at,
               reduce_count, add_count, status
        FROM positions
        WHERE is_simulated = 1 AND status = 'open'
        ORDER BY created_at DESC
    """).fetchall()

    # Fetch closed positions
    closed_rows = conn.execute("""
        SELECT id, side, entry_price, close_price, close_reason, entry_reason,
               leverage, position_size, pnl, realized_pnl, created_at, closed_at,
               reduce_count, add_count, status
        FROM positions
        WHERE is_simulated = 1 AND status = 'closed'
        ORDER BY closed_at DESC
    """).fetchall()

    # Build rows: for each closed position produce open+close, for open produce only open
    all_records = []
    for r in open_rows:
        rd = dict(r)
        rd["action_id"] = None
        rd["action"] = "open"
        rd["action_type"] = "open"
        rd["price"] = rd["entry_price"]
        rd["duration_hours"] = None
        all_records.append(rd)

    for r in closed_rows:
        rd = dict(r)
        rd["duration_hours"] = round((rd["closed_at"] - rd["created_at"]) / 3600, 1) if rd.get("closed_at") and rd.get("created_at") else None

        # Open row
        open_rec = dict(rd)
        open_rec["action"] = "open"
        open_rec["action_type"] = "open"
        open_rec["price"] = open_rec["entry_price"]
        all_records.append(open_rec)

        # Close row
        close_rec = dict(rd)
        close_rec["action"] = "close"
        close_rec["action_type"] = "close"
        close_rec["price"] = close_rec.get("close_price") or close_rec["entry_price"]
        close_rec["created_at"] = close_rec["closed_at"]  # show close time in UI
        all_records.append(close_rec)

    # Sort ALL records by display time descending
    for rec in all_records:
        rec["_sort_time"] = rec.get("created_at") or rec.get("closed_at") or 0
    all_records.sort(key=lambda r: r["_sort_time"], reverse=True)

    # Apply pagination
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    records = all_records[offset:offset + page_size]

    conn.close()

    return {
        "positions": records,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 0,
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
    realized_pnl = closed_pnl + (row["open_realized_pnl"] or 0)

    # Today's PnL: closed positions since start of today (UTC+8)
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

    # Current balance = initial + realized PnL
    initial_balance = settings.sim_initial_balance
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

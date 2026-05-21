from __future__ import annotations

import time
import threading

import ccxt

from app.database import get_connection

# 持久化 K 线缓存 —— 进程启动时从 DB 加载，后续增量更新写回 DB
# key: "{symbol}:{timeframe}" -> list[candle dict]
_kline_cache: dict[str, list] = {}
_last_fetch: dict[str, float] = {}
_FETCH_INTERVAL = 15  # 两次请求之间的最小间隔（秒）

_funding_cache: dict = {"rate": 0.0, "ts": 0.0}
_FUNDING_CACHE_TTL = 300  # 5 minutes

_oi_cache: dict = {"value": 0.0, "prev": 0.0, "ts": 0.0}
_OI_CACHE_TTL = 300  # 5 minutes, align with signal cycle

_exchange: ccxt.binance = None
_exchange_lock = threading.Lock()
_cache_lock = threading.Lock()  # 保护 _kline_cache 的并发访问
_funding_lock = threading.Lock()  # 保护 _funding_cache 的并发访问
_oi_lock = threading.Lock()  # 保护 _oi_cache 的并发访问

# 每个时间框架需要保留的最大K线数量（内存）
_MAX_CANDLES = {
    "30m": 200,
    "1h": 200,
    "4h": 100,
}


def _get_exchange() -> ccxt.binance:
    global _exchange
    if _exchange is None:
        with _exchange_lock:
            if _exchange is None:
                _exchange = ccxt.binance({
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap"},
                })
    return _exchange


def _cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}:{timeframe}"


def load_klines_from_db(symbol: str):
    """从 SQLite 加载最近的K线到内存缓存。"""
    with _cache_lock:
        for tf, max_n in _MAX_CANDLES.items():
            key = _cache_key(symbol, tf)
            if key in _kline_cache:
                continue
            try:
                conn = get_connection()
                rows = conn.execute(
                    "SELECT ts, open, high, low, close, volume FROM klines "
                    "WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT ?",
                    (symbol, tf, max_n),
                ).fetchall()
                conn.close()
                if rows:
                    candles = [
                        {"timestamp": r[0], "open": r[1], "high": r[2],
                         "low": r[3], "close": r[4], "volume": r[5]}
                        for r in reversed(rows)
                    ]
                    _kline_cache[key] = candles
            except Exception:
                pass


def _save_kline_to_db(symbol: str, timeframe: str, candle: dict):
    """写入新 K 线到 SQLite（INSERT OR IGNORE，避免重复）。"""
    _bulk_save_klines(symbol, timeframe, [candle])


def _update_klines_in_db(symbol: str, timeframe: str, candles: list[dict]):
    """更新已有 K 线的实时数据（未收盘→收盘的 finalization）。"""
    if not candles:
        return
    try:
        conn = get_connection()
        conn.executemany(
            "INSERT OR REPLACE INTO klines (symbol, timeframe, ts, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(symbol, timeframe, c["timestamp"], c["open"], c["high"],
              c["low"], c["close"], c["volume"]) for c in candles],
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def fetch_klines_from_db(symbol: str, timeframe: str, limit: int = 200) -> list[dict]:
    """从数据库读取 K 线数据（不请求 Binance API）。

    用于信号验证等场景，避免网络请求的依赖和延迟。
    """
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM klines "
            "WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT ?",
            (symbol, timeframe, limit),
        ).fetchall()
        conn.close()
        if rows:
            return [
                {"timestamp": r[0], "open": r[1], "high": r[2],
                 "low": r[3], "close": r[4], "volume": r[5]}
                for r in reversed(rows)
            ]
    except Exception:
        pass
    return []


def fetch_price(symbol: str) -> float:
    """Fetch latest price only (no OHLCV cache)."""
    ex = _get_exchange()
    ticker = ex.fetch_ticker(symbol)
    return round(ticker.get("last", 0), 1)


def fetch_klines(symbol: str, timeframe: str, limit: int = 200) -> list[dict]:
    """Fetch OHLCV from Binance with persistent incremental cache (SQLite-backed).

    首次启动从 DB 加载历史，后续只做增量更新并写回 DB。
    Binance 的 fetch_ohlcv 返回未收盘 K 线的实时 close/high/low，
    无需额外 ticker 注入，信号引擎的指标计算自动基于实时价格。
    """
    key = _cache_key(symbol, timeframe)
    now = time.time()
    max_candles = _MAX_CANDLES.get(timeframe, limit)

    with _cache_lock:
        has_cache = key in _kline_cache and len(_kline_cache[key]) > 0
        if has_cache and now - _last_fetch.get(key, 0) < _FETCH_INTERVAL:
            return _kline_cache[key][-limit:] if limit else _kline_cache[key][:]

    if not has_cache:
        # 首次全量拉取
        try:
            exchange = _get_exchange()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=max_candles)
        except Exception:
            return []

        if ohlcv:
            candles = []
            for candle in ohlcv:
                candles.append({
                    "timestamp": candle[0] / 1000,
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                })
            with _cache_lock:
                _kline_cache[key] = candles
                _last_fetch[key] = now
                result = candles[-limit:] if limit else candles[:]
            # DB 写入在锁外执行
            _bulk_save_klines(symbol, timeframe, candles)
            return result
        return []

    # 缓存存在，增量拉取（Binance 返回未收盘 K 线的实时数据）
    try:
        exchange = _get_exchange()
        with _cache_lock:
            last_ts = _kline_cache[key][-1]["timestamp"]
        # 不限制 limit，让 Binance 返回所有缺失的 K 线（覆盖宕机缺口）
        ohlcv = exchange.fetch_ohlcv(
            symbol, timeframe, since=int(last_ts * 1000)
        )
    except Exception:
        with _cache_lock:
            return _kline_cache[key][-limit:] if limit else _kline_cache[key][:]

    if ohlcv:
        new_candles = []
        for candle in ohlcv:
            new_candles.append({
                "timestamp": candle[0] / 1000,
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            })

        with _cache_lock:
            # 更新所有匹配的 K 线（包含未收盘 K 线的实时数据）
            ts_set = {c["timestamp"] for c in _kline_cache[key]}
            to_save = []
            updated = []
            for c in new_candles:
                if c["timestamp"] in ts_set:
                    # 已有 K 线 → 更新实时数据（替换整根K线）
                    idx = next((i for i, x in enumerate(_kline_cache[key]) if x["timestamp"] == c["timestamp"]), -1)
                    if idx >= 0:
                        _kline_cache[key][idx] = c
                    updated.append(c)
                else:
                    # 新 K 线 → 追加 + 标记写 DB
                    _kline_cache[key].append(c)
                    to_save.append(c)

            # 裁切到最大长度（仅内存，DB 保留完整历史）
            if len(_kline_cache[key]) > max_candles:
                _kline_cache[key] = _kline_cache[key][-max_candles:]

            _last_fetch[key] = now
            result = _kline_cache[key][-limit:] if limit else _kline_cache[key][:]

        # DB 写入在锁外执行，避免阻塞其他时间框架的 fetch
        # 新 K 线用 INSERT OR IGNORE，已有 K 线更新用 INSERT OR REPLACE
        for c in to_save:
            _save_kline_to_db(symbol, timeframe, c)
        if updated:
            _update_klines_in_db(symbol, timeframe, updated)

        return result
    else:
        with _cache_lock:
            _last_fetch[key] = now  # prevent rapid retry loops on empty response
            return _kline_cache[key][-limit:] if limit else _kline_cache[key][:]


def _bulk_save_klines(symbol: str, timeframe: str, candles: list[dict]):
    """批量写入 K 线到 SQLite。"""
    try:
        conn = get_connection()
        conn.executemany(
            "INSERT OR IGNORE INTO klines (symbol, timeframe, ts, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(symbol, timeframe, c["timestamp"], c["open"], c["high"],
              c["low"], c["close"], c["volume"]) for c in candles],
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def fetch_funding_rate(symbol: str) -> float:
    """Fetch Binance funding rate with 5min cache. Returns rate as decimal (e.g. 0.0001 = 0.01%)."""
    now = time.time()
    with _funding_lock:
        if now - _funding_cache["ts"] < _FUNDING_CACHE_TTL:
            return _funding_cache["rate"]
    try:
        ex = _get_exchange()
        info = ex.fetch_funding_rate(symbol)
        rate = info.get("fundingRate", 0.0)
        with _funding_lock:
            _funding_cache["rate"] = rate
            _funding_cache["ts"] = now
        return rate
    except Exception:
        return _funding_cache["rate"]


def fetch_open_interest(symbol: str) -> tuple[float, float]:
    """Fetch Binance open interest. Returns (current, previous) amounts."""
    now = time.time()
    with _oi_lock:
        if now - _oi_cache["ts"] < _OI_CACHE_TTL:
            return (_oi_cache["value"], _oi_cache["prev"])
    try:
        ex = _get_exchange()
        info = ex.fetch_open_interest(symbol)
        value = info.get("openInterestAmount", 0.0)
        with _oi_lock:
            _oi_cache["prev"] = _oi_cache["value"]
            _oi_cache["value"] = value
            _oi_cache["ts"] = now
        return (value, _oi_cache["prev"])
    except Exception:
        return (_oi_cache["value"], _oi_cache["prev"])


def fetch_all_market_data(symbol: str, timeframes: list[str] = None, limits: dict = None) -> dict:
    """Fetch klines for multiple timeframes + funding rate + OI in parallel threads.

    Args:
        symbol: trading pair symbol
        timeframes: list of timeframe strings (default: ["30m", "1h", "4h"])
        limits: dict mapping timeframe to candle count (default: 30m=200, 1h=200, 4h=100)

    Returns dict with:
      klines_30m, klines_1h, klines_4h: list[dict]
      funding_rate: float
      open_interest: float
      open_interest_prev: float
    """
    if timeframes is None:
        timeframes = ["30m", "1h", "4h"]
    if limits is None:
        limits = {"30m": 200, "1h": 200, "4h": 100}

    results: dict = {}
    errors: list[str] = []

    def _fetch_klines(tf: str):
        try:
            limit = limits.get(tf, 200)
            candles = fetch_klines(symbol, tf, limit=limit)
            if candles:
                results[f"klines_{tf}"] = candles
        except Exception as e:
            errors.append(f"klines_{tf}: {e}")

    def _fetch_funding():
        now = time.time()
        with _funding_lock:
            if now - _funding_cache["ts"] < _FUNDING_CACHE_TTL:
                results["funding_rate"] = _funding_cache["rate"]
                return
        try:
            ex = _get_exchange()
            info = ex.fetch_funding_rate(symbol)
            rate = info.get("fundingRate", 0.0)
            with _funding_lock:
                _funding_cache["rate"] = rate
                _funding_cache["ts"] = now
            results["funding_rate"] = rate
        except Exception as e:
            errors.append(f"funding: {e}")

    def _fetch_oi():
        now = time.time()
        with _oi_lock:
            if now - _oi_cache["ts"] < _OI_CACHE_TTL:
                results["open_interest"] = _oi_cache["value"]
                results["open_interest_prev"] = _oi_cache["prev"]
                return
        try:
            ex = _get_exchange()
            info = ex.fetch_open_interest(symbol)
            value = info.get("openInterestAmount", 0.0)
            with _oi_lock:
                _oi_cache["prev"] = _oi_cache["value"]
                _oi_cache["value"] = value
                _oi_cache["ts"] = now
            results["open_interest"] = value
            results["open_interest_prev"] = _oi_cache["prev"]
        except Exception as e:
            errors.append(f"oi: {e}")

    threads = [
        threading.Thread(target=_fetch_klines, args=(tf,)) for tf in timeframes
    ] + [
        threading.Thread(target=_fetch_funding),
        threading.Thread(target=_fetch_oi),
    ]

    # Warm exchange once before starting threads to avoid race condition
    _get_exchange()

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    return results

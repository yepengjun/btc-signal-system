from __future__ import annotations

import time
import threading

import ccxt

_kline_cache: dict[str, list] = {}
_last_fetch: dict[str, float] = {}
CACHE_TTL = 55  # seconds, slightly less than 1min candle close

_funding_cache: dict = {"rate": 0.0, "ts": 0.0}
_FUNDING_CACHE_TTL = 300  # 5 minutes

_oi_cache: dict = {"value": 0.0, "prev": 0.0, "ts": 0.0}
_OI_CACHE_TTL = 300  # 5 minutes, align with signal cycle

_exchange: ccxt.binance = None
_exchange_lock = threading.Lock()


def _get_exchange() -> ccxt.binance:
    global _exchange
    if _exchange is None:
        with _exchange_lock:
            if _exchange is None:  # double-check after acquiring lock
                _exchange = ccxt.binance({
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap"},
                })
    return _exchange


def fetch_price(symbol: str) -> float:
    """Fetch latest price only (no OHLCV cache)."""
    ex = _get_exchange()
    ticker = ex.fetch_ticker(symbol)
    return round(ticker.get("last", 0), 1)


def _cache_key(symbol: str, timeframe: str, limit: int) -> str:
    return f"{symbol}:{timeframe}:{limit}"


def fetch_klines(symbol: str, timeframe: str, limit: int = 200) -> list[dict]:
    """Fetch OHLCV from Binance and return list of dicts."""
    key = _cache_key(symbol, timeframe, limit)
    now = time.time()
    if key in _kline_cache and now - _last_fetch.get(key, 0) < CACHE_TTL:
        return _kline_cache[key]

    exchange = _get_exchange()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

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

    _kline_cache[key] = candles
    _last_fetch[key] = now
    return candles


def fetch_funding_rate(symbol: str) -> float:
    """Fetch Binance funding rate with 5min cache. Returns rate as decimal (e.g. 0.0001 = 0.01%)."""
    now = time.time()
    if now - _funding_cache["ts"] < _FUNDING_CACHE_TTL:
        return _funding_cache["rate"]
    try:
        ex = _get_exchange()
        info = ex.fetch_funding_rate(symbol)
        rate = info.get("fundingRate", 0.0)
        _funding_cache["rate"] = rate
        _funding_cache["ts"] = now
        return rate
    except Exception:
        return _funding_cache["rate"]


def fetch_open_interest(symbol: str) -> tuple[float, float]:
    """Fetch Binance open interest. Returns (current, previous) amounts."""
    now = time.time()
    if now - _oi_cache["ts"] < _OI_CACHE_TTL:
        return (_oi_cache["value"], _oi_cache["prev"])
    try:
        ex = _get_exchange()
        info = ex.fetch_open_interest(symbol)
        value = info.get("openInterestAmount", 0.0)
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
            key = _cache_key(symbol, tf, limit)
            now = time.time()
            if key in _kline_cache and now - _last_fetch.get(key, 0) < CACHE_TTL:
                results[f"klines_{tf}"] = _kline_cache[key]
                return
            exchange = _get_exchange()
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            candles = [{
                "timestamp": c[0] / 1000, "open": c[1], "high": c[2],
                "low": c[3], "close": c[4], "volume": c[5],
            } for c in ohlcv]
            _kline_cache[key] = candles
            _last_fetch[key] = now
            results[f"klines_{tf}"] = candles
        except Exception as e:
            errors.append(f"klines_{tf}: {e}")

    def _fetch_funding():
        now = time.time()
        if now - _funding_cache["ts"] < _FUNDING_CACHE_TTL:
            results["funding_rate"] = _funding_cache["rate"]
            return
        try:
            ex = _get_exchange()
            info = ex.fetch_funding_rate(symbol)
            rate = info.get("fundingRate", 0.0)
            _funding_cache["rate"] = rate
            _funding_cache["ts"] = now
            results["funding_rate"] = rate
        except Exception as e:
            errors.append(f"funding: {e}")

    def _fetch_oi():
        now = time.time()
        if now - _oi_cache["ts"] < _OI_CACHE_TTL:
            results["open_interest"] = _oi_cache["value"]
            results["open_interest_prev"] = _oi_cache["prev"]
            return
        try:
            ex = _get_exchange()
            info = ex.fetch_open_interest(symbol)
            value = info.get("openInterestAmount", 0.0)
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

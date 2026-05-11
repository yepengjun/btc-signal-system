from __future__ import annotations

import time

import ccxt

_kline_cache: dict[str, list] = {}
_last_fetch: dict[str, float] = {}
CACHE_TTL = 55  # seconds, slightly less than 1min candle close

_funding_cache: dict = {"rate": 0.0, "ts": 0.0}
_FUNDING_CACHE_TTL = 300  # 5 minutes

_oi_cache: dict = {"value": 0.0, "prev": 0.0, "ts": 0.0}
_OI_CACHE_TTL = 300  # 5 minutes, align with signal cycle

_exchange: ccxt.binance = None


def _get_exchange() -> ccxt.binance:
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({"enableRateLimit": True})
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

    exchange = ccxt.binance({"enableRateLimit": True})
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

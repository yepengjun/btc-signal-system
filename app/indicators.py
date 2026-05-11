from __future__ import annotations

"""Technical indicators: EMA, ATR, ADX, DI."""

import math


def calc_ema(values: list[float], period: int) -> list[float]:
    """Exponential moving average."""
    k = 2 / (period + 1)
    ema = [values[0]]
    for i in range(1, len(values)):
        ema.append(values[i] * k + ema[-1] * (1 - k))
    return ema


def calc_sma(values: list[float], period: int) -> list[float]:
    """Simple moving average."""
    sma = []
    for i in range(len(values)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(sum(values[i - period + 1 : i + 1]) / period)
    return sma


def calc_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float:
    """Average True Range — returns the latest value."""
    series = calc_atr_series(highs, lows, closes, period)
    return series[-1] if series else 0


def calc_atr_series(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[float]:
    """Average True Range time series — returns all computed values."""
    if len(highs) < 2:
        return []
    true_ranges = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        avg = sum(true_ranges) / len(true_ranges)
        return [avg] * len(true_ranges)

    atr = sum(true_ranges[:period]) / period
    result = [atr]
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period
        result.append(atr)
    return result


def calc_atr_percentile(atr_values: list[float], window: int = 200) -> float:
    """Percentile of the latest ATR within the last `window` values.

    Returns 0-100. < 25 = low vol, 25-75 = normal, > 75 = high vol.
    """
    if not atr_values:
        return 50.0
    recent = atr_values[-window:]
    current = recent[-1]
    sorted_vals = sorted(recent)
    rank = 0
    for v in sorted_vals:
        if v <= current:
            rank += 1
        else:
            break
    return round(rank / max(len(sorted_vals), 1) * 100, 1)


def calc_adx(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> dict:
    """
    Calculate ADX, +DI, -DI.
    Returns dict with latest values: adx, plus_di, minus_di.
    """
    n = len(highs)
    if n < period * 2 + 1:
        return {"adx": 0, "plus_di": 0, "minus_di": 0, "plus_dm": 0, "minus_dm": 0}

    plus_dm = [0.0]
    minus_dm = [0.0]
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

    def _wilder_smooth(series: list[float], period: int) -> list[float]:
        result = [sum(series[1 : period + 1])]
        for i in range(period + 1, len(series)):
            result.append(result[-1] - result[-1] / period + series[i])
        return result

    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )

    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    plus_di = []
    minus_di = []
    for i in range(len(smoothed_tr)):
        if smoothed_tr[i] == 0:
            plus_di.append(0)
            minus_di.append(0)
        else:
            plus_di.append(100 * smoothed_plus_dm[i] / smoothed_tr[i])
            minus_di.append(100 * smoothed_minus_dm[i] / smoothed_tr[i])

    dx = []
    for i in range(len(plus_di)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum == 0:
            dx.append(0)
        else:
            dx.append(100 * abs(plus_di[i] - minus_di[i]) / di_sum)

    adx_period = period
    if len(dx) < adx_period:
        adx_val = sum(dx) / len(dx) if dx else 0
    else:
        adx_val = sum(dx[:adx_period]) / adx_period
        for i in range(adx_period, len(dx)):
            adx_val = (adx_val * (adx_period - 1) + dx[i]) / adx_period

    return {
        "adx": round(adx_val, 1),
        "plus_di": round(plus_di[-1], 1),
        "minus_di": round(minus_di[-1], 1),
        "plus_dm": round(smoothed_plus_dm[-1], 1),
        "minus_dm": round(smoothed_minus_dm[-1], 1),
        "plus_di_series": [round(v, 1) for v in plus_di[-20:]],
        "minus_di_series": [round(v, 1) for v in minus_di[-20:]],
        "dx_series": [round(v, 1) for v in dx[-20:]],
    }

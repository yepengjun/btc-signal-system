"""Evolution module: signal verification and autonomous self-evolution.

This module:
1. Verifies past signals against actual market behavior
2. Tracks accuracy by regime, ADX range, and momentum
3. Autonomously adjusts signal thresholds based on historical performance
"""

import time
import json
import os

from app.database import get_connection
from app.binance import fetch_klines
from app.config import settings

# Verification: how many candles AFTER the signal candle to check.
# Extended from 2→3 to give trends enough time to develop.
# 30m: 3 candles = 1.5h (was 1h — too short to confirm)
# 1h:  3 candles = 3h  (was 2h — forming signals especially need more time)
# 4h:  3 candles = 12h (was 8h — borderline for breakout confirmation)
VERIFY_CANDLES = {
    "30m": 3,
    "1h": 3,
    "4h": 3,
}

# Evolution state file path
EVOLUTION_STATE_PATH = os.path.join(os.path.dirname(settings.db_path), "evolution.json")

DEFAULT_EVOLUTION_PARAMS = {
    "adx_trending_threshold": 25,
    "adx_forming_threshold": 20,
    "adx_exit_threshold": 20,
    "di_crossover_window": 2,
    "adx_drop_reduce": 5,
    "di_diff_reduce": 5,
    "price_divergence_threshold": 5,
    "momentum_accel_threshold": 3,
    "confidence_base": 50,
    "min_di_spread": 3,
    "vol_adjustment_factor": 0.1,
    "atr_stop_multiplier": 1.5,
    "rr_ratio": 1.5,
    "last_updated": None,
    # Per-timeframe overrides (evolved independently)
    "tf_thresholds": {
        "30m": {"adx_trending_threshold": 25, "adx_forming_threshold": 20, "adx_period": 14, "adx_fast_period": 10, "adx_slow_period": 21},
        "1h": {"adx_trending_threshold": 25, "adx_forming_threshold": 20, "adx_period": 14, "adx_fast_period": 10, "adx_slow_period": 21},
        "4h": {"adx_trending_threshold": 25, "adx_forming_threshold": 20, "adx_period": 10, "adx_fast_period": 7, "adx_slow_period": 14},
    },
}

# Hard bounds: thresholds can't drift beyond these limits
# Tightened: adx_trending upper cap 35→28 to prevent runaway (1h drifted to 31)
# Floor lowered 20→18 so 4h can adapt to lower-vol regimes
THRESHOLD_BOUNDS = {
    "adx_trending_threshold": (18, 28),
    "adx_forming_threshold": (15, 26),
    "atr_stop_multiplier": (1.0, 2.2),
    "rr_ratio": (1.2, 2.2),
}


def _load_evolution_params() -> dict:
    """Load evolution parameters from state file."""
    if os.path.exists(EVOLUTION_STATE_PATH):
        try:
            with open(EVOLUTION_STATE_PATH, "r") as f:
                params = json.load(f)
            for key, val in DEFAULT_EVOLUTION_PARAMS.items():
                if key not in params:
                    params[key] = val
            # Ensure tf_thresholds exists and is complete
            if "tf_thresholds" not in params:
                params["tf_thresholds"] = {}
            default_tf = DEFAULT_EVOLUTION_PARAMS["tf_thresholds"]
            for tf in ["30m", "1h", "4h"]:
                if tf not in params["tf_thresholds"]:
                    params["tf_thresholds"][tf] = {
                        "adx_trending_threshold": params["adx_trending_threshold"],
                        "adx_forming_threshold": params["adx_forming_threshold"],
                    }
                # Seed missing ADX period fields from defaults
                for key in ["adx_period", "adx_fast_period", "adx_slow_period"]:
                    if key not in params["tf_thresholds"][tf]:
                        params["tf_thresholds"][tf][key] = default_tf.get(tf, {}).get(key, 14)
            # Clamp all threshold values to hard bounds (fix drift from previous sessions)
            for tf in ["30m", "1h", "4h"]:
                tf_cfg = params["tf_thresholds"][tf]
                for key in ["adx_trending_threshold", "adx_forming_threshold"]:
                    if key in tf_cfg:
                        tf_cfg[key] = _clamp(key, tf_cfg[key])
            # Also clamp global values
            for key in ["adx_trending_threshold", "adx_forming_threshold", "atr_stop_multiplier", "rr_ratio"]:
                if key in params:
                    params[key] = _clamp(key, params[key])
            return params
        except (json.JSONDecodeError, IOError):
            pass
    return dict(DEFAULT_EVOLUTION_PARAMS)


def _save_evolution_params(params: dict):
    """Save evolution parameters to state file."""
    params["last_updated"] = time.time()
    os.makedirs(os.path.dirname(EVOLUTION_STATE_PATH), exist_ok=True)
    with open(EVOLUTION_STATE_PATH, "w") as f:
        json.dump(params, f, indent=2)


def _find_signal_candle(candles: list, created_at: float):
    """Find candle index closest to (but not after) signal creation time.

    Kline timestamps are at candle open; created_at is slightly after.
    We look for the candle whose open time is <= created_at and closest to it.
    """
    best_idx = None
    best_dist = float("inf")
    for i, c in enumerate(candles):
        if c["timestamp"] <= created_at:
            dist = created_at - c["timestamp"]
            if dist < best_dist:
                best_dist = dist
                best_idx = i
    return best_idx


def verify_pending_signals():
    """Check pending signals and verify them against actual market behavior.

    Uses timestamp-based candle matching (not price matching) to find
    the exact signal candle, then verifies the NEXT N candles.
    """
    conn = get_connection()
    now = time.time()

    for tf, _ in VERIFY_CANDLES.items():
        rows = conn.execute(
            "SELECT id, timeframe, direction, price_at_signal, adx, regime, "
            "momentum, target, stop, created_at, signal_type "
            "FROM signals WHERE verified = 0 AND timeframe = ?",
            (tf,),
        ).fetchall()

        for row in rows:
            sig_id = row["id"]
            direction = row["direction"]
            price_at_signal = row["price_at_signal"]
            created_at = row["created_at"]

            # Ranging/low_vol signals have direction=None — skip direction check,
            # but still evaluate regime correctness (no significant price move).
            is_ranging = direction is None
            if price_at_signal is None:
                conn.execute(
                    "UPDATE signals SET verified = 1 WHERE id = ?", (sig_id,)
                )
                continue

            candles = fetch_klines(settings.binance_symbol, tf, limit=200)
            if not candles or len(candles) < 10:
                continue

            # Find signal candle by timestamp matching
            signal_idx = _find_signal_candle(candles, created_at)
            if signal_idx is None:
                continue

            # Validate: matched candle must be within 1.5 candle intervals of signal time.
            # If the signal is older than the 200-candle window, skip it.
            candle_seconds = {"30m": 1800, "1h": 3600, "4h": 14400}
            max_age = candle_seconds.get(tf, 3600) * 1.5
            time_diff = created_at - candles[signal_idx]["timestamp"]
            if time_diff > max_age:
                # Signal is outside the 200-candle window — can't verify accurately.
                # Mark as unverifiable (excluded from accuracy stats) instead of
                # silently marking as verified which would pull down accuracy.
                conn.execute(
                    "UPDATE signals SET verified = 1, unverifiable = 1 WHERE id = ?", (sig_id,)
                )
                continue

            # Verify the NEXT N candles after the signal candle
            n_verify = VERIFY_CANDLES[tf]
            verify_start = signal_idx + 1
            verify_end = min(verify_start + n_verify, len(candles))

            if verify_start >= len(candles):
                # Signal too recent, not enough candles yet
                continue

            # Require full verify window — skip if not enough candles formed yet.
            # Without this, verification runs on partial candles and gives
            # random results (e.g., 0.01% move flagged as "wrong direction").
            if verify_end - verify_start < n_verify:
                continue

            verify_candles = candles[verify_start:verify_end]
            current_price = verify_candles[-1]["close"]

            # MFE/MAE across ALL candles from signal through verification window
            all_post = candles[signal_idx:verify_end]
            all_high = max(c["high"] for c in all_post)
            all_low = min(c["low"] for c in all_post)

            # ─── Pre-compute: ATR-based dynamic thresholds ───────────────────
            # Avoids fixed-percentage bias across volatility regimes.
            from app.indicators import calc_atr, calc_adx as calc_adx_func
            from app.evolution import _load_evolution_params
            highs_all = [c["high"] for c in candles]
            lows_all = [c["low"] for c in candles]
            closes_all = [c["close"] for c in candles]
            regime_at_signal = row["regime"]

            # Per-TF ADX period for verification consistency
            evol_params = _load_evolution_params()
            tf_cfg = evol_params.get("tf_thresholds", {}).get(tf, {})
            adx_verify_period = tf_cfg.get("adx_period", 14)

            atr_value = calc_atr(highs_all, lows_all, closes_all, period=14)
            atr_pct_price = (atr_value / max(price_at_signal, 1)) * 100

            # Signal type: check if this is an exhaustion reversal signal
            is_reversal_signal = dict(row).get("signal_type") == "exhaustion_reversal"

            # Signal type: check if this is a ranging breakout signal
            is_ranging_breakout = dict(row).get("signal_type") == "ranging_breakout"

            # Capture full signal_type string for Stage 1 detection
            signal_type_str = dict(row).get("signal_type", "trend_following")

            # Dynamic minimum move: based on ATR, adapts to current volatility.
            # A 0.3×ATR bar is meaningful; 0.5× is a "significant move".
            # Floor protection: even when ATR is extremely low, don't demand zero movement.
            tf_atr_mult_min = {"30m": 0.2, "1h": 0.3, "4h": 0.4}
            tf_atr_mult_sig = {"30m": 0.3, "1h": 0.5, "4h": 0.25}
            tf_min_move_floor = {"30m": 0.05, "1h": 0.08, "4h": 0.10}
            min_move_dynamic = max(atr_pct_price * tf_atr_mult_min.get(tf, 0.3), tf_min_move_floor.get(tf, 0.05))
            sig_move_dynamic = max(atr_pct_price * tf_atr_mult_sig.get(tf, 0.5), tf_min_move_floor.get(tf, 0.05) * 1.5)

            # Pre-compute move magnitude — needed by direction check below.
            move_pct = abs(current_price - price_at_signal) / max(price_at_signal, 1) * 100

            # 1. Direction check (for ranging signals, direction_correct = 1 since direction N/A)
            # For directional signals, require at least the minimum ATR-based move.
            # Price moving 0.01% in the "right" direction is noise, not confirmation.
            if is_ranging:
                direction_correct = 1
                actual_direction = "neutral"
            else:
                move_direction = (
                    "bullish" if current_price >= price_at_signal else "bearish"
                )
                # Only count as direction-correct if move exceeds minimum ATR threshold
                if move_direction == direction and move_pct >= min_move_dynamic:
                    direction_correct = 1
                else:
                    direction_correct = 0
                # actual_direction based on actual price movement
                actual_direction = move_direction

            # 2. Magnitude check — now ATR-based (replaces fixed percentages)
            significant_move = move_pct >= sig_move_dynamic

            # 2b. Ranging-specific: check that no trend developed.
            # For ranging, the key question is: did the market stay non-trending?
            # Price range alone is misleading — a choppy market can have large wicks
            # but no real directional bias. So we check:
            #   (1) No significant directional move occurred, AND
            #   (2) Verification-period ADX stayed below trending level.
            # high_volatility gets extra room since it implies wider swings.
            price_range_pct = (all_high - all_low) / max(price_at_signal, 1) * 100

            adx_verify = calc_adx_func(highs_all, lows_all, closes_all, period=adx_verify_period)
            verify_adx = adx_verify["adx"]

            # No significant trend developed?
            # ATR-based move threshold replaces fixed percentages.
            no_trend = move_pct < sig_move_dynamic or verify_adx < 28

            # Also check price range didn't explode beyond even high-vol bounds
            if regime_at_signal == "high_volatility":
                tf_atr_mult = {"30m": 5.0, "1h": 4.5, "4h": 3.5}
            else:
                tf_atr_mult = {"30m": 3.5, "1h": 3.5, "4h": 2.5}
            atr_threshold_pct = (atr_value / max(price_at_signal, 1)) * 100 * tf_atr_mult.get(tf, 1.2)
            range_ok = price_range_pct <= atr_threshold_pct

            # Ranging is correct if: no significant trend developed AND
            # range didn't explode beyond extreme volatility bounds.
            # range_ok is a safety valve for true breakouts, not a hard gate.
            ranging_correct = no_trend and range_ok

            # 2c. Price structure validation — verify trend integrity.
            # For exhaustion reversal or ranging breakout signals, skip price structure check:
            # reversals often oscillate initially (V/W patterns), so strict
            # candle-by-candle direction checks produce false failures.
            # Ranging breakouts are short-term and noisy.
            verify_candle_list = candles[verify_start:verify_end]
            if is_reversal_signal or is_ranging_breakout:
                price_structure_ok = True
            elif direction == "bearish" and len(verify_candle_list) >= 2:
                # Bearish: require closes to be mostly declining,
                # and the last candle's close near its lower range.
                declines = sum(
                    1 for i in range(1, len(verify_candle_list))
                    if verify_candle_list[i]["close"] < verify_candle_list[i - 1]["close"]
                )
                last_candle = verify_candle_list[-1]
                candle_range = last_candle["high"] - last_candle["low"]
                close_near_low = (
                    (last_candle["close"] - last_candle["low"]) / max(candle_range, 1) < 0.4
                    if candle_range > 0 else True
                )
                price_structure_ok = (declines >= len(verify_candle_list) // 2) and close_near_low
            elif direction == "bullish" and len(verify_candle_list) >= 2:
                # Bullish: require closes to be mostly rising,
                # and the last candle's close near its upper range.
                advances = sum(
                    1 for i in range(1, len(verify_candle_list))
                    if verify_candle_list[i]["close"] > verify_candle_list[i - 1]["close"]
                )
                last_candle = verify_candle_list[-1]
                candle_range = last_candle["high"] - last_candle["low"]
                close_near_high = (
                    (last_candle["high"] - last_candle["close"]) / max(candle_range, 1) < 0.4
                    if candle_range > 0 else True
                )
                price_structure_ok = (advances >= len(verify_candle_list) // 2) and close_near_high
            else:
                price_structure_ok = True  # too few candles or ranging — skip

            # 3. Regime correctness — decomposed into sub-metrics
            from app.indicators import calc_adx
            adx_data = calc_adx(highs_all, lows_all, closes_all, period=adx_verify_period)
            current_adx = adx_data["adx"]

            # Sub-metrics: independent outputs for evolution system
            move_sufficient = 1 if move_pct >= min_move_dynamic else 0
            structure_aligned = 1 if price_structure_ok else 0

            if regime_at_signal == "trending":
                # Direction correct AND minimum move AND price structure (strict mode).
                # In low-volatility markets, direction correct + one auxiliary is sufficient (loose mode).
                trend_persisted = (
                    direction_correct
                    and move_sufficient
                    and structure_aligned
                )
                # Loose: direction correct + meaningful move. Structure check is too
                # noisy on short verify windows (3 candles can easily break structure
                # even when the trend goes the right way).
                regime_correct_loose = 1 if (direction_correct and move_sufficient) else 0
            elif regime_at_signal == "forming":
                # Forming = "a trend is developing". ADX should show some rise,
                # but +2 is too strict for early-stage signals. Use +1.
                adx_at_signal = row["adx"] or 0
                adx_rising = current_adx >= adx_at_signal + 1
                has_price_action = move_pct >= (min_move_dynamic * 0.5)
                trend_persisted = adx_rising and has_price_action
                # Loose: direction correct + any meaningful move, no ADX gate.
                # Captures trends where price moved correctly but ADX hasn't caught up yet.
                regime_correct_loose = 1 if (direction_correct and move_sufficient) else 0
            elif regime_at_signal == "low_vol_trend":
                trend_persisted = direction_correct and current_adx >= 25
                regime_correct_loose = direction_correct and current_adx >= 20
            elif regime_at_signal in ("breakout", "exhaustion"):
                if is_reversal_signal:
                    # Reversal signals start with small moves, not large ones.
                    # Require direction correct + minimal move (half the normal bar).
                    trend_persisted = direction_correct and move_pct >= (min_move_dynamic * 0.5)
                else:
                    trend_persisted = significant_move and direction_correct
                regime_correct_loose = direction_correct
            elif is_ranging_breakout:
                # Ranging breakout: short-term signal, relaxed verification.
                # Direction correct + any meaningful move (half the normal bar).
                # These are inherently lower-confidence, smaller targets.
                trend_persisted = direction_correct and move_pct >= (min_move_dynamic * 0.5)
                regime_correct_loose = direction_correct
            elif regime_at_signal in ("ranging", "low_volatility", "high_volatility"):
                trend_persisted = ranging_correct
                regime_correct_loose = 1 if ranging_correct else 0
            else:
                trend_persisted = direction_correct
                regime_correct_loose = direction_correct

            # 4. Target/stop check (only meaningful for directional signals)
            target_hit = 0
            stop_hit = 0
            if not is_ranging:
                target_val = row["target"]
                stop_val = row["stop"]
                if target_val is not None:
                    if direction == "bullish":
                        target_hit = 1 if all_high >= target_val else 0
                    else:
                        target_hit = 1 if all_low <= target_val else 0
                if stop_val is not None:
                    if direction == "bullish":
                        stop_hit = 1 if all_low <= stop_val else 0
                    else:
                        stop_hit = 1 if all_high >= stop_val else 0

            # 5. Max favorable/adverse excursion
            if is_ranging:
                # For ranging, use absolute price range as excursion
                max_favorable = (all_high - price_at_signal) / max(price_at_signal, 1) * 100
                max_adverse = (price_at_signal - all_low) / max(price_at_signal, 1) * 100
            elif direction == "bullish":
                max_favorable = (all_high - price_at_signal) / max(price_at_signal, 1) * 100
                max_adverse = (price_at_signal - all_low) / max(price_at_signal, 1) * 100
            else:
                max_favorable = (price_at_signal - all_low) / max(price_at_signal, 1) * 100
                max_adverse = (all_high - price_at_signal) / max(price_at_signal, 1) * 100

            # ─── Stage 1 Verification ──────────────────────────────────────
            # Stage 1 signals (stage1_*) get specialized metrics:
            #   - post_entry_adx_rise: max ADX rise in 4 candles after signal
            #   - stage2_upgraded: did a Stage 2 (trend_following) signal appear
            #     within 8 candles of the same direction?
            is_stage1 = signal_type_str.startswith("stage1_")
            stage1_type = signal_type_str.replace("stage1_", "") if is_stage1 else None
            post_entry_adx_rise = None
            stage2_upgraded = 0

            if is_stage1 and direction:
                # Compute post-entry ADX: check ADX at verification candle vs signal
                adx_at_signal = row["adx"] or 0
                if verify_adx > adx_at_signal:
                    post_entry_adx_rise = round(verify_adx - adx_at_signal, 1)
                else:
                    post_entry_adx_rise = 0

                # Check stage2_upgraded: look for a same-direction trending signal
                # within 8 candles after this Stage 1 signal.
                window_bars = 8
                window_seconds = candle_seconds.get(tf, 3600) * window_bars
                s2_row = conn.execute(
                    "SELECT id FROM signals WHERE timeframe = ? "
                    "AND direction = ? AND regime = 'trending' "
                    "AND signal_type = 'trend_following' "
                    "AND created_at > ? AND created_at <= ? LIMIT 1",
                    (tf, direction, created_at, created_at + window_seconds),
                ).fetchone()
                if s2_row:
                    stage2_upgraded = 1

            # ─── Longer-term regime check (for trending signals that fail short window)
            # Look 6 more candles beyond the verification window to see if the trend
            # eventually materialized. This prevents premature "regime wrong" verdicts
            # on slow-moving markets where the trend needs more time to develop.
            # Only set for directional trending/forming signals.
            longer_term_valid = 0
            if not trend_persisted and direction and regime_at_signal in ("trending", "forming"):
                lt_extend = 6
                lt_end = min(signal_idx + VERIFY_CANDLES[tf] + lt_extend + 1, len(candles))
                lt_candles = candles[signal_idx + 1:lt_end]
                if len(lt_candles) > VERIFY_CANDLES[tf]:
                    lt_close = lt_candles[-1]["close"]
                    lt_move = (lt_close - price_at_signal) / max(price_at_signal, 1) * 100
                    if direction == "bullish" and lt_close >= price_at_signal:
                        longer_term_valid = 1
                    elif direction == "bearish" and lt_close <= price_at_signal:
                        longer_term_valid = 1

            conn.execute(
                "UPDATE signals SET price_at_verify = ?, actual_trending = ?, "
                "actual_direction = ?, regime_correct = ?, direction_correct = ?, "
                "verified = 1, max_favorable_excursion = ?, max_adverse_excursion = ?, "
                "target_hit = ?, stop_hit = ?, move_pct = ?, "
                "verify_adx = ?, verify_price = ?, verify_time = ?, "
                "stage2_upgraded = ?, move_sufficient = ?, structure_aligned = ?, "
                "regime_correct_loose = ?, longer_term_regime_valid = ? "
                "WHERE id = ?",
                (
                    round(current_price, 1),
                    1 if trend_persisted else 0,
                    actual_direction,
                    1 if trend_persisted else 0,
                    direction_correct,
                    round(max_favorable, 2),
                    round(max_adverse, 2),
                    target_hit,
                    stop_hit,
                    round(move_pct, 2),
                    round(current_adx, 1),
                    round(current_price, 1),
                    now,
                    stage2_upgraded,
                    move_sufficient,
                    structure_aligned,
                    regime_correct_loose,
                    longer_term_valid,
                    sig_id,
                ),
            )

    conn.commit()
    conn.close()


def _clamp(key: str, value: float) -> float:
    """Clamp a parameter value within its hard bounds."""
    if key in THRESHOLD_BOUNDS:
        lo, hi = THRESHOLD_BOUNDS[key]
        return max(lo, min(hi, value))
    return value


def _get_tf_thresholds(params: dict, tf: str) -> dict:
    """Get thresholds for a specific timeframe, falling back to global defaults."""
    tf_cfg = params.get("tf_thresholds", {})
    if tf in tf_cfg:
        return dict(tf_cfg[tf])
    # Fallback: use global values
    return {
        "adx_trending_threshold": params["adx_trending_threshold"],
        "adx_forming_threshold": params["adx_forming_threshold"],
    }


def _compute_evolution_adjustments(stats: dict) -> dict:
    """Compute autonomous adjustments to signal parameters based on evolution stats.

    Per-timeframe threshold evolution:
    - Each timeframe (30m/1h/4h) has its own trending/forming thresholds
    - When a timeframe's regime is wrong (regime_accuracy < 45%): raise its trending threshold
    - When a timeframe's direction is wrong (dir_accuracy < 50%): raise its forming threshold
    - When accurate (regime > 70%, dir > 70%): lower thresholds (trust earlier signals)
    - Global parameters (stop, rr_ratio) still use 4h as primary

    Rules:
    - Trend WRONG (regime_correct=0): raise ADX trending threshold (need stronger signal)
    - Trend RIGHT + high confidence (regime_correct=1, acc>70%): lower threshold
    - Direction WRONG: raise forming threshold (trends starting too early)
    - Direction RIGHT + high acc: lower forming threshold (can trust earlier signals)
    - MFE/MAE ratio > 2.5: stop too wide, tighten
    - MFE/MAE ratio < 1.2: stop too tight, loosen
    - High target_hit rate: can increase profit targets
    - High stop_hit rate: tighten atr_stop_multiplier slightly
    - Minimum 5 verified signals for per-TF adjustments
    - Adjustments are ±1 per cycle with hard bounds to prevent runaway drift
    """
    params = _load_evolution_params()

    # Initialize per-TF thresholds if missing
    if "tf_thresholds" not in params:
        params["tf_thresholds"] = {}
    for tf in ["30m", "1h", "4h"]:
        if tf not in params["tf_thresholds"]:
            params["tf_thresholds"][tf] = {
                "adx_trending_threshold": params["adx_trending_threshold"],
                "adx_forming_threshold": params["adx_forming_threshold"],
            }
        # Track consecutive wrong regime calls (force correction bypasses last_verified)
        if "consecutive_wrong_regime" not in params["tf_thresholds"][tf]:
            params["tf_thresholds"][tf]["consecutive_wrong_regime"] = 0
        # When at floor and still wrong, mark as unreliable
        if "unreliable" not in params["tf_thresholds"][tf]:
            params["tf_thresholds"][tf]["unreliable"] = False

    # Track last verified counts to avoid repeated adjustments on same data
    last_counts = params.get("last_verified_counts", {})
    new_params = {}

    # Per-timeframe adjustments
    for tf in ["30m", "1h", "4h"]:
        tf_data = stats.get("timeframes", {}).get(tf, {})
        regime_acc = tf_data.get("regime_accuracy", 50)
        dir_acc = tf_data.get("dir_accuracy", 50)
        verified = tf_data.get("verified", 0)
        recent = tf_data.get("recent", [])

        tf_trending = _get_tf_thresholds(params, tf)["adx_trending_threshold"]
        tf_forming = _get_tf_thresholds(params, tf)["adx_forming_threshold"]
        tf_cfg = params["tf_thresholds"][tf]

        # Count consecutive wrong regime from recent signals (ordered newest first).
        # Use loose regime for consecutive counting to avoid false "streak" triggers
        # in low-vol markets where direction is correct but move is too small.
        consecutive_wrong = 0
        for r in recent:
            if r.get("verified") and not r.get("regime_correct_loose"):
                consecutive_wrong += 1
            else:
                break

        # ─── Force correction: bypass last_verified when consecutive wrong >= 3
        # Only allow force correction when we have enough verified signals (>=10)
        # to avoid overreacting to short streaks with tiny sample sizes.
        if consecutive_wrong >= 3 and verified >= 10:
            floor, _ = THRESHOLD_BOUNDS["adx_trending_threshold"]
            if tf_trending <= floor + 0.5:
                # Already at floor: only mark unreliable if overall accuracy is also poor.
                # Short-term streaks during a specific regime (e.g. chop) shouldn't
                # permanently disable a timeframe with decent overall accuracy.
                if regime_acc < 60:
                    tf_cfg["unreliable"] = True
                else:
                    tf_cfg["unreliable"] = False  # recover: overall accuracy healthy
            else:
                tf_cfg["unreliable"] = False  # clear unreliable if recovering
                # Force +1 trending threshold (reduced from +2 to avoid runaway drift)
                new_val = _clamp("adx_trending_threshold", tf_trending + 1)
                params["tf_thresholds"][tf]["adx_trending_threshold"] = new_val
            # Reset counter and update last_verified to prevent re-trigger
            tf_cfg["consecutive_wrong_regime"] = 0
            params["last_verified_counts"][tf] = verified
            continue

        # Clear consecutive counter if regime accuracy is healthy
        if regime_acc >= 60:
            tf_cfg["consecutive_wrong_regime"] = 0
            tf_cfg["unreliable"] = False

        # ─── Normal adjustment: only when enough signals verified (>=20)
        # Below 20, use gray-zone micro-tuning only to prevent runaway drift
        # from small sample sizes.
        last_verified = last_counts.get(tf, 0)
        if verified <= last_verified:
            continue

        new_params[tf] = verified  # Mark that we'll adjust for this TF

        if verified >= 20:
            # Full adjustment: regime/dir accuracy driven
            # Regime accuracy → adjust this TF's trending threshold
            if regime_acc > 70:
                new_val = _clamp("adx_trending_threshold", tf_trending - 1)
                if new_val != tf_trending:
                    params["tf_thresholds"][tf]["adx_trending_threshold"] = new_val
            elif regime_acc < 45:
                new_val = _clamp("adx_trending_threshold", tf_trending + 1)
                if new_val != tf_trending:
                    params["tf_thresholds"][tf]["adx_trending_threshold"] = new_val
            else:
                # Gray-zone: micro-tune toward 25 (default)
                if abs(tf_trending - 25) > 0.5:
                    delta = -0.5 if tf_trending > 25 else 0.5
                    new_val = _clamp("adx_trending_threshold", tf_trending + delta)
                    params["tf_thresholds"][tf]["adx_trending_threshold"] = round(new_val, 1)

            # Direction accuracy → adjust this TF's forming threshold
            if dir_acc > 70:
                new_val = _clamp("adx_forming_threshold", tf_forming - 1)
                if new_val != tf_forming:
                    params["tf_thresholds"][tf]["adx_forming_threshold"] = new_val
            elif dir_acc < 50:
                new_val = _clamp("adx_forming_threshold", tf_forming + 1)
                if new_val != tf_forming:
                    params["tf_thresholds"][tf]["adx_forming_threshold"] = new_val
            else:
                if abs(tf_forming - 20) > 0.5:
                    delta = -0.5 if tf_forming > 20 else 0.5
                    new_val = _clamp("adx_forming_threshold", tf_forming + delta)
                    params["tf_thresholds"][tf]["adx_forming_threshold"] = round(new_val, 1)
        elif verified >= 3:
            # Partial data: no adjustment yet. Wait for more verified signals.
            # This prevents threshold drift from small sample sizes.
            pass

    # MFE/MAE-based stop optimization (uses 4h as primary)
    tf_stats = stats.get("timeframes", {}).get("4h", {})
    recent = tf_stats.get("recent", [])
    mfe_values = [r.get("max_favorable_excursion", 0) for r in recent if r.get("max_favorable_excursion") is not None]
    mae_values = [r.get("max_adverse_excursion", 0) for r in recent if r.get("max_adverse_excursion") is not None and r.get("max_adverse_excursion", 0) > 0]

    if mfe_values and mae_values:
        avg_mfe = sum(mfe_values) / len(mfe_values)
        avg_mae = sum(mae_values) / len(mae_values)
        mfe_mae_ratio = avg_mfe / max(avg_mae, 0.1)

        if mfe_mae_ratio > 2.5:
            params["atr_stop_multiplier"] = _clamp(
                "atr_stop_multiplier",
                params.get("atr_stop_multiplier", 1.5) - 0.1,
            )
        elif mfe_mae_ratio < 1.2:
            params["atr_stop_multiplier"] = _clamp(
                "atr_stop_multiplier",
                params.get("atr_stop_multiplier", 1.5) + 0.1,
            )

    # Target hit / Stop hit statistics
    target_hits = [r.get("target_hit", 0) for r in recent if r.get("target_hit") is not None]
    stop_hits = [r.get("stop_hit", 0) for r in recent if r.get("stop_hit") is not None]

    if target_hits:
        target_rate = sum(target_hits) / len(target_hits)
        if target_rate > 0.6:
            params["rr_ratio"] = _clamp(
                "rr_ratio",
                params.get("rr_ratio", 1.5) + 0.05,
            )
    if stop_hits:
        stop_rate = sum(stop_hits) / len(stop_hits)
        if stop_rate > 0.5:
            params["atr_stop_multiplier"] = _clamp(
                "atr_stop_multiplier",
                params.get("atr_stop_multiplier", 1.5) - 0.05,
            )

    # Update last verified counts so we don't re-adjust on same data
    params["last_verified_counts"] = last_counts
    for tf, count in new_params.items():
        params["last_verified_counts"][tf] = count

    _save_evolution_params(params)
    return params


def get_evolution_stats() -> dict:
    """Get evolution statistics grouped by timeframe, with evolution parameters."""
    conn = get_connection()
    timeframes = {}

    for tf in ["30m", "1h", "4h"]:
        row = conn.execute(
            "SELECT COUNT(*) as total, "
            "COUNT(CASE WHEN verified = 1 AND unverifiable != 1 THEN 1 END) as verified, "
            "COUNT(CASE WHEN verified = 0 THEN 1 END) as pending, "
            "COUNT(CASE WHEN unverifiable = 1 THEN 1 END) as unverifiable, "
            "COUNT(CASE WHEN regime_correct = 1 THEN 1 END) as regime_correct, "
            "COUNT(CASE WHEN direction_correct = 1 THEN 1 END) as dir_correct, "
            "COUNT(CASE WHEN move_sufficient = 1 THEN 1 END) as move_sufficient, "
            "COUNT(CASE WHEN structure_aligned = 1 THEN 1 END) as structure_aligned, "
            "COUNT(CASE WHEN regime_correct_loose = 1 THEN 1 END) as regime_correct_loose, "
            "COUNT(CASE WHEN longer_term_regime_valid = 1 THEN 1 END) as longer_term_valid "
            "FROM signals WHERE timeframe = ?",
            (tf,),
        ).fetchone()

        total = row["total"]
        verified = row["verified"]
        pending = row["pending"]
        unverifiable = row["unverifiable"] or 0
        regime_correct = row["regime_correct"] or 0
        dir_correct = row["dir_correct"] or 0
        move_sufficient_count = row["move_sufficient"] or 0
        structure_aligned_count = row["structure_aligned"] or 0
        regime_correct_loose_count = row["regime_correct_loose"] or 0
        longer_term_valid_count = row["longer_term_valid"] or 0

        regimes = conn.execute(
            "SELECT regime, COUNT(*) as cnt FROM signals "
            "WHERE timeframe = ? GROUP BY regime",
            (tf,),
        ).fetchall()
        regime_counts = {r["regime"]: r["cnt"] for r in regimes}

        # Accuracy by ADX range
        adx_ranges = conn.execute(
            "SELECT "
            "SUM(CASE WHEN adx < 20 AND regime_correct = 1 THEN 1 ELSE 0 END) as low_correct, "
            "SUM(CASE WHEN adx < 20 THEN 1 ELSE 0 END) as low_total, "
            "SUM(CASE WHEN adx >= 20 AND adx < 25 AND regime_correct = 1 THEN 1 ELSE 0 END) as mid_correct, "
            "SUM(CASE WHEN adx >= 20 AND adx < 25 THEN 1 ELSE 0 END) as mid_total, "
            "SUM(CASE WHEN adx >= 25 AND adx < 35 AND regime_correct = 1 THEN 1 ELSE 0 END) as high_correct, "
            "SUM(CASE WHEN adx >= 25 AND adx < 35 THEN 1 ELSE 0 END) as high_total, "
            "SUM(CASE WHEN adx >= 35 AND regime_correct = 1 THEN 1 ELSE 0 END) as vhigh_correct, "
            "SUM(CASE WHEN adx >= 35 THEN 1 ELSE 0 END) as vhigh_total "
            "FROM signals WHERE timeframe = ? AND verified = 1 AND unverifiable != 1",
            (tf,),
        ).fetchone()

        adx_accuracy = {}
        for key, correct_col, total_col in [
            ("low", "low_correct", "low_total"),
            ("mid", "mid_correct", "mid_total"),
            ("high", "high_correct", "high_total"),
            ("vhigh", "vhigh_correct", "vhigh_total"),
        ]:
            c = adx_ranges[correct_col] or 0
            t = adx_ranges[total_col] or 0
            adx_accuracy[key] = {
                "correct": c,
                "total": t,
                "accuracy": round(c / max(t, 1) * 100, 1) if t > 0 else None,
            }

        recent_rows = conn.execute(
            "SELECT regime, direction, adx, confidence, price_at_signal, "
            "price_at_verify, actual_trending, actual_direction, "
            "regime_correct, direction_correct, created_at, verified, "
            "max_favorable_excursion, max_adverse_excursion, target_hit, stop_hit, move_pct, "
            "signal_type, move_sufficient, structure_aligned, regime_correct_loose, "
            "longer_term_regime_valid "
            "FROM signals WHERE timeframe = ? ORDER BY created_at DESC LIMIT 10",
            (tf,),
        ).fetchall()
        recent = []
        for r in recent_rows:
            rd = dict(r)
            recent.append({
                "regime": rd["regime"],
                "direction": rd["direction"],
                "adx": rd["adx"],
                "confidence": rd["confidence"],
                "price_at_signal": rd["price_at_signal"],
                "price_at_verify": rd["price_at_verify"],
                "actual_trending": rd["actual_trending"],
                "actual_direction": rd["actual_direction"],
                "regime_correct": rd["regime_correct"],
                "direction_correct": rd["direction_correct"],
                "move_sufficient": rd.get("move_sufficient"),
                "structure_aligned": rd.get("structure_aligned"),
                "regime_correct_loose": rd.get("regime_correct_loose"),
                "longer_term_regime_valid": rd.get("longer_term_regime_valid"),
                "created_at": rd["created_at"],
                "verified": rd["verified"],
                "max_favorable_excursion": rd["max_favorable_excursion"],
                "max_adverse_excursion": rd["max_adverse_excursion"],
                "target_hit": rd["target_hit"],
                "stop_hit": rd["stop_hit"],
                "move_pct": rd["move_pct"],
                "signal_type": rd.get("signal_type", "trend_following"),
            })

        # Aggregate MFE/MAE and target/stop stats
        mfe_mae_stats = conn.execute(
            "SELECT "
            "AVG(max_favorable_excursion) as avg_mfe, "
            "AVG(max_adverse_excursion) as avg_mae, "
            "SUM(target_hit) as total_target_hits, "
            "SUM(stop_hit) as total_stop_hits, "
            "COUNT(CASE WHEN target_hit IS NOT NULL THEN 1 END) as target_checked, "
            "COUNT(CASE WHEN stop_hit IS NOT NULL THEN 1 END) as stop_checked "
            "FROM signals WHERE timeframe = ? AND verified = 1 AND unverifiable != 1",
            (tf,),
        ).fetchone()

        avg_mfe = mfe_mae_stats["avg_mfe"] or 0
        avg_mae = mfe_mae_stats["avg_mae"] or 0
        target_hit_rate = round(
            (mfe_mae_stats["total_target_hits"] or 0) / max(mfe_mae_stats["target_checked"] or 1, 1) * 100, 1
        )
        stop_hit_rate = round(
            (mfe_mae_stats["total_stop_hits"] or 0) / max(mfe_mae_stats["stop_checked"] or 1, 1) * 100, 1
        )

        dir_total = verified
        timeframes[tf] = {
            "total": total,
            "verified": verified,
            "pending": pending,
            "unverifiable": unverifiable,
            "regime_correct": regime_correct,
            "regime_accuracy": round(regime_correct / max(verified, 1) * 100, 1),
            "dir_total": dir_total,
            "dir_correct": dir_correct,
            "dir_accuracy": round(dir_correct / max(dir_total, 1) * 100, 1),
            "regime_correct_loose": regime_correct_loose_count,
            "regime_accuracy_loose": round(regime_correct_loose_count / max(verified, 1) * 100, 1),
            "move_sufficient": move_sufficient_count,
            "move_sufficient_pct": round(move_sufficient_count / max(verified, 1) * 100, 1),
            "structure_aligned": structure_aligned_count,
            "structure_aligned_pct": round(structure_aligned_count / max(verified, 1) * 100, 1),
            "longer_term_valid": longer_term_valid_count,
            "longer_term_valid_pct": round(longer_term_valid_count / max(verified, 1) * 100, 1),
            "regime_counts": regime_counts,
            "adx_accuracy": adx_accuracy,
            "mfe_mae_stats": {
                "avg_mfe": round(avg_mfe, 2),
                "avg_mae": round(avg_mae, 2),
                "mfe_mae_ratio": round(avg_mfe / max(avg_mae, 0.1), 2),
                "target_hit_rate": target_hit_rate,
                "stop_hit_rate": stop_hit_rate,
            },
            "recent": recent,
        }

    total_row = conn.execute(
        "SELECT COUNT(*) as total, COUNT(CASE WHEN verified = 1 THEN 1 END) as verified "
        "FROM signals"
    ).fetchone()

    # ─── Stage 1 Signal Statistics ──────────────────────────────────────
    # Note: older Stage 1 signals have signal_type='trend_following' but
    # action LIKE 'Stage1%', so we match on action text for backward compat.
    stage1_stats = {}
    try:
        s1_rows = conn.execute(
            "SELECT signal_type, timeframe, "
            "COUNT(*) as total, "
            "SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as dir_correct, "
            "SUM(CASE WHEN regime_correct = 1 THEN 1 ELSE 0 END) as regime_correct, "
            "AVG(max_favorable_excursion) as avg_mfe, "
            "AVG(max_adverse_excursion) as avg_mae, "
            "SUM(CASE WHEN stage2_upgraded = 1 THEN 1 ELSE 0 END) as stage2_count, "
            "SUM(target_hit) as tp_hits, "
            "SUM(stop_hit) as sl_hits "
            "FROM signals WHERE (signal_type LIKE 'stage1_%' OR action LIKE 'Stage1%') AND verified = 1 "
            "GROUP BY signal_type, timeframe"
        ).fetchall()

        for r in s1_rows:
            rd = dict(r)
            key = f"{rd['signal_type']}_{rd['timeframe']}"
            total = rd["total"]
            stage1_stats[key] = {
                "type": rd["signal_type"],
                "timeframe": rd["timeframe"],
                "samples": total,
                "direction_correct_pct": round(rd["dir_correct"] / max(total, 1) * 100, 1),
                "regime_correct_pct": round(rd["regime_correct"] / max(total, 1) * 100, 1),
                "avg_mfe": round(rd["avg_mfe"] or 0, 2),
                "avg_mae": round(rd["avg_mae"] or 0, 2),
                "stage2_upgrade_pct": round(rd["stage2_count"] / max(total, 1) * 100, 1),
                "tp_hit_pct": round((rd["tp_hits"] or 0) / max(total, 1) * 100, 1),
                "sl_hit_pct": round((rd["sl_hits"] or 0) / max(total, 1) * 100, 1),
            }
    except Exception:
        pass  # stage1_stats may be empty until enough signals accumulate

    conn.close()

    # Compute autonomous adjustments
    evolution_params = _compute_evolution_adjustments({
        "timeframes": timeframes,
        "total_predictions": total_row["total"],
        "total_verified": total_row["verified"],
    })

    return {
        "timeframes": timeframes,
        "total_predictions": total_row["total"],
        "total_verified": total_row["verified"],
        "evolution_params": evolution_params,
        "stage1_stats": stage1_stats,
    }


def get_active_thresholds() -> dict:
    """Get current active thresholds for signal generation.

    Returns a dict with global params + per-TF overrides.
    signal_engine.py reads tf_thresholds[tf] for each timeframe.
    """
    return _load_evolution_params()

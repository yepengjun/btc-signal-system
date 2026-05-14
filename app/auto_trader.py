"""Auto-trader: callback hooks to mirror simulated positions to Hyperliquid.

When AUTO_TRADE_ENABLED is true, every simulated position action (open/close/reduce/add)
triggers a corresponding real order on Hyperliquid.

Architecture:
  auto_trader.register_callbacks() → pushes hooks into simulated_position_manager
  simulated_position_manager fires hooks after each DB write.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.hyperliquid import place_order as hl_place_order, close_position as hl_close_position

_logger = logging.getLogger(__name__)


def _hl_mirror_open(conn, position: dict, verdict: dict, price: float):
    """Mirror a simulated open to Hyperliquid."""
    if not settings.auto_trade_enabled:
        return

    side = position.get("side", "")
    btc_size = position.get("position_size") or 0
    leverage = int(position.get("leverage", 20))

    if not btc_size or btc_size <= 0:
        _logger.warning("HL mirror open skipped: invalid position_size=%s", btc_size)
        return

    hl_side = "long" if side == "long" else "short"
    _logger.info("HL mirror OPEN: side=%s size=%s leverage=%s", hl_side, btc_size, leverage)
    result = hl_place_order(hl_side, btc_size, leverage)

    if result.get("ok"):
        conn.execute(
            "UPDATE positions SET hl_enabled=1, hl_sz=?, hl_entry_oid=? WHERE id=?",
            (round(btc_size, 6), str(result.get("oid")), position["id"]),
        )
        conn.commit()
    else:
        _logger.error("HL mirror open FAILED: %s", result.get("error"))


def _hl_mirror_close(conn, position: dict, close_price: float, reason: str):
    """Mirror a simulated close to Hyperliquid."""
    if not settings.auto_trade_enabled:
        return

    hl_sz = position.get("hl_sz")
    position_size = position.get("position_size") or hl_sz
    btc_size = position_size or 0

    if not btc_size or btc_size <= 0:
        _logger.warning("HL mirror close skipped: no size info")
        return

    _logger.info("HL mirror CLOSE: size=%s reason=%s", btc_size, reason)
    result = hl_close_position(round(btc_size, 6))

    if result.get("ok"):
        conn.execute(
            "UPDATE positions SET hl_close_oid=? WHERE id=?",
            (str(result.get("oid")), position["id"]),
        )
        conn.commit()
    else:
        _logger.error("HL mirror close FAILED: %s", result.get("error"))


def _hl_mirror_reduce(conn, position: dict, reduce_size: float):
    """Mirror a reduce action: close partial size on Hyperliquid."""
    if not settings.auto_trade_enabled:
        return

    if not reduce_size or reduce_size <= 0:
        _logger.warning("HL mirror reduce skipped: invalid reduce_size=%s", reduce_size)
        return

    _logger.info("HL mirror REDUCE: size=%s", reduce_size)
    result = hl_close_position(round(reduce_size, 6))

    if not result.get("ok"):
        _logger.error("HL mirror reduce FAILED: %s", result.get("error"))


def _hl_mirror_add(conn, position: dict, add_size: float):
    """Mirror an add (pyramid) action: open additional size on Hyperliquid."""
    if not settings.auto_trade_enabled:
        return

    if not add_size or add_size <= 0:
        _logger.warning("HL mirror add skipped: invalid add_size=%s", add_size)
        return

    side = position.get("side", "")
    leverage = int(position.get("leverage", 20))
    hl_side = "long" if side == "long" else "short"

    _logger.info("HL mirror ADD: side=%s size=%s leverage=%s", hl_side, add_size, leverage)
    result = hl_place_order(hl_side, round(add_size, 6), leverage)

    if not result.get("ok"):
        _logger.error("HL mirror add FAILED: %s", result.get("error"))


def register_callbacks():
    """Register Hyperliquid mirror hooks into the simulated position manager."""
    from app.simulated_position_manager import _register_auto_hooks

    _register_auto_hooks(
        open_hooks=[_hl_mirror_open],
        close_hooks=[_hl_mirror_close],
        reduce_hooks=[_hl_mirror_reduce],
        add_hooks=[_hl_mirror_add],
    )

    _logger.info("Auto-trader callbacks registered (enabled=%s)", settings.auto_trade_enabled)

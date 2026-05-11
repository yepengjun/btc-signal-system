"""Hyperliquid exchange wrapper for BTC perpetual trading."""

from __future__ import annotations

from typing import Optional

import logging

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from app.config import settings

_logger = logging.getLogger(__name__)

_exchange: Optional[Exchange] = None


def get_exchange() -> Optional[Exchange]:
    """Lazy-init Hyperliquid exchange singleton."""
    global _exchange
    if _exchange is None:
        if not settings.hyperliquid_private_key:
            raise RuntimeError("HYPERLIQUID_PRIVATE_KEY not configured")
        pk = settings.hyperliquid_private_key
        if not pk.startswith("0x"):
            pk = "0x" + pk
        wallet = eth_account.Account.from_key(pk)
        base_url = (
            constants.TESTNET_API_URL
            if settings.hyperliquid_testnet
            else constants.MAINNET_API_URL
        )
        _exchange = Exchange(wallet, base_url)
        _logger.info(
            "Hyperliquid exchange initialized (%s, wallet: %s)",
            "testnet" if settings.hyperliquid_testnet else "mainnet",
            wallet.address[:10] + "...",
        )
    return _exchange


def place_order(side: str, btc_size: float, leverage: int) -> dict:
    """Place a market order on Hyperliquid.

    Args:
        side: "long" or "short"
        btc_size: BTC quantity (e.g. 0.01)
        leverage: leverage multiplier

    Returns:
        {"ok": True, "oid": ...} or {"ok": False, "error": "..."}
    """
    try:
        ex = get_exchange()

        # Set leverage (cross margin)
        ex.update_leverage(leverage, "BTC", is_cross=True)

        is_buy = side == "long"
        result = ex.market_open(name="BTC", is_buy=is_buy, sz=btc_size, px=None, slippage=0.01)

        if result.get("status") == "ok":
            # Extract order ID from response
            oid = None
            for status in result.get("response", {}).get("data", {}).get("statuses", []):
                if "filled" in status:
                    oid = status["filled"].get("oid")
                    break
                if "resting" in status:
                    oid = status["resting"].get("oid")
                    break
            return {"ok": True, "oid": oid}
        else:
            return {"ok": False, "error": str(result)}
    except Exception as e:
        _logger.exception("Hyperliquid place_order failed")
        return {"ok": False, "error": str(e)}


def close_position(btc_size: Optional[float] = None) -> dict:
    """Close entire BTC position on Hyperliquid.

    Args:
        btc_size: if None, closes full position; otherwise closes given BTC amount

    Returns:
        {"ok": True, "oid": ...} or {"ok": False, "error": "..."}
    """
    try:
        ex = get_exchange()
        result = ex.market_close(coin="BTC", sz=btc_size, slippage=0.01)

        if result and result.get("status") == "ok":
            oid = None
            for status in result.get("response", {}).get("data", {}).get("statuses", []):
                if "filled" in status:
                    oid = status["filled"].get("oid")
                    break
            return {"ok": True, "oid": oid}
        else:
            return {"ok": False, "error": str(result)}
    except Exception as e:
        _logger.exception("Hyperliquid close_position failed")
        return {"ok": False, "error": str(e)}

"""Hyperliquid real-time read queries for the live trading dashboard tab.

Provides portfolio overview, positions, orders, fills, funding, fees,
and net value curve — all read-only via the Info API.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.config import settings

_logger = logging.getLogger(__name__)


class HyperliquidViewer:
    """Read-only viewer for Hyperliquid portfolio data."""

    def __init__(self):
        self._info = None
        self._init_client()

    def _init_client(self):
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        import eth_account

        is_testnet = settings.hyperliquid_testnet
        if is_testnet:
            self._info = Info(base_url=constants.TESTNET_API_URL, skip_ws=True, timeout=10)
        else:
            self._info = Info(base_url=constants.MAINNET_API_URL, skip_ws=True)

        self._network = "testnet" if is_testnet else "mainnet"
        _logger.info("HyperliquidViewer initialized (%s, account: %s)",
                     self._network, (settings.hyperliquid_account_address or "NOT_CONFIGED")[:10] + "...")

    @property
    def _account_address(self) -> str:
        """Use configured account address (main wallet for Agent Key mode),
        or fall back to deriving from private key."""
        if settings.hyperliquid_account_address:
            return settings.hyperliquid_account_address
        if settings.hyperliquid_private_key:
            import eth_account
            pk = settings.hyperliquid_private_key
            if not pk.startswith("0x"):
                pk = "0x" + pk
            return eth_account.Account.from_key(pk).address
        return ""

    def get_portfolio(self) -> dict:
        """Get portfolio summary: account value, collateral, free collateral, margin used."""
        try:
            state = self._info.user_state(self._account_address)
            ms = state.get("marginSummary", {})
            account_value = float(ms.get("accountValue", 0))
            collateral = float(state.get("withdrawable", 0))
            total_margin = float(ms.get("totalMarginUsed", 0))
            return {
                "account_value": account_value,
                "collateral": collateral,
                "free_collateral": account_value - total_margin if account_value > 0 else 0,
                "margin_used": total_margin,
                "leverage_ratio": account_value / (account_value - collateral) if account_value > collateral else 0,
            }
        except Exception as e:
            _logger.exception("HL get_portfolio failed")
            return {"account_value": 0, "collateral": 0, "free_collateral": 0,
                    "margin_used": 0, "leverage_ratio": 0, "error": str(e)}

    def get_positions(self) -> list[dict]:
        """Get current perpetual positions with PnL, leverage, entry/liquidation prices."""
        try:
            state = self._info.user_state(self._account_address)
            positions = []
            for p in state.get("assetPositions", []):
                pos = p.get("position", {})
                if not pos:
                    continue
                szi = float(pos.get("szi", 0))
                if szi == 0:
                    continue
                entry_px = float(pos.get("entryPx", 0))
                mark_px = float(pos.get("markPx", 0))
                lev = pos.get("leverage", {})
                lev_value = float(lev.get("value", 0)) if isinstance(lev, dict) else float(lev)
                positions.append({
                    "coin": pos.get("coin", ""),
                    "side": "long" if szi > 0 else "short",
                    "size": abs(szi),
                    "entry_price": entry_px,
                    "mark_price": mark_px,
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "roe_pct": (float(pos.get("unrealizedPnl", 0)) / (abs(szi) * entry_px)) * lev_value * 100 if entry_px > 0 else 0,
                    "leverage": lev_value,
                    "liquidation_price": float(pos.get("liquidationPx", 0)),
                    "notional": abs(szi) * mark_px,
                })
            return positions
        except Exception as e:
            _logger.exception("HL get_positions failed")
            return [{"error": str(e)}]

    def get_open_orders(self) -> list[dict]:
        """Get all open orders (limit, stop, trigger)."""
        try:
            orders = self._info.open_orders(self._account_address)
            result = []
            for o in orders:
                if o.get("coin") != "BTC":
                    continue
                result.append({
                    "oid": str(o.get("oid", "")),
                    "coin": o.get("coin", ""),
                    "side": "B" if o.get("side") == "A" else "S",
                    "size": float(o.get("sz", 0)),
                    "price": float(o.get("limitPx", 0)),
                    "type": o.get("orderType", ""),
                    "timestamp": o.get("timestamp", 0),
                    "reduce_only": o.get("reduceOnly", False),
                })
            return result
        except Exception as e:
            _logger.exception("HL get_open_orders failed")
            return [{"error": str(e)}]

    def get_fills(self, page: int = 1, size: int = 50, coin: str = "BTC") -> dict:
        """Get trade history (fills) with pagination. Returns {fills, total_page}."""
        try:
            all_fills = self._info.user_fills(self._account_address)
            # Filter by coin client-side since API doesn't support coin param
            if coin:
                all_fills = [f for f in all_fills if f.get("coin") == coin]
            total = len(all_fills)
            total_pages = max(1, (total + size - 1) // size)
            start = (page - 1) * size
            end = start + size
            page_fills = all_fills[start:end]

            result = []
            for f in page_fills:
                result.append({
                    "coin": f.get("coin", ""),
                    "side": "long" if f.get("side", "") == "B" else "short",
                    "size": float(f.get("sz", 0)),
                    "price": float(f.get("px", 0)),
                    "fee": float(f.get("fee", 0)),
                    "timestamp": f.get("time", 0),
                    "closed_pnl": float(f.get("closedPnl", 0)),
                    "dir": f.get("dir", ""),
                    "oid": str(f.get("oid", "")),
                    "start_position": float(f.get("startPosition", 0)),
                    "position": float(f.get("position", 0)),
                })
            return {"fills": result, "total": total, "page": page, "total_pages": total_pages}
        except Exception as e:
            _logger.exception("HL get_fills failed")
            return {"fills": [], "total": 0, "page": page, "total_pages": 1, "error": str(e)}

    def get_funding_history(self, hours: int = 24) -> list[dict]:
        """Get recent funding rate history for BTC perp."""
        try:
            funding_rates = self._info.funding_history(
                name="BTC",
                start_time=int(time.time() * 1000) - hours * 3600 * 1000,
            )
            result = []
            for f in funding_rates:
                result.append({
                    "timestamp": f.get("time", 0),
                    "rate": float(f.get("fundingRate", 0)),
                    "premium": float(f.get("premium", 0)),
                })
            return result
        except Exception as e:
            _logger.exception("HL get_funding_history failed")
            return [{"error": str(e)}]

    def get_fee_stats(self) -> dict:
        """Get fee statistics: total fees paid, fee breakdown."""
        try:
            all_fills = self._info.user_fills(self._account_address)
            fills = [f for f in all_fills if f.get("coin") == "BTC"]
            total_fees = sum(abs(float(f.get("fee", 0))) for f in fills)
            total_volume = sum(float(f.get("sz", 0)) * float(f.get("px", 0)) for f in fills)
            return {
                "total_fees": total_fees,
                "total_volume": total_volume,
                "fill_count": len(fills),
                "avg_fee_per_fill": total_fees / len(fills) if fills else 0,
                "avg_fee_bps": (total_fees / total_volume * 10000) if total_volume > 0 else 0,
            }
        except Exception as e:
            _logger.exception("HL get_fee_stats failed")
            return {"total_fees": 0, "total_volume": 0, "fill_count": 0,
                    "avg_fee_per_fill": 0, "avg_fee_bps": 0, "error": str(e)}

    def get_net_value_curve(self, period: str = "7d") -> list[dict]:
        """Get perp-only net value over time for chart.

        Hyperliquid doesn't expose direct portfolio history, so we return
        the current account value. A real curve requires local persistence.
        """
        try:
            portfolio = self.get_portfolio()
            now = int(time.time() * 1000)
            return [{"timestamp": now, "net_value": portfolio["account_value"]}]
        except Exception as e:
            _logger.exception("HL get_net_value_curve failed")
            return [{"timestamp": int(time.time() * 1000), "net_value": 0, "error": str(e)}]


# Singleton
_viewer: Optional[HyperliquidViewer] = None

def get_viewer() -> HyperliquidViewer:
    global _viewer
    if _viewer is None:
        _viewer = HyperliquidViewer()
    return _viewer

"""Trade execution abstraction layer.

Architecture:
  TradeExecutor (ABC) → HyperliquidExecutor / SimulatedExecutor
  TradeRouter → routes to correct executor based on TRADE_BACKEND config

Every operation (open/add/reduce/close/set_tp/set_sl) returns a dict with:
  {"ok": True, "order_id": "...", "price": ..., "size": ...}
  or {"ok": False, "error": "..."}
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.config import settings

_logger = logging.getLogger(__name__)


class TradeExecutor(ABC):
    """Abstract trade execution interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Executor name: 'hyperliquid', 'simulated'"""

    @abstractmethod
    def open(self, side: str, size: float, leverage: int,
             tp: Optional[float], sl: Optional[float]) -> dict:
        """Open a position. Returns order_id + fill info."""

    @abstractmethod
    def close(self, size: float = 0) -> dict:
        """Close entire position. Returns order_id."""

    @abstractmethod
    def reduce(self, size: float) -> dict:
        """Partial close. Returns order_id."""

    @abstractmethod
    def add(self, side: str, size: float, leverage: int) -> dict:
        """Add to existing position. Returns order_id."""

    @abstractmethod
    def set_take_profit(self, oid: str, price: float, size: float, side: str) -> dict:
        """Set or update take-profit limit order. Returns order_id."""

    @abstractmethod
    def set_stop_loss(self, oid: str, price: float, size: float, side: str) -> dict:
        """Set or update stop-loss limit order. Returns order_id."""

    @abstractmethod
    def cancel_order(self, oid: str) -> dict:
        """Cancel an open order by oid."""

    @abstractmethod
    def get_open_orders(self) -> list:
        """List all open orders."""

    @abstractmethod
    def get_position_state(self) -> tuple[dict | None, str | None]:
        """Get actual position state from exchange (for sync).

        Returns:
            (state, None) on success — state is {} if no position
            (None, error_msg) on failure
        """

    @abstractmethod
    def get_account_value(self) -> Optional[float]:
        """Get account value (total equity) from exchange, or None if not available."""

    @abstractmethod
    def get_realized_pnl(self) -> float:
        """Get cumulative realized PnL from closed positions. Returns 0 if not available."""


class HyperliquidExecutor(TradeExecutor):
    """Hyperliquid mainnet executor. Uses Info for reads, Exchange for writes."""

    def __init__(self):
        self._exchange = None
        self._info = None
        self._init_client()

    def _init_client(self):
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        import eth_account

        pk = settings.hyperliquid_private_key
        if not pk:
            raise RuntimeError("HYPERLIQUID_PRIVATE_KEY not configured")
        if not pk.startswith("0x"):
            pk = "0x" + pk

        is_testnet = settings.hyperliquid_testnet
        wallet = eth_account.Account.from_key(pk)
        account_address = settings.hyperliquid_account_address or wallet.address

        # When account_address differs from wallet address, it means
        # we're using an Agent Key (sub-account authorized by main wallet).
        # Exchange must receive account_address for order signing.
        kwargs = {}
        if account_address.lower() != wallet.address.lower():
            kwargs["account_address"] = account_address

        if is_testnet:
            base_url = constants.TESTNET_API_URL
            self._exchange = Exchange(wallet, base_url=base_url, timeout=10, **kwargs)
            self._info = Info(base_url=base_url, skip_ws=True, timeout=10)
        else:
            base_url = constants.MAINNET_API_URL
            self._exchange = Exchange(wallet, base_url=base_url, **kwargs)
            self._info = Info(base_url=base_url, skip_ws=True)

        self._wallet_address = wallet.address
        network = "testnet" if is_testnet else "mainnet"
        _logger.info("HyperliquidExecutor initialized (%s, agent=%s, account=%s)",
                     network, wallet.address[:10] + "...", account_address[:10] + "...")

    @property
    def name(self) -> str:
        return "hyperliquid"

    @property
    def account_address(self) -> str:
        """The account address used for queries (main wallet or Agent Key wallet)."""
        return settings.hyperliquid_account_address or self._wallet_address

    def open(self, side: str, size: float, leverage: int,
             tp: Optional[float] = None, sl: Optional[float] = None) -> dict:
        try:
            ex = self._exchange
            ex.update_leverage(leverage, "BTC", is_cross=True)
            is_buy = side == "long"

            # Step 1: Market entry
            # BTC szDecimals=5 on testnet, so round to 5 decimal places
            result = ex.market_open(name="BTC", is_buy=is_buy, sz=round(size, 5), px=None, slippage=0.01)

            if result.get("status") != "ok":
                return {"ok": False, "error": str(result)}

            entry_oid = self._extract_oid(result)
            if not entry_oid:
                _logger.warning("HL open: market_open succeeded but order_id not extracted — treating as failure")
                return {"ok": False, "error": "market_open succeeded but order_id not extracted"}
            _logger.info("HL OPEN (market): side=%s size=%s leverage=%s entry_oid=%s",
                         side, size, leverage, entry_oid)

            # Step 2: Set stop-loss
            sl_oid = None
            if sl:
                sl_result = self.set_stop_loss("", sl, round(size, 5), side)
                if sl_result["ok"]:
                    sl_oid = sl_result["order_id"]
                    _logger.info("HL SL set: oid=%s price=%s", sl_oid, sl)
                else:
                    _logger.warning("HL SL failed (will be retried by background sync): %s",
                                    sl_result.get("error"))

            # Step 3: Set take-profit
            tp_oid = None
            if tp:
                tp_result = self.set_take_profit("", tp, round(size, 5), side)
                if tp_result["ok"]:
                    tp_oid = tp_result["order_id"]
                    _logger.info("HL TP set: oid=%s price=%s", tp_oid, tp)
                else:
                    _logger.warning("HL TP failed (will be retried by background sync): %s",
                                    tp_result.get("error"))

            return {"ok": True, "order_id": entry_oid, "side": side, "size": size,
                    "leverage": leverage, "tp_oid": tp_oid, "sl_oid": sl_oid}
        except Exception as e:
            _logger.exception("HL open failed")
            return {"ok": False, "error": str(e)}

    def close(self, size: float = 0) -> dict:
        """Close position. size=0 means full close."""
        try:
            ex = self._exchange
            pos, err = self.get_position_state()
            sz = size if size and size > 0 else (pos or {}).get("size")
            if not sz:
                return {"ok": False, "error": "No position to close"}
            is_buy = (pos or {}).get("side") == "short"
            price = ex._slippage_price("BTC", is_buy, 0.01, None)
            result = ex.market_close(coin="BTC", sz=sz, px=price, slippage=0.01)
            if result and result.get("status") == "ok":
                oid = self._extract_oid(result)
                if not oid:
                    _logger.warning("HL close: market_close succeeded but order_id not extracted — treating as failure")
                    return {"ok": False, "error": "close succeeded but order_id not extracted"}
                fill_px = self._extract_fill_price(result, price)
                _logger.info("HL CLOSE: size=%s oid=%s fill_price=%s", sz, oid, fill_px)
                return {"ok": True, "order_id": oid, "size": sz, "fill_price": fill_px}
            else:
                return {"ok": False, "error": str(result)}
        except Exception as e:
            _logger.exception("HL close failed")
            return {"ok": False, "error": str(e)}

    def reduce(self, size: float) -> dict:
        """Reduce = partial close with explicit size."""
        try:
            ex = self._exchange
            pos, err = self.get_position_state()
            if not pos or not pos.get("size"):
                return {"ok": False, "error": "No position to reduce"}
            is_buy = pos.get("side") == "short"
            price = ex._slippage_price("BTC", is_buy, 0.01, None)
            result = ex.order(
                coin="BTC", is_buy=is_buy, sz=size, px=price,
                order_type={"limit": {"tif": "Ioc"}}, reduce_only=True,
            )
            if result.get("status") == "ok":
                oid = self._extract_oid(result)
                fill_px = self._extract_fill_price(result, price)
                _logger.info("HL REDUCE: size=%s oid=%s fill_price=%s", size, oid, fill_px)
                return {"ok": True, "order_id": oid, "size": size, "fill_price": fill_px}
            else:
                return {"ok": False, "error": str(result)}
        except Exception as e:
            _logger.exception("HL reduce failed")
            return {"ok": False, "error": str(e)}

    def add(self, side: str, size: float, leverage: int) -> dict:
        """Add to existing position: market open only, no TP/SL.

        HL is single-position mode: add-on merges into existing position
        and TP/SL auto-adjust by size. No need to set new TP/SL orders.
        """
        try:
            ex = self._exchange
            ex.update_leverage(leverage, "BTC", is_cross=True)
            is_buy = side == "long"
            result = ex.market_open(name="BTC", is_buy=is_buy, sz=size, px=None, slippage=0.01)

            if result.get("status") == "ok":
                oid = self._extract_oid(result)
                _logger.info("HL ADD: side=%s size=%s oid=%s", side, size, oid)
                return {"ok": True, "order_id": oid, "side": side, "size": size, "leverage": leverage}
            else:
                return {"ok": False, "error": str(result)}
        except Exception as e:
            _logger.exception("HL add failed")
            return {"ok": False, "error": str(e)}

    def set_take_profit(self, oid: str, price: float, size: float, side: str) -> dict:
        """Place/update a take-profit limit order."""
        try:
            ex = self._exchange
            is_buy = side == "short"  # take profit = close: long→sell, short→buy
            price = round(price)  # BTC tick size is 1 (whole numbers)
            if oid:
                result = ex.modify_order(
                    oid=int(oid), name="BTC", is_buy=is_buy, sz=size,
                    limit_px=price, order_type={"limit": {"tif": "Gtc"}}, reduce_only=True
                )
                if not result.get("status") == "ok":
                    # modify_order failed — cancel and recreate
                    try:
                        cancel_result = self.cancel_order(str(oid))
                        if not cancel_result.get("ok", False):
                            _logger.warning("HL TP cancel after modify fail (oid=%s): %s",
                                            oid, cancel_result.get("error", "unknown"))
                    except Exception:
                        _logger.warning("HL TP cancel exception after modify fail (oid=%s)", oid)
                    result = ex.order(
                        name="BTC", is_buy=is_buy, sz=size, limit_px=price,
                        order_type={"limit": {"tif": "Gtc"}}, reduce_only=True
                    )
            else:
                result = ex.order(
                    name="BTC", is_buy=is_buy, sz=size, limit_px=price,
                    order_type={"limit": {"tif": "Gtc"}}, reduce_only=True
                )
            if result.get("status") == "ok":
                new_oid = self._extract_oid(result)
                if not new_oid:
                    # Check for nested error in statuses
                    error = self._extract_error(result)
                    if error:
                        _logger.warning("HL TP error: %s", error)
                        return {"ok": False, "error": error}
                    new_oid = oid  # fallback to existing oid (update case)
                _logger.info("HL TP: price=%s size=%s oid=%s", price, size, new_oid)

                # Sweep stale TP orders: clean up any extra reduce-only limit orders
                # for BTC that aren't the current TP oid. Prevents accumulation from
                # failed modify_order attempts where cancel succeeded but the DB oid
                # was already stale.
                try:
                    open_orders = self.get_open_orders()
                    for o in open_orders:
                        o_oid = o.get("oid")
                        if o_oid and str(o_oid) != str(new_oid) and str(o_oid) != str(oid):
                            if o.get("reduceOnly") and o.get("orderType") == "limit":
                                self.cancel_order(str(o_oid))
                                _logger.info("HL cleaned up stale TP limit order: oid=%s", o_oid)
                except Exception:
                    pass  # cleanup is best-effort

                return {"ok": True, "order_id": new_oid, "price": price}
            else:
                return {"ok": False, "error": str(result)}
        except Exception as e:
            _logger.exception("HL set_take_profit failed")
            return {"ok": False, "error": str(e)}

    def set_stop_loss(self, oid: str, price: float, size: float, side: str) -> dict:
        """Place/update a stop-loss trigger order (market execution on trigger)."""
        try:
            ex = self._exchange
            is_buy = side == "short"
            price = round(price)  # BTC tick size is 1 (whole numbers)
            # HL SDK uses "trigger" type, not "stop"
            # tpsl="sl" marks this as a stop-loss (vs "tp" for take-profit)
            order_type = {
                "trigger": {
                    "triggerPx": price,
                    "isMarket": True,
                    "tpsl": "sl",
                }
            }

            if oid:
                # modify_order doesn't support trigger orders directly,
                # cancel old and place new
                cancel_ok = False
                old_oid = oid
                try:
                    cancel_result = self.cancel_order(str(oid))
                    cancel_ok = cancel_result.get("ok", False)
                    if not cancel_ok:
                        _logger.warning("HL SL cancel failed (oid=%s): %s — still placing new order",
                                        oid, cancel_result.get("error", "unknown"))
                except Exception:
                    _logger.warning("HL SL cancel exception (oid=%s) — still placing new order", oid)

                # Clean up stale orders: cancel may fail due to already-filled,
                # already-cancelled, or invalid OID. Sweep open orders to remove
                # any leftover SL/TP trigger orders for BTC.
                try:
                    open_orders = self.get_open_orders()
                    for o in open_orders:
                        o_oid = o.get("oid") or o.get("coin")
                        if o_oid and str(o_oid) != str(oid) and str(o.get("order_id", "")) != str(oid):
                            if o.get("reduceOnly") and o.get("orderType") == "trigger":
                                self.cancel_order(str(o_oid))
                                _logger.info("HL cleaned up stale trigger order: oid=%s", o_oid)
                except Exception:
                    pass  # cleanup is best-effort

                result = ex.order(
                    name="BTC", is_buy=is_buy, sz=size,
                    limit_px=price,  # ignored for market trigger
                    order_type=order_type,
                    reduce_only=True
                )

                # If cancel succeeded but new order failed, the position has
                # NO SL protection. Log urgently so reconcile loop picks it up.
                if not result.get("status") == "ok" and cancel_ok:
                    _logger.error("HL SL: cancel succeeded but new order FAILED (old_oid=%s, new_price=%s) — "
                                  "position has NO stop-loss until reconcile recreates it", old_oid, price)
            else:
                result = ex.order(
                    name="BTC", is_buy=is_buy, sz=size,
                    limit_px=price,  # ignored for market trigger
                    order_type=order_type,
                    reduce_only=True
                )
            if result.get("status") == "ok":
                new_oid = self._extract_oid(result)
                if not new_oid:
                    error = self._extract_error(result)
                    if error:
                        _logger.warning("HL SL error: %s", error)
                        return {"ok": False, "error": error}
                    new_oid = oid  # fallback to existing oid (update case)
                _logger.info("HL SL: trigger=%s size=%s oid=%s", price, size, new_oid)
                return {"ok": True, "order_id": new_oid, "price": price}
            else:
                return {"ok": False, "error": str(result)}
        except Exception as e:
            _logger.exception("HL set_stop_loss failed")
            return {"ok": False, "error": str(e)}

    def cancel_order(self, oid: str) -> dict:
        try:
            ex = self._exchange
            result = ex.cancel(name="BTC", oid=int(oid))
            if result.get("status") == "ok":
                return {"ok": True}
            else:
                return {"ok": False, "error": str(result)}
        except Exception as e:
            _logger.exception("HL cancel_order failed")
            return {"ok": False, "error": str(e)}

    def get_open_orders(self) -> list:
        try:
            return self._info.open_orders(self.account_address)
        except Exception as e:
            _logger.exception("HL get_open_orders failed")
            return []

    def get_account_value(self) -> Optional[float]:
        """Get HL account value (total equity: available + margin used)."""
        try:
            state = self._info.user_state(self.account_address)
            ms = state.get("marginSummary", {})
            return float(ms.get("accountValue", 0))
        except Exception:
            _logger.exception("HL get_account_value failed")
            return None

    def get_realized_pnl(self) -> float:
        """Get cumulative realized PnL from all BTC fills on Hyperliquid."""
        try:
            fills = self._info.user_fills(self.account_address)
            btc_fills = [f for f in fills if f.get("coin") == "BTC"]
            return sum(float(f.get("closedPnl", 0)) for f in btc_fills)
        except Exception:
            _logger.exception("HL get_realized_pnl failed")
            return 0.0

    def get_position_state(self) -> tuple[dict | None, str | None]:
        """Get actual BTC perp position from Hyperliquid."""
        try:
            state = self._info.user_state(self.account_address)
            positions = state.get("assetPositions", [])
            for p in positions:
                pos = p.get("position", {})
                if pos.get("coin") == "BTC":
                    return {
                        "side": "long" if float(pos.get("szi", 0)) > 0 else "short",
                        "size": abs(float(pos.get("szi", 0))),
                        "entry_price": float(pos.get("entryPx", 0)),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                        "leverage": float(pos.get("leverage", {}).get("value", 20)),
                    }, None
            return {}, None  # no position
        except Exception as e:
            _logger.exception("HL get_position_state failed")
            return None, str(e)

    def _extract_oid(self, result: dict) -> Optional[str]:
        import json as _json
        resp = result.get("response")
        # Handle case where response is a JSON string
        if isinstance(resp, str):
            try:
                resp = _json.loads(resp)
            except Exception:
                _logger.warning("_extract_oid: failed to parse response as JSON: %s", resp[:200])
                return None
        data = (resp or {}).get("data", {})
        statuses = data.get("statuses", [])
        for status in statuses:
            if "filled" in status:
                return str(status["filled"].get("oid"))
            if "resting" in status:
                return str(status["resting"].get("oid"))
        _logger.warning("_extract_oid: no oid found in statuses=%s", statuses)
        return None

    def _extract_fill_price(self, result: dict, fallback: float) -> Optional[float]:
        """Extract actual fill price from SDK response."""
        for status in result.get("response", {}).get("data", {}).get("statuses", []):
            if "filled" in status:
                filled = status["filled"]
                px = filled.get("avgPx") or filled.get("limitPx")
                if px:
                    return float(px)
        return fallback

    def _extract_error(self, result: dict) -> Optional[str]:
        """Extract error message from nested statuses.
        HL SDK returns {\"status\": \"ok\", ...statuses: [{\"error\": \"...\"}]}
        for certain failures, so we must check inner statuses, not just outer status."""
        for status in result.get("response", {}).get("data", {}).get("statuses", []):
            if "error" in status:
                return status["error"]
        return None


class SimulatedExecutor(TradeExecutor):
    """Simulated executor: logs operations, generates local order IDs."""

    @property
    def name(self) -> str:
        return "simulated"

    def _gen_id(self) -> str:
        return f"sim-{uuid.uuid4().hex[:8]}"

    def open(self, side: str, size: float, leverage: int,
             tp: Optional[float] = None, sl: Optional[float] = None) -> dict:
        oid = self._gen_id()
        _logger.info("SIM OPEN: side=%s size=%s leverage=%s tp=%s sl=%s oid=%s",
                     side, size, leverage, tp, sl, oid)
        return {"ok": True, "order_id": oid, "side": side, "size": size, "leverage": leverage}

    def close(self, size: float = 0) -> dict:
        oid = self._gen_id()
        _logger.info("SIM CLOSE: size=%s oid=%s", size, oid)
        return {"ok": True, "order_id": oid, "size": size}

    def reduce(self, size: float) -> dict:
        oid = self._gen_id()
        _logger.info("SIM REDUCE: size=%s oid=%s", size, oid)
        return {"ok": True, "order_id": oid, "size": size}

    def add(self, side: str, size: float, leverage: int) -> dict:
        oid = self._gen_id()
        _logger.info("SIM ADD: side=%s size=%s leverage=%s oid=%s", side, size, leverage, oid)
        return {"ok": True, "order_id": oid, "side": side, "size": size, "leverage": leverage}

    def set_take_profit(self, oid: str, price: float, size: float, side: str) -> dict:
        new_oid = oid or self._gen_id()
        _logger.info("SIM TP: price=%s size=%s oid=%s", price, size, new_oid)
        return {"ok": True, "order_id": new_oid, "price": price}

    def set_stop_loss(self, oid: str, price: float, size: float, side: str) -> dict:
        new_oid = oid or self._gen_id()
        _logger.info("SIM SL: price=%s size=%s oid=%s", price, size, new_oid)
        return {"ok": True, "order_id": new_oid, "price": price}

    def cancel_order(self, oid: str) -> dict:
        _logger.info("SIM CANCEL: oid=%s", oid)
        return {"ok": True}

    def get_open_orders(self) -> list:
        return []

    def get_position_state(self) -> tuple[dict | None, str | None]:
        return {}, None

    def get_account_value(self) -> Optional[float]:
        return None

    def get_realized_pnl(self) -> float:
        return 0.0


class TradeRouter:
    """Routes trade operations to the correct executor based on config."""

    def __init__(self):
        backend = getattr(settings, 'trade_backend', 'simulated')
        if backend == "hyperliquid":
            self._executor = HyperliquidExecutor()
        else:
            self._executor = SimulatedExecutor()
        _logger.info("TradeRouter initialized: backend=%s", self._executor.name)

    @property
    def name(self) -> str:
        return self._executor.name

    def open(self, side: str, size: float, leverage: int,
             tp: Optional[float] = None, sl: Optional[float] = None) -> dict:
        return self._executor.open(side, size, leverage, tp, sl)

    def close(self, size: float = 0) -> dict:
        return self._executor.close(size)

    def reduce(self, size: float) -> dict:
        return self._executor.reduce(size)

    def add(self, side: str, size: float, leverage: int) -> dict:
        return self._executor.add(side, size, leverage)

    def set_take_profit(self, oid: str, price: float, size: float, side: str) -> dict:
        return self._executor.set_take_profit(oid, price, size, side)

    def set_stop_loss(self, oid: str, price: float, size: float, side: str) -> dict:
        return self._executor.set_stop_loss(oid, price, size, side)

    def cancel_order(self, oid: str) -> dict:
        return self._executor.cancel_order(oid)

    def get_open_orders(self) -> list:
        return self._executor.get_open_orders()

    def get_position_state(self) -> tuple[dict | None, str | None]:
        return self._executor.get_position_state()

    def get_account_value(self) -> Optional[float]:
        return self._executor.get_account_value()

    def get_realized_pnl(self) -> float:
        return self._executor.get_realized_pnl()


# Singleton
_trade_router: Optional[TradeRouter] = None

def get_router() -> TradeRouter:
    global _trade_router
    if _trade_router is None:
        _trade_router = TradeRouter()
    return _trade_router

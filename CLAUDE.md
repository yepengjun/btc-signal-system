# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTC perpetual contract signal system with multi-timeframe ADX/DI analysis, regime detection, simulated position management, and self-evolution. FastAPI backend serving a Vue 3 SPA dashboard. Connects to Binance Futures for market data and optionally Hyperliquid for real order execution.

## Architecture

```
app/
  main.py              - FastAPI app: routes, auth, signal generation pipeline, WS endpoint
  signal_engine.py     - Core logic: 200-candle analysis, regime detection, ADX/DI, verdict generation
  indicators.py        - Technical indicators: ADX/DI, ATR, BB, EMA, squeeze detection
  binance.py           - Binance Futures data: klines, price, funding rate, open interest (via ccxt)
  simulated_position_manager.py - Auto-managed simulated positions with priority chain exit logic
  evolution.py         - Self-evolution: weight adjustment based on signal accuracy
  database.py          - SQLite schema init + connection factory (inline migrations)
  config.py            - Settings from env vars via python-dotenv (no Pydantic)
  hyperliquid.py       - Real order execution on Hyperliquid (market open/close)
  auth.py              - Session-based auth with cookie tokens (SHA256 password hashing)
  templates/
    dashboard.html     - Vue 3 SPA (inline, ~2170 lines): 3 tabs (signals, position, evolution)
    login.html         - Login page
data/
  signals.db           - SQLite database (auto-created on startup)
  evolution.json       - Weight evolution state
```

## Data Flow

1. **Signal Generation** (`main.py` → `signal_engine.py`):
   - Fetch 200 candles per timeframe (30m, 1h, 4h) from Binance via ccxt
   - Calculate ADX/DI, ATR, Bollinger Bands, EMA, squeeze detection per timeframe
   - Dual ADX (fast=10, slow=21) for decay detection
   - Volatility percentile (ATR over 200-candle history)
   - Regime classification: trending, forming, breakout, exhaustion, high_volatility, low_volatility, ranging
   - Generate verdict with order signal (entry/stop/target/leverage)
   - Apply filters: funding rate, OI divergence, circuit breaker (flash crash, liquidation cascade)

2. **Position Management** (`simulated_position_manager.py`):
   - Strict priority chain: stop loss → take profit → emergency → reversal → ADX trailing → signal exit → reduce → add → hold
   - Pyramid add-on (max 3), reduce (max 2), dynamic leverage by regime
   - Signal cooldown for opposite direction re-entry

3. **Self-Evolution** (`evolution.py`):
   - Weights for timeframe credibility adjust based on prediction accuracy
   - Verified signals update regime_accuracy and direction_accuracy per timeframe

## Common Development Commands

### Run locally (no Docker)
```bash
pip install -r requirements.txt
# Optional: create .env with BINANCE_SYMBOL, APP_USERNAME, APP_PASSWORD, etc.
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker
```bash
docker compose up --build
# Access at http://localhost:8888
```

### Database
- SQLite at `data/signals.db` (auto-created on startup)
- Schema migrations are inline in `database.py` (ALTER TABLE with try/except for idempotency)
- To reset: `rm data/signals.db` and restart

### API Endpoints
- `GET /` - Dashboard (requires auth)
- `POST /login` - Login (JSON: {username, password})
- `POST /logout` - Logout
- `GET /api/signals` - Full signal data + verdict + simulated position
- `GET /api/position` - Open manual position info
- `POST /api/position` - Create manual position (with optional Hyperliquid execution)
- `POST /api/position/{id}/close` - Close position (auto-executes Hyperliquid if enabled)
- `POST /api/position/{id}/action` - Record position action state change (add/reduce/exit)
- `POST /api/position/{id}` - Update position metadata (target, stop, leverage)
- `GET /api/position/simulated` - Open simulated position
- `GET /api/positions/simulated/history` - Simulated position history
- `GET /api/position/simulated/stats` - Simulated trading stats (balance, PnL, trade count)
- `GET /api/evolution` - Evolution stats
- `POST /api/evolution/reset` - Reset evolution weights
- `GET /ws` - WebSocket for real-time updates

## Key Implementation Details

### Signal Engine (`signal_engine.py`, ~1325 lines)
- `generate_verdict(history_rows, position, market_context)` is the main entry point
- `history_rows`: list of past verdict_history rows for trend analysis
- `position`: open manual position dict (if any) for context-aware signals
- `market_context`: dict with funding_rate (raw decimal, e.g. 0.0001), open_interest, open_interest_prev
- Inside the engine, `funding_rate` is converted to percentage via `* 100`
- Returns verdict with: regime, direction, strength, confidence, order_signal, market_context, timeframes
- Circuit breaker can force order_signal to "观望" (wait)
- Funding rate thresholds: >0.1% blocks long, <-0.1% blocks short, >0.05%/-0.05% reduces leverage

### Dashboard (`templates/dashboard.html`, ~2170 lines)
- Vue 3 composition API with `setup()` pattern
- 3 tabs: signals (主判断), position (持仓), evolution (进化)
- Auto-refreshes every 10 seconds via polling
- Displays: verdict, order signal, entry timing, forecast, timeframe analysis, simulated positions
- `market_context` section shows funding rate, OI, leverage, circuit breaker warnings
- TF cards show vol_percentile and adx_decay when available
- `circuit_breaker` is a string (reason) or null, NOT an object with `.active` property

### Binance Data Layer (`binance.py`)
- Uses ccxt for Binance Futures API
- Kline caching with 55s TTL; funding rate cache 5min; OI cache 1min
- `fetch_funding_rate()` returns raw decimal (e.g., 0.0001 = 0.01%) — NOT multiplied by 100
- Signal engine handles the `* 100` conversion internally

### Main App (`main.py`, ~650 lines)
- Signal caching: responses cached for `SIGNAL_INTERVAL` seconds (default 300s) to avoid spamming Binance
- WebSocket sends full signal data every `SIGNAL_INTERVAL` seconds to authenticated clients
- Startup warms cache in background thread
- `_build_signals_response()` is the core signal builder — handles both cached and fresh generation
- `_refresh_live_prices()` updates ticker price and entry timing fields from cached data

### Auth (`auth.py`)
- Cookie-based sessions with `itsdangerous` token signing
- Passwords hashed via SHA256 (simple, not bcrypt — sufficient for single-user app)
- Default credentials: wang / trad2026

### Database Schema (`database.py`)
- **signals**: per-timeframe signal snapshots with verification fields (actual_direction, regime_correct, etc.)
- **positions**: manual + simulated positions with Hyperliquid integration columns (hl_enabled, hl_sz, hl_entry_oid, action_state, entry_adx, max_adx, is_simulated, close_reason, etc.)
- **verdict_history**: aggregated multi-TF verdict snapshots (last 50, deduplicated for display)
- **position_action_state**: audit trail of position action transitions (open → add → reduce → exit)
- **users**: single-user auth (default user auto-created on startup)

### Environment Variables (all optional, with defaults)
- `BINANCE_SYMBOL` - Default: BTC/USDT:USDT
- `DB_PATH` - Default: /app/data/signals.db
- `SESSION_SECRET` - Default: change-me-in-production
- `APP_USERNAME` - Default: wang
- `APP_PASSWORD` - Default: trad2026
- `HOST` / `PORT` - Default: 0.0.0.0:8888
- `SIGNAL_INTERVAL` - Default: 300 (seconds)
- `HYPERLIQUID_ENABLED` - Default: false
- `HYPERLIQUID_TESTNET` - Default: true
- `HYPERLIQUID_PRIVATE_KEY` - Default: empty
- `SIM_INITIAL_BALANCE` - Default: 10000

## Testing Notes

No formal test suite exists. Testing is done by:
1. Starting the server and checking the dashboard
2. Verifying signal accuracy via the evolution tab
3. Manual UI interaction

The system is designed to run continuously, with signals auto-generated on each API call.

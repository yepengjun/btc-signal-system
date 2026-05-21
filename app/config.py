import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    binance_symbol: str = os.getenv("BINANCE_SYMBOL", "BTC/USDT:USDT")
    db_path: str = os.getenv("DB_PATH", "./data/signals.db")
    session_secret: str = os.getenv("SESSION_SECRET", "change-me-in-production")
    username: str = os.getenv("APP_USERNAME", "wang")
    password: str = os.getenv("APP_PASSWORD", "trad2026")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8888"))
    signal_interval_seconds: int = int(os.getenv("SIGNAL_INTERVAL", "5"))

    # Hyperliquid
    hyperliquid_enabled: bool = os.getenv("HYPERLIQUID_ENABLED", "false").lower() == "true"
    hyperliquid_testnet: bool = os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"
    hyperliquid_private_key: str = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    hyperliquid_account_address: str = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS", "")

    # Trade backend (simulated / hyperliquid)
    trade_backend: str = os.getenv("TRADE_BACKEND", "simulated")

    # Funding rate filter thresholds (raw decimal, e.g. 0.001 = 0.1%)
    funding_rate_block_threshold: float = float(os.getenv("FUNDING_RATE_BLOCK_THRESHOLD", "0.001"))
    funding_rate_warn_threshold: float = float(os.getenv("FUNDING_RATE_WARN_THRESHOLD", "0.0005"))

    # Simulated trading
    sim_initial_balance: float = float(os.getenv("SIM_INITIAL_BALANCE", "10000"))

    # Auto trade (mirror sim positions to Hyperliquid)
    auto_trade_enabled: bool = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"


settings = Settings()

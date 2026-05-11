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
    signal_interval_seconds: int = int(os.getenv("SIGNAL_INTERVAL", "60"))

    # Hyperliquid
    hyperliquid_enabled: bool = os.getenv("HYPERLIQUID_ENABLED", "false").lower() == "true"
    hyperliquid_testnet: bool = os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"
    hyperliquid_private_key: str = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")

    # Simulated trading
    sim_initial_balance: float = float(os.getenv("SIM_INITIAL_BALANCE", "10000"))


settings = Settings()

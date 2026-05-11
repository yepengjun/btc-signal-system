import sqlite3
import hashlib
import os

from app.config import settings


def get_connection() -> sqlite3.Connection:
    db_dir = os.path.dirname(settings.db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timeframe TEXT NOT NULL,
            regime TEXT,
            direction TEXT,
            adx REAL,
            plus_di REAL,
            minus_di REAL,
            confidence INTEGER,
            strength TEXT,
            momentum TEXT,
            duration_hours REAL,
            price_at_signal REAL,
            price_at_verify REAL,
            verdict TEXT,
            action TEXT,
            target REAL,
            stop REAL,
            actual_trending INTEGER,
            actual_direction TEXT,
            regime_correct INTEGER,
            direction_correct INTEGER,
            created_at REAL,
            verified INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            side TEXT NOT NULL,
            entry_price REAL,
            target REAL,
            stop REAL,
            leverage REAL DEFAULT 1,
            status TEXT DEFAULT 'open',
            pnl REAL,
            created_at REAL,
            updated_at REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS verdict_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            regime TEXT,
            direction TEXT,
            strength TEXT,
            confidence INTEGER,
            momentum TEXT,
            advice TEXT,
            price REAL,
            adx_4h REAL,
            adx_1h REAL,
            dir_4h TEXT,
            dir_1h TEXT,
            created_at REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS position_action_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            adx_4h REAL,
            adx_1h REAL,
            price REAL,
            position_size REAL,
            created_at REAL
        )
    """)

    # Add position_size column if not exists
    try:
        c.execute("ALTER TABLE position_action_state ADD COLUMN position_size REAL")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Signal enhancement columns
    for col_sql in [
        "ALTER TABLE signals ADD COLUMN max_favorable_excursion REAL",
        "ALTER TABLE signals ADD COLUMN max_adverse_excursion REAL",
        "ALTER TABLE signals ADD COLUMN target_hit INTEGER DEFAULT 0",
        "ALTER TABLE signals ADD COLUMN stop_hit INTEGER DEFAULT 0",
        "ALTER TABLE signals ADD COLUMN move_pct REAL",
        "ALTER TABLE signals ADD COLUMN verify_adx REAL",
        "ALTER TABLE signals ADD COLUMN verify_price REAL",
        "ALTER TABLE signals ADD COLUMN verify_time REAL",
        "ALTER TABLE signals ADD COLUMN unverifiable INTEGER DEFAULT 0",
    ]:
        try:
            c.execute(col_sql)
        except sqlite3.OperationalError:
            pass

    # Hyperliquid migration: add columns if not exists
    for col_sql in [
        "ALTER TABLE positions ADD COLUMN hl_enabled INTEGER DEFAULT 0",
        "ALTER TABLE positions ADD COLUMN hl_sz REAL",
        "ALTER TABLE positions ADD COLUMN hl_entry_oid TEXT",
        "ALTER TABLE positions ADD COLUMN hl_close_oid TEXT",
        "ALTER TABLE positions ADD COLUMN action_state TEXT DEFAULT 'open'",
        "ALTER TABLE positions ADD COLUMN entry_adx REAL",
        "ALTER TABLE positions ADD COLUMN max_adx REAL",
    ]:
        try:
            c.execute(col_sql)
        except sqlite3.OperationalError:
            pass  # column already exists

    # Simulated position columns
    for col_sql in [
        "ALTER TABLE positions ADD COLUMN is_simulated INTEGER DEFAULT 0",
        "ALTER TABLE positions ADD COLUMN close_reason TEXT",
        "ALTER TABLE positions ADD COLUMN closed_at REAL",
        "ALTER TABLE positions ADD COLUMN close_price REAL",
        "ALTER TABLE positions ADD COLUMN entry_reason TEXT",
        "ALTER TABLE positions ADD COLUMN reduce_count INTEGER DEFAULT 0",
        "ALTER TABLE positions ADD COLUMN last_signal_time REAL",
        "ALTER TABLE positions ADD COLUMN add_count INTEGER DEFAULT 0",
        "ALTER TABLE positions ADD COLUMN adx_trailing_stop REAL",
        "ALTER TABLE positions ADD COLUMN position_size REAL",
        "ALTER TABLE positions ADD COLUMN realized_pnl REAL DEFAULT 0",
    ]:
        try:
            c.execute(col_sql)
        except sqlite3.OperationalError:
            pass  # column already exists

    # Create default user
    pw_hash = hashlib.sha256(settings.password.encode()).hexdigest()
    try:
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (settings.username, pw_hash),
        )
    except sqlite3.IntegrityError:
        pass

    conn.commit()
    conn.close()

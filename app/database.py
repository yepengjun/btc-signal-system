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

    # Signal type column (trend_following / exhaustion_reversal)
    try:
        c.execute("ALTER TABLE signals ADD COLUMN signal_type TEXT DEFAULT 'trend_following'")
    except sqlite3.OperationalError:
        pass

    # Stage 1 signals: track whether a Stage 1 warning was later upgraded to Stage 2
    try:
        c.execute("ALTER TABLE signals ADD COLUMN stage2_upgraded INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    # Stage 1 statistics table (aggregated by signal type)
    c.execute("""
        CREATE TABLE IF NOT EXISTS stage1_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage1_type TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            sample_count INTEGER DEFAULT 0,
            direction_correct_pct REAL,
            significant_move_pct REAL,
            tp_hit_pct REAL,
            sl_hit_pct REAL,
            avg_mfe REAL,
            avg_mae REAL,
            stage2_upgrade_pct REAL,
            avg_time_to_stage2 REAL,
            avg_adx_rise REAL,
            last_updated REAL,
            UNIQUE(stage1_type, timeframe)
        )
    """)

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

    # Regime split columns: decompose trend_persisted into sub-metrics
    for col_sql in [
        "ALTER TABLE signals ADD COLUMN move_sufficient INTEGER DEFAULT 0",
        "ALTER TABLE signals ADD COLUMN structure_aligned INTEGER DEFAULT 0",
        "ALTER TABLE signals ADD COLUMN regime_correct_loose INTEGER DEFAULT 0",
        "ALTER TABLE signals ADD COLUMN longer_term_regime_valid INTEGER DEFAULT 0",
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
        "ALTER TABLE positions ADD COLUMN last_decay_tightened REAL",
    ]:
        try:
            c.execute(col_sql)
        except sqlite3.OperationalError:
            pass  # column already exists

    # Signal type tracking on positions — enables signal-type-specific exit logic
    # and accurate evolution tracking per signal type
    try:
        c.execute("ALTER TABLE positions ADD COLUMN signal_type TEXT")
    except sqlite3.OperationalError:
        pass

    # Bug fix: max_price/min_price for accurate add-on price-extreme detection.
    # Without these, _is_price_extreme falls back to entry_price, treating any
    # minor breakout as a "new extreme" even after a significant pullback.
    for col_sql in [
        "ALTER TABLE positions ADD COLUMN max_price REAL",
        "ALTER TABLE positions ADD COLUMN min_price REAL",
    ]:
        try:
            c.execute(col_sql)
        except sqlite3.OperationalError:
            pass

    # Backfill max_price/min_price for existing open positions from entry_price
    c.execute(
        "UPDATE positions SET max_price = entry_price, min_price = entry_price "
        "WHERE max_price IS NULL OR min_price IS NULL"
    )

    # Signal metadata: record signal context at open time
    for col_sql in [
        "ALTER TABLE positions ADD COLUMN signal_regime TEXT",
        "ALTER TABLE positions ADD COLUMN signal_strength TEXT",
        "ALTER TABLE positions ADD COLUMN signal_confidence INTEGER",
        "ALTER TABLE positions ADD COLUMN signal_adx_4h REAL",
        "ALTER TABLE positions ADD COLUMN signal_adx_1h REAL",
        "ALTER TABLE positions ADD COLUMN signal_verdict_dir TEXT",
        "ALTER TABLE positions ADD COLUMN original_stop REAL",
        "ALTER TABLE positions ADD COLUMN trailing_stop REAL",
        "ALTER TABLE positions ADD COLUMN stop_update_reason TEXT",
        "ALTER TABLE positions ADD COLUMN highest_pnl_pct REAL DEFAULT 0",
        "ALTER TABLE positions ADD COLUMN hl_tp_oid TEXT",
        "ALTER TABLE positions ADD COLUMN hl_sl_oid TEXT",
        "ALTER TABLE positions ADD COLUMN hl_fees REAL DEFAULT 0",
        "ALTER TABLE positions ADD COLUMN funding_paid REAL DEFAULT 0",
    ]:
        try:
            c.execute(col_sql)
        except sqlite3.OperationalError:
            pass  # column already exists

    # Order ID in position_action_state for trade tracking
    try:
        c.execute("ALTER TABLE position_action_state ADD COLUMN order_id TEXT")
    except sqlite3.OperationalError:
        pass

    # K-line persistent cache table
    c.execute("""
        CREATE TABLE IF NOT EXISTS klines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            ts REAL NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(symbol, timeframe, ts)
        )
    """)

    # Index for fast time-range queries
    try:
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_klines_tf_ts ON klines(symbol, timeframe, ts)"
        )
    except sqlite3.OperationalError:
        pass

    # Create default user (use salted hash from auth module)
    from app.auth import hash_password
    pw_hash = hash_password(settings.password)
    try:
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (settings.username, pw_hash),
        )
    except sqlite3.IntegrityError:
        pass

    # Migrate existing unsalted password hashes to salted
    # Unsalted SHA256 is 64 hex chars; salted uses a different value since
    # the salt is session-secret-dependent, so any stale hash won't match.
    # Force password reset by replacing known default hash with salted version.
    old_default_hash = hashlib.sha256("trad2026".encode()).hexdigest()
    c.execute(
        "UPDATE users SET password_hash = ? WHERE password_hash = ?",
        (pw_hash, old_default_hash),
    )

    conn.commit()
    conn.close()

import hashlib
import time

from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import settings

_serializer = URLSafeTimedSerializer(settings.session_secret, salt="btc-signals")


def create_session(username: str) -> str:
    return _serializer.dumps({"u": username, "t": time.time()})


def verify_session(token: str) -> Optional[str]:
    try:
        data = _serializer.loads(token, max_age=86400 * 7)
        return data.get("u")
    except (BadSignature, SignatureExpired):
        return None


def authenticate(username: str, password: str) -> bool:
    from app.database import get_connection
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM users WHERE username = ? AND password_hash = ?",
        (username, pw_hash),
    ).fetchone()
    conn.close()
    return row is not None

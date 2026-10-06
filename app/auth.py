"""Authentication for the dashboard.

Deliberately session-cookie based rather than JWT: this is a server-rendered
dashboard, the cookie is httponly + signed + samesite=lax, and there is no
third-party API consuming it. That is simpler and strictly safer here.

  * Argon2id password hashing
  * Signed, expiring session cookies (itsdangerous)
  * Optional TOTP second factor for admins
  * Per-IP login rate limiting
  * A short numeric PIN for quick unlock on the 7" touchscreen, which grants
    the same role as the user it belongs to but is only accepted from
    localhost - so a stolen PIN is useless remotely.
"""
import os
import secrets
import time

import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from . import config, db

COOKIE = "fa_session"
_hasher = PasswordHasher()
_attempts = {}          # ip -> [count, first_attempt_ts]
MAX_ATTEMPTS = 8
WINDOW_S = 300

ROLE_RANK = {"viewer": 1, "hr": 2, "admin": 3}


def secret_key():
    """Persisted outside git so sessions survive restarts but not a re-clone."""
    path = config.abspath("data/secret.key")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(secrets.token_urlsafe(48))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    with open(path) as fh:
        return fh.read().strip()


def _serializer():
    return URLSafeTimedSerializer(secret_key(), salt="fa-session")


def hash_password(raw):
    return _hasher.hash(raw)


def verify_password(stored, raw):
    try:
        return _hasher.verify(stored, raw)
    except (VerifyMismatchError, VerificationError):
        return False


def rate_limited(ip):
    rec = _attempts.get(ip)
    if not rec:
        return False
    count, first = rec
    if time.time() - first > WINDOW_S:
        _attempts.pop(ip, None)
        return False
    return count >= MAX_ATTEMPTS


def note_failure(ip):
    count, first = _attempts.get(ip, [0, time.time()])
    if time.time() - first > WINDOW_S:
        count, first = 0, time.time()
    _attempts[ip] = [count + 1, first]


def clear_failures(ip):
    _attempts.pop(ip, None)


def issue(response, user_row):
    token = _serializer().dumps({
        "uid": user_row["id"],
        "u": user_row["username"],
        "r": user_row["role"],
    })
    response.set_cookie(
        COOKIE, token,
        max_age=int(config.g("web.session_days", 7)) * 86400,
        httponly=True, samesite="lax",
        secure=bool(config.g("web.https_only", False)),
        path="/")


def revoke(response):
    response.delete_cookie(COOKIE, path="/")


def current_user(request):
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    try:
        data = _serializer().loads(
            token, max_age=int(config.g("web.session_days", 7)) * 86400)
    except (BadSignature, SignatureExpired):
        return None
    return data


def has_role(user, minimum):
    if not user:
        return False
    return ROLE_RANK.get(user.get("r"), 0) >= ROLE_RANK.get(minimum, 99)


def authenticate(conn, username, password, totp_code=None):
    """Return (user_row, error). Never leaks which half was wrong."""
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None or not verify_password(row["pw_hash"], password):
        return None, "Incorrect username or password"
    if row["totp_secret"]:
        if not totp_code:
            return None, "This account requires a 6-digit code"
        if not pyotp.TOTP(row["totp_secret"]).verify(totp_code, valid_window=1):
            return None, "That code is not valid right now"
    return row, None


def authenticate_pin(conn, pin, client_host):
    """Touchscreen quick-unlock. Localhost only, by design."""
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        return None, "PIN unlock only works on the device itself"
    expected = str(config.g("web.admin_pin", ""))
    if not expected or not secrets.compare_digest(str(pin), expected):
        return None, "Wrong PIN"
    row = conn.execute(
        "SELECT * FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()
    if row is None:
        return None, "No admin account exists yet"
    return row, None


def ensure_admin(conn, username="admin", password=None):
    """Create the first admin if the table is empty. Returns the password if
    one was generated so the installer can print it exactly once."""
    existing = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    if existing:
        return None
    generated = password or secrets.token_urlsafe(9)
    conn.execute(
        "INSERT INTO users(username, pw_hash, role, must_change, created_at) "
        "VALUES (?,?,?,?,?)",
        (username, hash_password(generated), "admin", 1, db.now()))
    db.audit(conn, "user.create", actor="installer", target=username,
             detail="initial admin account")
    return generated


def set_password(conn, user_id, new_password, actor=None):
    conn.execute(
        "UPDATE users SET pw_hash=?, must_change=0 WHERE id=?",
        (hash_password(new_password), user_id))
    db.audit(conn, "user.password_change", actor=actor, target=str(user_id))


def enable_totp(conn, user_id, actor=None):
    secret = pyotp.random_base32()
    conn.execute("UPDATE users SET totp_secret=? WHERE id=?", (secret, user_id))
    db.audit(conn, "user.totp_enable", actor=actor, target=str(user_id))
    return secret


def totp_uri(conn, user_id):
    row = conn.execute(
        "SELECT username, totp_secret FROM users WHERE id=?", (user_id,)).fetchone()
    if not row or not row["totp_secret"]:
        return None
    return pyotp.TOTP(row["totp_secret"]).provisioning_uri(
        name=row["username"], issuer_name="FaceAttendance")

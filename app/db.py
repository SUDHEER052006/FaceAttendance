"""SQLite access layer. WAL mode, one schema, no migrations framework needed yet."""
import os
import sqlite3
import time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_code     TEXT UNIQUE,
    name         TEXT NOT NULL,
    department   TEXT,
    phone        TEXT,
    active       INTEGER NOT NULL DEFAULT 1,
    consent_at   TEXT,                 -- DPDP: when this person consented
    created_at   TEXT NOT NULL
);

-- One row per enrolled angle. 512 float32 little-endian = 2048 bytes.
CREATE TABLE IF NOT EXISTS templates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    quality    REAL,
    pose       TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tpl_person ON templates(person_id);

-- Every recognition the pipeline commits to. Append-only.
-- Only *attendance transitions* land here: one row for the check-in and one
-- for the check-out. Repeat sightings of somebody already logged are dropped
-- by the recognizer and never reach this table - that is what keeps the feed
-- readable and stops the same face being written to disk forty times.
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER REFERENCES people(id) ON DELETE SET NULL,
    ts         TEXT NOT NULL,
    kind       TEXT,                  -- check_in | check_out
    score      REAL,
    liveness   REAL,
    votes      INTEGER,
    thumb      TEXT,                  -- NULL unless privacy.store_event_thumbs
    synced     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ev_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_ev_person ON events(person_id, ts);

-- One row per person per day. The thing reports are built from.
CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    day         TEXT NOT NULL,
    check_in    TEXT,
    check_out   TEXT,
    status      TEXT,                  -- punctuality: on_time | late | absent
    half_day    INTEGER NOT NULL DEFAULT 0,  -- short day, tracked separately so
                                             -- a late arrival who leaves early
                                             -- still counts as late
    minutes     INTEGER DEFAULT 0,
    last_seen   TEXT,                  -- most recent sighting, punch or not
    sightings   INTEGER NOT NULL DEFAULT 0,
    corrected   INTEGER NOT NULL DEFAULT 0,
    note        TEXT,
    UNIQUE(person_id, day)
);
CREATE INDEX IF NOT EXISTS idx_att_day ON attendance(day);

CREATE TABLE IF NOT EXISTS unknowns (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    image    TEXT NOT NULL,
    score    REAL,
    liveness REAL,
    resolved INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT UNIQUE NOT NULL,
    pw_hash     TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'admin',   -- admin | hr | viewer
    totp_secret TEXT,
    must_change INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

-- Immutable trail. Everything sensitive writes here.
CREATE TABLE IF NOT EXISTS audit (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    actor   TEXT,
    action  TEXT NOT NULL,
    target  TEXT,
    detail  TEXT,
    ip      TEXT
);

-- Web -> recognizer control channel. The recognizer owns the camera, so the
-- dashboard asks it to do things through this table.
CREATE TABLE IF NOT EXISTS commands (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,          -- enroll_live | enroll_files | reload
    payload    TEXT,
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|error
    progress   INTEGER NOT NULL DEFAULT 0,
    message    TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS health (
    id        INTEGER PRIMARY KEY CHECK (id = 1),
    ts        TEXT,
    cpu_temp  REAL,
    fps       REAL,
    state     TEXT,
    faces     INTEGER,
    uptime_s  INTEGER,
    disk_free INTEGER,
    load1     REAL,
    detail    TEXT
);
"""


def path():
    return config.abspath(config.g("paths.db", "data/attendance.db"))


def connect():
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    conn = sqlite3.connect(p, timeout=15.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    # Attendance must survive abrupt power loss; this is the one place we pay for it.
    conn.execute("PRAGMA synchronous=FULL")
    return conn


# Columns added after the first release. CREATE TABLE IF NOT EXISTS will not
# touch a table that already exists, so new columns are applied by hand here.
# Keep appending to this list; never reorder or remove an entry.
MIGRATIONS = [
    ("attendance", "last_seen", "TEXT"),
    ("attendance", "sightings", "INTEGER NOT NULL DEFAULT 0"),
    ("events",     "kind",      "TEXT"),
]


def migrate(conn):
    """Add any missing columns to an existing database. Idempotent.

    This runs on every connection, and the web process opens one per request,
    so it must stay read-only in the steady state - the back-fill below fires
    only on the single connection that actually adds the column.
    """
    added = []
    for table, column, decl in MIGRATIONS:
        have = {r["name"] for r in
                conn.execute("PRAGMA table_info(%s)" % table).fetchall()}
        if column not in have:
            conn.execute("ALTER TABLE %s ADD COLUMN %s %s"
                         % (table, column, decl))
            added.append(column)
    if "last_seen" in added:
        # Rows written before the column existed have nothing for the
        # de-duplication window to compare against.
        conn.execute("UPDATE attendance SET "
                     "last_seen = COALESCE(check_out, check_in) "
                     "WHERE last_seen IS NULL")
    return added


def init():
    conn = connect()
    conn.executescript(SCHEMA)
    migrate(conn)
    conn.execute("INSERT OR IGNORE INTO health(id) VALUES (1)")
    return conn


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def today():
    return time.strftime("%Y-%m-%d")


def audit(conn, action, actor=None, target=None, detail=None, ip=None):
    conn.execute(
        "INSERT INTO audit(ts, actor, action, target, detail, ip) VALUES (?,?,?,?,?,?)",
        (now(), actor, action, target, detail, ip),
    )

#!/usr/bin/env python3
"""Self-test for the attendance logic. No camera, no models, no network.

Run it with `./run.sh test`. It is fast and it is the thing to run after
touching attendance.py, tracker.py or the commit path in recognizer_service.py,
because the behaviour it checks is exactly what people complain about when it
regresses: one person at the door turning into a screenful of log entries.

Everything runs against a throwaway database in a temp directory, so it is
safe on a live device.
"""
import io
import os
import sqlite3
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL = [], []


def check(label, got, want):
    ok = got == want
    (PASS if ok else FAIL).append(label)
    print("  %-5s %-52s got %-8s want %s"
          % ("ok" if ok else "FAIL", label, got, want))
    return ok


def sandbox():
    """Point FA_CONFIG at a copy of the real config with a temp database."""
    tmp = tempfile.mkdtemp(prefix="fa-selftest-")
    cfg_path = os.path.join(tmp, "config.yaml")
    cfg = io.open(os.path.join(ROOT, "config.yaml"), encoding="utf-8").read()
    io.open(cfg_path, "w", encoding="utf-8").write(cfg.replace(
        "db: data/attendance.db",
        "db: %s" % os.path.join(tmp, "test.db").replace(os.sep, "/")))
    os.environ["FA_CONFIG"] = cfg_path
    return tmp


TMP = sandbox()

from app import attendance, config, db          # noqa: E402  (needs FA_CONFIG)


def new_person(conn, name="Priya Raman"):
    conn.execute("INSERT INTO people(name, active, created_at) VALUES (?,1,?)",
                 (name, db.now()))
    return conn.execute("SELECT id FROM people WHERE name=?",
                        (name,)).fetchone()["id"]


# ===================================================== 1. attendance policy

def test_policy():
    print("\n1. attendance policy - one person, a full day at the door")
    conn = db.init()
    pid = new_person(conn, "Policy Tester")
    day = "2026-10-08"

    # (time, expected outcome, what is actually happening)
    script = [
        ("09:05:00", "check_in",  "arrives"),
        ("09:05:20", "duplicate", "still at the gate 20s later"),
        ("09:05:50", "duplicate", "track respawned after a head turn"),
        ("09:07:30", "seen",      "walks back past the camera"),
        ("09:40:00", "seen",      "back from the canteen"),
        ("10:20:00", "seen",      "still inside the min_work window"),
        ("11:00:00", "check_out", "genuine departure, 115 min in"),
        ("11:00:30", "duplicate", "lingering in frame on the way out"),
        ("18:02:00", "check_out", "real end of day, overrides the earlier one"),
    ]
    for hhmm, want, why in script:
        kind, _ = attendance.record(conn, pid, "%s %s" % (day, hhmm))
        check("%s  %s" % (hhmm, why), kind, want)

    row = conn.execute("SELECT * FROM attendance WHERE person_id=?",
                       (pid,)).fetchone()
    check("check_in preserved", row["check_in"][11:], "09:05:00")
    check("check_out is the last sighting", row["check_out"][11:], "18:02:00")
    check("minutes worked", row["minutes"], 537)
    check("status", row["status"], "on_time")
    check("not a half day", row["half_day"], 0)
    check("sightings counted", row["sightings"], 6)
    check("no event rows from the policy layer",
          conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"], 0)
    conn.close()


# ====================================== 2. the commit path, end to end

def test_commit_dedup():
    print("\n2. recognizer commit path - tracks dying and respawning")
    import numpy as np
    from app import recognizer_service as rs
    from app.tracker import Track

    conn = db.init()
    pid = new_person(conn, "Commit Tester")
    crop = np.full((112, 112, 3), 128, dtype=np.uint8)

    class Matcher:
        names = {pid: "Commit Tester"}
        owners = [pid]

    svc = object.__new__(rs.Service)       # the real methods, no camera
    svc.conn = conn
    svc.liveness = None
    svc.matcher = Matcher()
    svc.banner = None
    svc.seen_recent = {}
    svc.unknown_recent = []
    svc.spoof_logged = {}

    for _ in range(25):
        svc.commit(Track((10, 10, 80, 80)), pid, 0.61, crop, 5)

    check("25 fresh tracks -> event rows",
          conn.execute("SELECT COUNT(*) c FROM events WHERE person_id=?",
                       (pid,)).fetchone()["c"], 1)
    row = conn.execute("SELECT * FROM attendance WHERE person_id=?",
                       (pid,)).fetchone()
    check("25 sightings did not fabricate a check-out",
          row["check_out"], None)

    thumbs = os.path.join(config.abspath("data"), "thumbs")
    n_files = (sum(len(f) for _, _, f in os.walk(thumbs))
               if os.path.isdir(thumbs) else 0)
    check("face images written (storage off by default)", n_files, 0)

    # Let the cooldown lapse and backdate the arrival: now a departure is due.
    svc.seen_recent.clear()
    conn.execute("UPDATE attendance SET check_in=?, last_seen=? WHERE id=?",
                 (time.strftime("%Y-%m-%d 08:00:00"),
                  time.strftime("%Y-%m-%d 08:00:00"), row["id"]))
    svc.commit(Track((10, 10, 80, 80)), pid, 0.64, crop, 6)

    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM events WHERE person_id=? ORDER BY id", (pid,))]
    check("event kinds after the cooldown lapsed", kinds,
          ["check_in", "check_out"])
    conn.close()


# ================================================ 3. unknown-face collapsing

def test_unknown_dedup():
    print("\n3. unknown faces - one stranger, one snapshot")
    import numpy as np
    from app import recognizer_service as rs
    from app.tracker import Track

    conn = db.init()
    crop = np.full((112, 112, 3), 128, dtype=np.uint8)

    svc = object.__new__(rs.Service)
    svc.conn = conn
    svc.liveness = None
    svc.matcher = type("M", (), {"names": {}, "owners": []})()
    svc.banner = None
    svc.seen_recent = {}
    svc.unknown_recent = []
    svc.spoof_logged = {}

    rng = np.random.default_rng(7)

    def unit(v):
        return v / np.linalg.norm(v)

    stranger = unit(rng.normal(size=512).astype(np.float32))
    for _ in range(30):
        jittered = unit(stranger + rng.normal(scale=0.02, size=512
                                              ).astype(np.float32))
        svc.log_unknown(Track((10, 10, 80, 80)), crop, jittered, 0.21)

    check("30 tracks of the same stranger -> snapshots",
          conn.execute("SELECT COUNT(*) c FROM unknowns").fetchone()["c"], 1)

    svc.log_unknown(Track((10, 10, 80, 80)), crop,
                    unit(rng.normal(size=512).astype(np.float32)), 0.19)
    check("a genuinely different stranger is still logged",
          conn.execute("SELECT COUNT(*) c FROM unknowns").fetchone()["c"], 2)
    conn.close()


# ========================================================= 4. anti-spoofing

def test_spoof():
    print("\n4. anti-spoofing - rejected every time, logged once")
    import numpy as np
    from app import recognizer_service as rs
    from app.tracker import Track

    conn = db.init()
    pid = new_person(conn, "Spoof Tester")
    crop = np.full((112, 112, 3), 128, dtype=np.uint8)

    class AlwaysSpoof:
        name = "selftest"

        def score(self, _crop):
            return 0.10            # always below any sane threshold

    svc = object.__new__(rs.Service)
    svc.conn = conn
    svc.liveness = AlwaysSpoof()
    svc.matcher = type("M", (), {"names": {pid: "Spoof Tester"},
                                 "owners": [pid]})()
    svc.banner = None
    svc.seen_recent = {}
    svc.unknown_recent = []
    svc.spoof_logged = {}

    for _ in range(20):
        svc.commit(Track((10, 10, 80, 80)), pid, 0.61, crop, 5)

    check("spoofed attendance rows", conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE person_id=?",
        (pid,)).fetchone()["c"], 0)
    check("spoofed event rows", conn.execute(
        "SELECT COUNT(*) c FROM events WHERE person_id=?",
        (pid,)).fetchone()["c"], 0)
    check("audit rows (throttled, not one per frame)", conn.execute(
        "SELECT COUNT(*) c FROM audit WHERE action='liveness.reject'"
    ).fetchone()["c"], 1)
    conn.close()


# ======================================== 5. in-place upgrade of an old db

OLD_SCHEMA = """
CREATE TABLE people(id INTEGER PRIMARY KEY AUTOINCREMENT, emp_code TEXT UNIQUE,
  name TEXT NOT NULL, department TEXT, phone TEXT,
  active INTEGER NOT NULL DEFAULT 1, consent_at TEXT, created_at TEXT NOT NULL);
CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER,
  ts TEXT NOT NULL, score REAL, liveness REAL, votes INTEGER, thumb TEXT,
  synced INTEGER NOT NULL DEFAULT 0);
CREATE TABLE attendance(id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id INTEGER NOT NULL, day TEXT NOT NULL, check_in TEXT, check_out TEXT,
  status TEXT, half_day INTEGER NOT NULL DEFAULT 0, minutes INTEGER DEFAULT 0,
  corrected INTEGER NOT NULL DEFAULT 0, note TEXT, UNIQUE(person_id, day));
INSERT INTO people(name, created_at) VALUES ('Legacy Row', '2026-10-01');
INSERT INTO attendance(person_id, day, check_in, check_out, status, minutes)
  VALUES (1, '2026-10-07', '2026-10-07 09:02:00', '2026-10-07 18:10:00',
          'on_time', 548);
INSERT INTO events(person_id, ts, score) VALUES (1, '2026-10-07 09:02:00', 0.6);
"""


def test_migration():
    print("\n5. migration - a database written before these columns existed")
    tmp = tempfile.mkdtemp(prefix="fa-selftest-old-")
    path = os.path.join(tmp, "old.db")
    legacy = sqlite3.connect(path)
    legacy.executescript(OLD_SCHEMA)
    legacy.commit()
    legacy.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    added = db.migrate(conn)
    check("columns added on first open", sorted(added),
          ["kind", "last_seen", "sightings"])
    check("second open adds nothing (no write per web request)",
          db.migrate(conn), [])
    check("existing attendance rows kept", conn.execute(
        "SELECT COUNT(*) c FROM attendance").fetchone()["c"], 1)
    check("last_seen back-filled", conn.execute(
        "SELECT last_seen FROM attendance").fetchone()["last_seen"],
        "2026-10-07 18:10:00")
    conn.close()


# ==================================================================== main

def main():
    print("Face Attendance self-test")
    print("config : %s" % os.environ["FA_CONFIG"])
    print("policy : min_rescan_gap_s=%s  min_work_minutes=%s"
          % (config.g("attendance.min_rescan_gap_s"),
             config.g("attendance.min_work_minutes")))
    print("privacy: store_event_thumbs=%s"
          % config.g("privacy.store_event_thumbs"))

    for fn in (test_policy, test_commit_dedup, test_unknown_dedup,
               test_spoof, test_migration):
        try:
            fn()
        except Exception as exc:
            FAIL.append("%s raised %s: %s" % (fn.__name__,
                                              type(exc).__name__, exc))
            print("  FAIL  %s raised %s: %s"
                  % (fn.__name__, type(exc).__name__, exc))

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("\nfailures:")
        for f in FAIL:
            print("  - %s" % f)
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    sys.exit(main())

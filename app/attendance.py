"""Attendance policy: turning recognition events into check-in/check-out rows.

Rules, deliberately simple and all driven from config.yaml:

  * First recognition of the day  -> check_in
  * Any later recognition         -> check_out (last one of the day wins)
  * Repeat sightings inside attendance.min_rescan_gap_s are ignored entirely

Status and day-length are two SEPARATE facts, and conflating them loses
information a payroll report needs:

  status    on_time | late | absent      - was the arrival punctual?
  half_day  0 | 1                        - was the day shorter than the
                                           half-day threshold?

So somebody who turns up at 10:45 and leaves at 13:30 is recorded as *both*
late and a half day, and still shows up in the late count.
"""
import datetime as dt

from . import config, db


def _parse_hhmm(value, fallback):
    try:
        hh, mm = str(value).split(":")
        return int(hh), int(mm)
    except Exception:
        return fallback


def _minutes(ts):
    """'YYYY-mm-dd HH:MM:SS' -> minutes since midnight."""
    t = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return t.hour * 60 + t.minute + t.second / 60.0


def classify(check_in, check_out):
    """Return (status, minutes_worked, half_day)."""
    sh, sm = _parse_hhmm(config.g("attendance.shift_start", "09:00"), (9, 0))
    grace = int(config.g("attendance.grace_minutes", 10))
    half = int(config.g("attendance.half_day_minutes", 240))

    worked = 0
    if check_in and check_out:
        worked = int(max(0.0, _minutes(check_out) - _minutes(check_in)))

    if not check_in:
        return "absent", 0, 0

    status = "late" if _minutes(check_in) > (sh * 60 + sm + grace) else "on_time"
    # Only a completed day can be judged short - somebody still at their desk
    # has not worked a half day, they just have not left yet.
    is_half = 1 if (check_out and worked < half) else 0
    return status, worked, is_half


def record(conn, person_id, ts=None):
    """Apply one confirmed recognition.

    Returns (kind, row) where kind is 'check_in' | 'check_out' | 'duplicate'.
    """
    ts = ts or db.now()
    day = ts[:10]
    gap = int(config.g("attendance.min_rescan_gap_s", 120))

    row = conn.execute(
        "SELECT * FROM attendance WHERE person_id=? AND day=?",
        (person_id, day)).fetchone()

    if row is None:
        status, worked, half = classify(ts, None)
        conn.execute(
            "INSERT INTO attendance(person_id, day, check_in, status, half_day, "
            "minutes) VALUES (?,?,?,?,?,?)",
            (person_id, day, ts, status, half, worked))
        return "check_in", conn.execute(
            "SELECT * FROM attendance WHERE person_id=? AND day=?",
            (person_id, day)).fetchone()

    # Debounce: the same person seen again a moment later is not a new punch.
    last = row["check_out"] or row["check_in"]
    if last:
        delta = (dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                 - dt.datetime.strptime(last, "%Y-%m-%d %H:%M:%S")).total_seconds()
        if delta < gap:
            return "duplicate", row

    status, worked, half = classify(row["check_in"], ts)
    conn.execute(
        "UPDATE attendance SET check_out=?, status=?, half_day=?, minutes=? "
        "WHERE id=?", (ts, status, half, worked, row["id"]))
    return "check_out", conn.execute(
        "SELECT * FROM attendance WHERE id=?", (row["id"],)).fetchone()


def day_rows(conn, day=None):
    """Every active person for a day, present or not. Drives the Today page."""
    day = day or db.today()
    rows = conn.execute("""
        SELECT p.id AS person_id, p.name, p.emp_code, p.department,
               a.check_in, a.check_out, a.status, a.half_day, a.minutes,
               a.corrected, a.note, a.id AS att_id
        FROM people p
        LEFT JOIN attendance a ON a.person_id = p.id AND a.day = ?
        WHERE p.active = 1
        ORDER BY (a.check_in IS NULL), a.check_in ASC, p.name ASC
    """, (day,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["status"] = d["status"] or "absent"
        d["minutes"] = d["minutes"] or 0
        d["half_day"] = d["half_day"] or 0
        out.append(d)
    return out


def summary(conn, day=None):
    rows = day_rows(conn, day)
    total = len(rows)
    absent = sum(1 for r in rows if r["status"] == "absent")
    late = sum(1 for r in rows if r["status"] == "late")
    on_time = sum(1 for r in rows if r["status"] == "on_time")
    half = sum(1 for r in rows if r["half_day"])
    present = total - absent
    return {
        "day": day or db.today(),
        "total": total,
        "present": present,
        "absent": absent,
        "late": late,
        "on_time": on_time,
        "half_day": half,
        "percent": round(100.0 * present / total, 1) if total else 0.0,
    }


def range_rows(conn, start, end, person_id=None):
    """Report query over a date range."""
    sql = """
        SELECT a.day, p.name, p.emp_code, p.department, a.check_in, a.check_out,
               a.status, a.half_day, a.minutes, a.corrected, a.note
        FROM attendance a JOIN people p ON p.id = a.person_id
        WHERE a.day BETWEEN ? AND ?
    """
    args = [start, end]
    if person_id:
        sql += " AND a.person_id = ?"
        args.append(person_id)
    sql += " ORDER BY a.day DESC, a.check_in ASC"
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def recent_events(conn, limit=25):
    return [dict(r) for r in conn.execute("""
        SELECT e.id, e.ts, e.score, e.liveness, e.votes, e.thumb,
               COALESCE(p.name, 'Unknown') AS name, e.person_id
        FROM events e LEFT JOIN people p ON p.id = e.person_id
        ORDER BY e.id DESC LIMIT ?
    """, (limit,)).fetchall()]


def correct(conn, att_id, check_in, check_out, note, actor, ip=None):
    """Manual correction. Always audited - never silently editable."""
    row = conn.execute("SELECT * FROM attendance WHERE id=?", (att_id,)).fetchone()
    if row is None:
        return None
    status, worked, half = classify(check_in or None, check_out or None)
    conn.execute(
        "UPDATE attendance SET check_in=?, check_out=?, status=?, half_day=?, "
        "minutes=?, corrected=1, note=? WHERE id=?",
        (check_in or None, check_out or None, status, half, worked, note, att_id))
    db.audit(conn, "attendance.correct", actor=actor, target=str(att_id),
             detail="was in=%s out=%s -> in=%s out=%s; note=%s" % (
                 row["check_in"], row["check_out"], check_in, check_out, note),
             ip=ip)
    return conn.execute("SELECT * FROM attendance WHERE id=?", (att_id,)).fetchone()

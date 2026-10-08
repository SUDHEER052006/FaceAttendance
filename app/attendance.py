"""Attendance policy: turning recognition events into check-in/check-out rows.

Rules, deliberately simple and all driven from config.yaml:

  * First recognition of the day                 -> check_in
  * Seen again within min_rescan_gap_s           -> ignored entirely
  * Seen again, but less than min_work_minutes
    after check-in                               -> a re-sighting, not a punch
  * Seen again after that                        -> check_out (latest one wins)

That third rule is the one that matters in the real world. A doorway camera
sees the same person on their way to the canteen, coming back from a smoke, or
just standing in the corridor talking. Treating every one of those as a
departure produced twenty-minute working days and a log full of the same face.
A sighting only becomes a departure once somebody has been at work long enough
for leaving to be plausible; everything before that just refreshes last_seen.

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


def _seconds_between(a, b):
    return (dt.datetime.strptime(b, "%Y-%m-%d %H:%M:%S")
            - dt.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")).total_seconds()


def record(conn, person_id, ts=None):
    """Apply one confirmed recognition.

    Returns (kind, row) where kind is one of:

      check_in   a new row was created for the day
      check_out  the departure time was set, or moved later
      seen       genuine sighting, but too soon after check-in to be a
                 departure - only last_seen moved
      duplicate  inside the rescan window; nothing was written at all

    The caller must not log an event or store a face image for 'seen' or
    'duplicate'. Those two outcomes exist precisely so that it does not.
    """
    ts = ts or db.now()
    day = ts[:10]
    gap = int(config.g("attendance.min_rescan_gap_s", 90))
    min_work = int(config.g("attendance.min_work_minutes", 90))

    row = conn.execute(
        "SELECT * FROM attendance WHERE person_id=? AND day=?",
        (person_id, day)).fetchone()

    if row is None:
        status, worked, half = classify(ts, None)
        conn.execute(
            "INSERT INTO attendance(person_id, day, check_in, last_seen, "
            "sightings, status, half_day, minutes) VALUES (?,?,?,?,?,?,?,?)",
            (person_id, day, ts, ts, 1, status, half, worked))
        return "check_in", conn.execute(
            "SELECT * FROM attendance WHERE person_id=? AND day=?",
            (person_id, day)).fetchone()

    # Debounce: the same person seen again a moment later is not a new punch.
    last = row["last_seen"] or row["check_out"] or row["check_in"]
    if last and _seconds_between(last, ts) < gap:
        return "duplicate", row

    # Past the debounce, so this is a real new sighting - but not necessarily
    # a departure. Nobody goes home eight minutes after arriving.
    if row["check_in"] and _seconds_between(row["check_in"], ts) < min_work * 60:
        conn.execute(
            "UPDATE attendance SET last_seen=?, sightings=sightings+1 WHERE id=?",
            (ts, row["id"]))
        return "seen", conn.execute(
            "SELECT * FROM attendance WHERE id=?", (row["id"],)).fetchone()

    status, worked, half = classify(row["check_in"], ts)
    conn.execute(
        "UPDATE attendance SET check_out=?, last_seen=?, sightings=sightings+1, "
        "status=?, half_day=?, minutes=? WHERE id=?",
        (ts, ts, status, half, worked, row["id"]))
    return "check_out", conn.execute(
        "SELECT * FROM attendance WHERE id=?", (row["id"],)).fetchone()


def day_rows(conn, day=None):
    """Every active person for a day, present or not. Drives the Today page."""
    day = day or db.today()
    rows = conn.execute("""
        SELECT p.id AS person_id, p.name, p.emp_code, p.department,
               a.check_in, a.check_out, a.status, a.half_day, a.minutes,
               a.last_seen, a.sightings, a.corrected, a.note, a.id AS att_id
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
        d["sightings"] = d["sightings"] or 0
        # "Still here" drives the live headcount: checked in, no departure yet.
        d["inside"] = bool(d["check_in"] and not d["check_out"])
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
    inside = sum(1 for r in rows if r["inside"])
    return {
        "day": day or db.today(),
        "total": total,
        "present": present,       # turned up at all today
        "inside": inside,         # turned up and has not checked out yet
        "left": present - inside,
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


def initials(name):
    """'Priya Raman' -> 'PR'. Shown in the feed instead of a stored face."""
    parts = [p for p in str(name or "").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def avatar(name):
    """Stable colour slot 0-5 for a name, so the same person always gets the
    same tint without storing anything."""
    return sum(ord(c) for c in str(name or "")) % 6


def recent_events(conn, limit=25, day=None):
    """The activity feed. Today only by default - a dashboard showing
    yesterday's arrivals at 9 a.m. only confuses whoever is reading it.

    The feed NEVER carries a face crop, whatever privacy.store_event_thumbs
    is set to. It identifies people by name and initials, which is all an
    attendance log needs, and it is consumed by /api/live - which the kiosk
    polls without a login. A stored crop is reachable only from that person's
    own detail page, behind authentication.
    """
    day = day or db.today()
    rows = [dict(r) for r in conn.execute("""
        SELECT e.id, e.ts, e.kind, e.score, e.liveness, e.votes,
               COALESCE(p.name, 'Unknown') AS name, p.emp_code, p.department,
               e.person_id
        FROM events e LEFT JOIN people p ON p.id = e.person_id
        WHERE e.ts >= ? ORDER BY e.id DESC LIMIT ?
    """, (day + " 00:00:00", limit)).fetchall()]
    for r in rows:
        r["initials"] = initials(r["name"])
        # Always an int, so the template never emits class="cNone"; the
        # .av.unknown rule overrides the colour for unmatched faces anyway.
        r["avatar"] = avatar(r["name"])
        r["kind"] = r["kind"] or "check_in"
    return rows


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

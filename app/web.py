"""Process B: the dashboard.

Runs as its own systemd unit pinned to core 2. It never touches the camera -
the recognizer owns that. For a live preview it streams the JPEG the
recognizer publishes to /dev/shm; to enroll somebody it drops a row in the
`commands` table and polls for progress.
"""
import csv
import io
import json
import os
import secrets
import time

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, RedirectResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import attendance, auth, config, db

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Face Attendance", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")),
          name="static")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))

# Initials and a stable colour slot stand in for a face image everywhere a
# list shows a person. Face crops appear only on the Unknown review page and
# on a person's own detail page, both behind authentication.
templates.env.filters["initials"] = attendance.initials
templates.env.filters["avatar"] = attendance.avatar

STATUS_LABEL = {
    "on_time": "On time",
    "late": "Late",
    "half_day": "Half day",
    "absent": "Absent",
}


# --------------------------------------------------------------------- helpers

def conn():
    return db.init()


def recognizer_up(c):
    """True if the recognizer has sent a heartbeat recently. Everything the
    dashboard says about 'now' is meaningless if this is False, so every page
    that implies live data checks it."""
    row = c.execute("SELECT ts FROM health WHERE id=1").fetchone()
    if not row or not row["ts"]:
        return False
    try:
        return (time.time() - time.mktime(
            time.strptime(row["ts"], "%Y-%m-%d %H:%M:%S"))) < 45
    except Exception:
        return False


def pending_unknowns(c):
    return c.execute(
        "SELECT COUNT(*) AS n FROM unknowns WHERE resolved=0").fetchone()["n"]


def client_ip(request):
    return request.client.host if request.client else "-"


def csrf_token(request):
    token = request.cookies.get("fa_csrf")
    return token or secrets.token_urlsafe(24)


def check_csrf(request, token):
    cookie = request.cookies.get("fa_csrf")
    if not cookie or not token or not secrets.compare_digest(cookie, token):
        raise HTTPException(status_code=400, detail="Stale form - please retry")


def render(request, name, **ctx):
    user = auth.current_user(request)
    token = csrf_token(request)
    ctx.setdefault("user", user)
    ctx.setdefault("site", config.g("site.name", "Main Gate"))
    ctx.setdefault("status_label", STATUS_LABEL)
    # Drives the red count on the Unknown tab in the nav rail.
    if user and "pending_unknowns" not in ctx:
        try:
            ctx["pending_unknowns"] = pending_unknowns(conn())
        except Exception:
            ctx["pending_unknowns"] = 0
    ctx["csrf"] = token
    ctx["request"] = request
    response = templates.TemplateResponse(name, ctx)
    response.set_cookie("fa_csrf", token, httponly=False, samesite="lax",
                        path="/", max_age=86400)
    return response


def require(request, minimum="viewer"):
    user = auth.current_user(request)
    if not auth.has_role(user, minimum):
        raise HTTPException(status_code=303, detail="/login",
                            headers={"Location": "/login"})
    return user


@app.exception_handler(HTTPException)
async def redirect_handler(request, exc):
    if exc.status_code == 303 and "Location" in (exc.headers or {}):
        return RedirectResponse(exc.headers["Location"], status_code=303)
    return HTMLResponse(
        "<h1>%s</h1><p>%s</p><p><a href='/'>Back</a></p>"
        % (exc.status_code, exc.detail), status_code=exc.status_code)


# ----------------------------------------------------------------- kiosk view

@app.get("/", response_class=HTMLResponse)
def kiosk(request: Request):
    """Fullscreen view for the 7 inch screen. No login needed to watch it."""
    c = conn()
    return render(request, "kiosk.html",
                  summary=attendance.summary(c),
                  events=attendance.recent_events(c, 8))


def mjpeg_frames():
    path = config.g("paths.frame", "/dev/shm/fa_frame.jpg")
    placeholder = os.path.join(HERE, "static", "offline.jpg")
    last_mtime = 0.0
    while True:
        try:
            mtime = os.path.getmtime(path)
            if mtime != last_mtime:
                last_mtime = mtime
                with open(path, "rb") as fh:
                    data = fh.read()
            else:
                time.sleep(0.05)
                continue
        except OSError:
            try:
                with open(placeholder, "rb") as fh:
                    data = fh.read()
            except OSError:
                data = b""
            time.sleep(1.0)
        if data:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
        time.sleep(0.04)


@app.get("/stream.mjpg")
def stream():
    return StreamingResponse(
        mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache", "Pragma": "no-cache"})


@app.get("/api/live")
def api_live():
    """Polled by the kiosk and the dashboard feed every two seconds.

    Face crops are already stripped by attendance.recent_events() unless the
    installation turned them on, so this endpoint is safe to expose on the
    kiosk page, which needs no login.
    """
    c = conn()
    health = dict(c.execute("SELECT * FROM health WHERE id=1").fetchone() or {})
    return JSONResponse({
        "summary": attendance.summary(c),
        "events": attendance.recent_events(c, 10),
        "health": health,
        "recognizer_up": recognizer_up(c),
        "clock": time.strftime("%H:%M:%S"),
    })


# ---------------------------------------------------------------------- login

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, error: str = ""):
    return render(request, "login.html", error=error,
                  local=client_ip(request) in ("127.0.0.1", "::1"))


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...),
          totp: str = Form(""), csrf: str = Form(...)):
    check_csrf(request, csrf)
    ip = client_ip(request)
    if auth.rate_limited(ip):
        return render(request, "login.html",
                      error="Too many attempts. Wait five minutes.", local=False)
    c = conn()
    row, error = auth.authenticate(c, username, password, totp or None)
    if error:
        auth.note_failure(ip)
        db.audit(c, "login.fail", actor=username, detail=error, ip=ip)
        return render(request, "login.html", error=error,
                      local=ip in ("127.0.0.1", "::1"))
    auth.clear_failures(ip)
    db.audit(c, "login.ok", actor=username, ip=ip)
    target = "/admin/account" if row["must_change"] else "/admin"
    response = RedirectResponse(target, status_code=303)
    auth.issue(response, row)
    return response


@app.post("/login/pin")
def login_pin(request: Request, pin: str = Form(...), csrf: str = Form(...)):
    check_csrf(request, csrf)
    ip = client_ip(request)
    if auth.rate_limited(ip):
        return render(request, "login.html", error="Too many attempts.", local=True)
    c = conn()
    row, error = auth.authenticate_pin(c, pin, ip)
    if error:
        auth.note_failure(ip)
        db.audit(c, "login.pin_fail", detail=error, ip=ip)
        return render(request, "login.html", error=error, local=True)
    auth.clear_failures(ip)
    db.audit(c, "login.pin_ok", actor=row["username"], ip=ip)
    response = RedirectResponse("/admin", status_code=303)
    auth.issue(response, row)
    return response


@app.post("/logout")
def logout(request: Request):
    response = RedirectResponse("/", status_code=303)
    auth.revoke(response)
    return response


# ------------------------------------------------------------------ dashboard

@app.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request):
    require(request, "viewer")
    c = conn()
    health = dict(c.execute("SELECT * FROM health WHERE id=1").fetchone() or {})
    counts = c.execute("""
        SELECT (SELECT COUNT(*) FROM people WHERE active=1)      AS people,
               (SELECT COUNT(*) FROM templates)                  AS templates,
               (SELECT COUNT(*) FROM unknowns WHERE resolved=0)   AS unknowns
    """).fetchone()
    return render(request, "dashboard.html",
                  summary=attendance.summary(c),
                  events=attendance.recent_events(c, 10),
                  health=health, counts=dict(counts),
                  recognizer_up=recognizer_up(c),
                  pending_unknowns=counts["unknowns"],
                  store_thumbs=bool(config.g("privacy.store_event_thumbs", False)),
                  shift_start=config.g("attendance.shift_start", "09:00"),
                  grace=config.g("attendance.grace_minutes", 10))


@app.get("/admin/today", response_class=HTMLResponse)
def today_page(request: Request, day: str = ""):
    require(request, "viewer")
    c = conn()
    day = day or db.today()
    return render(request, "today.html", day=day,
                  rows=attendance.day_rows(c, day),
                  summary=attendance.summary(c, day))


@app.post("/admin/attendance/{att_id}/correct")
def correct_attendance(request: Request, att_id: int,
                       check_in: str = Form(""), check_out: str = Form(""),
                       note: str = Form(""), csrf: str = Form(...)):
    user = require(request, "hr")
    check_csrf(request, csrf)
    c = conn()
    attendance.correct(c, att_id, check_in.strip() or None,
                       check_out.strip() or None, note.strip(),
                       actor=user["u"], ip=client_ip(request))
    return RedirectResponse("/admin/today", status_code=303)


# ------------------------------------------------------------------- register

@app.get("/admin/register", response_class=HTMLResponse)
def register_form(request: Request):
    require(request, "hr")
    return render(request, "register.html")


@app.post("/admin/register")
def register_submit(request: Request, name: str = Form(...),
                    emp_code: str = Form(""), department: str = Form(""),
                    phone: str = Form(""), samples: int = Form(7),
                    consent: str = Form(""), csrf: str = Form(...)):
    user = require(request, "hr")
    check_csrf(request, csrf)
    if not consent:
        return render(request, "register.html",
                      error="Biometric consent must be recorded before enrolling.")
    c = conn()
    try:
        cur = c.execute(
            "INSERT INTO people(emp_code, name, department, phone, consent_at, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (emp_code.strip() or None, name.strip(), department.strip() or None,
             phone.strip() or None, db.now(), db.now()))
    except Exception:
        return render(request, "register.html",
                      error="That employee code is already in use.")
    person_id = cur.lastrowid
    db.audit(c, "person.create", actor=user["u"], target=str(person_id),
             detail=name, ip=client_ip(request))
    cmd = c.execute(
        "INSERT INTO commands(kind, payload, created_at) VALUES (?,?,?)",
        ("enroll_live",
         json.dumps({"person_id": person_id, "samples": int(samples),
                     "name": name.strip()}),
         db.now()))
    return RedirectResponse("/admin/register/capture/%d" % cmd.lastrowid,
                            status_code=303)


@app.get("/admin/register/capture/{cmd_id}", response_class=HTMLResponse)
def register_capture(request: Request, cmd_id: int):
    require(request, "hr")
    return render(request, "capture.html", cmd_id=cmd_id)


@app.get("/api/enroll/{cmd_id}")
def enroll_status(request: Request, cmd_id: int):
    require(request, "hr")
    c = conn()
    row = c.execute("SELECT * FROM commands WHERE id=?", (cmd_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "No such enrollment")
    return JSONResponse(dict(row))


@app.post("/admin/people/{person_id}/photos")
async def upload_photos(request: Request, person_id: int,
                        files: list[UploadFile] = File(...),
                        csrf: str = Form(...)):
    user = require(request, "hr")
    check_csrf(request, csrf)
    c = conn()
    staged = []
    folder = config.abspath(os.path.join("data", "uploads", str(person_id)))
    os.makedirs(folder, exist_ok=True)
    for upload in files:
        if not upload.filename:
            continue
        safe = "%d_%s" % (int(time.time() * 1000),
                          os.path.basename(upload.filename)[:60])
        path = os.path.join(folder, safe)
        with open(path, "wb") as fh:
            fh.write(await upload.read())
        staged.append(path)
    if not staged:
        return RedirectResponse("/admin/people/%d" % person_id, status_code=303)
    cmd = c.execute(
        "INSERT INTO commands(kind, payload, created_at) VALUES (?,?,?)",
        ("enroll_files",
         json.dumps({"person_id": person_id, "paths": staged}), db.now()))
    db.audit(c, "person.enroll_photos", actor=user["u"], target=str(person_id),
             detail="%d files" % len(staged), ip=client_ip(request))
    return RedirectResponse("/admin/register/capture/%d" % cmd.lastrowid,
                            status_code=303)


# --------------------------------------------------------------------- people

@app.get("/admin/people", response_class=HTMLResponse)
def people_page(request: Request, q: str = ""):
    require(request, "viewer")
    c = conn()
    sql = """
        SELECT p.*, (SELECT COUNT(*) FROM templates t WHERE t.person_id=p.id)
               AS templates,
               (SELECT MAX(ts) FROM events e WHERE e.person_id=p.id) AS last_seen
        FROM people p
    """
    args = []
    if q:
        sql += " WHERE p.name LIKE ? OR p.emp_code LIKE ? OR p.department LIKE ?"
        args = ["%%%s%%" % q] * 3
    sql += " ORDER BY p.active DESC, p.name"
    return render(request, "people.html", q=q,
                  rows=[dict(r) for r in c.execute(sql, args).fetchall()])


@app.get("/admin/people/{person_id}", response_class=HTMLResponse)
def person_detail(request: Request, person_id: int):
    require(request, "viewer")
    c = conn()
    person = c.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
    if person is None:
        raise HTTPException(404, "No such person")
    # If privacy.store_event_thumbs is on, surface the stored proof image for
    # each day here - on an authenticated, single-person page - and nowhere
    # else. This is the only route by which a stored attendance crop is shown.
    proof = {}
    for row in c.execute(
            "SELECT ts, kind, thumb FROM events WHERE person_id=? "
            "AND thumb IS NOT NULL ORDER BY id", (person_id,)).fetchall():
        proof.setdefault(row["ts"][:10], {})[row["kind"] or "check_in"] = \
            row["thumb"]

    return render(request, "person.html", person=dict(person), proof=proof,
                  templates_=[dict(r) for r in c.execute(
                      "SELECT id, pose, quality, created_at FROM templates "
                      "WHERE person_id=? ORDER BY id", (person_id,)).fetchall()],
                  history=[dict(r) for r in c.execute(
                      "SELECT day, check_in, check_out, status, half_day, minutes "
                      "FROM attendance WHERE person_id=? ORDER BY day DESC "
                      "LIMIT 30", (person_id,)).fetchall()])


@app.post("/admin/people/{person_id}/reenroll")
def reenroll(request: Request, person_id: int, samples: int = Form(7),
             replace: str = Form(""), csrf: str = Form(...)):
    user = require(request, "hr")
    check_csrf(request, csrf)
    c = conn()
    person = c.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
    if person is None:
        raise HTTPException(404, "No such person")
    if replace:
        c.execute("DELETE FROM templates WHERE person_id=?", (person_id,))
        db.audit(c, "person.templates_cleared", actor=user["u"],
                 target=str(person_id), ip=client_ip(request))
    cmd = c.execute(
        "INSERT INTO commands(kind, payload, created_at) VALUES (?,?,?)",
        ("enroll_live",
         json.dumps({"person_id": person_id, "samples": int(samples),
                     "name": person["name"]}), db.now()))
    return RedirectResponse("/admin/register/capture/%d" % cmd.lastrowid,
                            status_code=303)


@app.post("/admin/people/{person_id}/active")
def toggle_active(request: Request, person_id: int, csrf: str = Form(...)):
    user = require(request, "hr")
    check_csrf(request, csrf)
    c = conn()
    c.execute("UPDATE people SET active = 1 - active WHERE id=?", (person_id,))
    db.audit(c, "person.toggle_active", actor=user["u"], target=str(person_id),
             ip=client_ip(request))
    c.execute("INSERT INTO commands(kind, created_at) VALUES ('reload', ?)",
              (db.now(),))
    return RedirectResponse("/admin/people/%d" % person_id, status_code=303)


@app.post("/admin/people/{person_id}/delete")
def delete_person(request: Request, person_id: int, confirm: str = Form(""),
                  csrf: str = Form(...)):
    """DPDP erasure: biometric templates and face images are destroyed.

    Attendance rows are kept but anonymised (person_id -> NULL on events), so
    historical headcount stays correct without retaining the biometrics.
    """
    user = require(request, "admin")
    check_csrf(request, csrf)
    c = conn()
    person = c.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
    if person is None:
        raise HTTPException(404, "No such person")
    if confirm.strip().lower() != person["name"].strip().lower():
        return RedirectResponse(
            "/admin/people/%d?err=name-mismatch" % person_id, status_code=303)

    import shutil
    folder = os.path.join(config.abspath(config.g("paths.faces")), str(person_id))
    shutil.rmtree(folder, ignore_errors=True)
    c.execute("DELETE FROM templates WHERE person_id=?", (person_id,))
    c.execute("DELETE FROM people WHERE id=?", (person_id,))
    db.audit(c, "person.delete", actor=user["u"], target=str(person_id),
             detail="DPDP erasure: %s" % person["name"], ip=client_ip(request))
    c.execute("INSERT INTO commands(kind, created_at) VALUES ('reload', ?)",
              (db.now(),))
    return RedirectResponse("/admin/people", status_code=303)


# -------------------------------------------------------------------- unknowns

@app.get("/admin/unknowns", response_class=HTMLResponse)
def unknowns_page(request: Request):
    require(request, "hr")
    c = conn()
    return render(request, "unknowns.html",
                  retention_days=config.g("retention.unknown_days", 14),
                  rows=[dict(r) for r in c.execute(
                      "SELECT * FROM unknowns WHERE resolved=0 "
                      "ORDER BY id DESC LIMIT 60").fetchall()])


@app.post("/admin/unknowns/{unknown_id}/dismiss")
def dismiss_unknown(request: Request, unknown_id: int, csrf: str = Form(...)):
    require(request, "hr")
    check_csrf(request, csrf)
    conn().execute("UPDATE unknowns SET resolved=1 WHERE id=?", (unknown_id,))
    return RedirectResponse("/admin/unknowns", status_code=303)


@app.post("/admin/unknowns/{unknown_id}/enroll")
def enroll_unknown(request: Request, unknown_id: int, name: str = Form(...),
                   emp_code: str = Form(""), csrf: str = Form(...)):
    """Turn a logged unknown into a person, seeding from that snapshot."""
    user = require(request, "hr")
    check_csrf(request, csrf)
    c = conn()
    row = c.execute("SELECT * FROM unknowns WHERE id=?", (unknown_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "No such snapshot")
    cur = c.execute(
        "INSERT INTO people(emp_code, name, consent_at, created_at) "
        "VALUES (?,?,?,?)",
        (emp_code.strip() or None, name.strip(), db.now(), db.now()))
    person_id = cur.lastrowid
    c.execute("UPDATE unknowns SET resolved=1 WHERE id=?", (unknown_id,))
    db.audit(c, "person.create_from_unknown", actor=user["u"],
             target=str(person_id), detail=name, ip=client_ip(request))
    path = os.path.join(config.abspath("data"), row["image"])
    cmd = c.execute(
        "INSERT INTO commands(kind, payload, created_at) VALUES (?,?,?)",
        ("enroll_files",
         json.dumps({"person_id": person_id, "paths": [path]}), db.now()))
    return RedirectResponse("/admin/register/capture/%d" % cmd.lastrowid,
                            status_code=303)


# --------------------------------------------------------------------- reports

@app.get("/admin/reports", response_class=HTMLResponse)
def reports_page(request: Request, start: str = "", end: str = "",
                 person_id: str = ""):
    require(request, "viewer")
    c = conn()
    end = end or db.today()
    start = start or time.strftime(
        "%Y-%m-%d", time.localtime(time.time() - 29 * 86400))
    pid = int(person_id) if person_id.isdigit() else None
    rows = attendance.range_rows(c, start, end, pid)
    totals = {
        "rows": len(rows),
        "late": sum(1 for r in rows if r["status"] == "late"),
        "half": sum(1 for r in rows if r["half_day"]),
        "hours": round(sum(r["minutes"] or 0 for r in rows) / 60.0, 1),
    }
    return render(request, "reports.html", start=start, end=end, rows=rows,
                  totals=totals, person_id=person_id,
                  people=[dict(r) for r in c.execute(
                      "SELECT id, name FROM people ORDER BY name").fetchall()])


@app.get("/admin/export.csv")
def export_csv(request: Request, start: str = "", end: str = "",
               person_id: str = ""):
    require(request, "viewer")
    c = conn()
    end = end or db.today()
    start = start or time.strftime(
        "%Y-%m-%d", time.localtime(time.time() - 29 * 86400))
    pid = int(person_id) if person_id.isdigit() else None
    rows = attendance.range_rows(c, start, end, pid)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Employee", "Code", "Department", "Check in",
                     "Check out", "Status", "Half day", "Minutes", "Hours",
                     "Corrected", "Note"])
    for r in rows:
        writer.writerow([r["day"], r["name"], r["emp_code"] or "",
                         r["department"] or "", r["check_in"] or "",
                         r["check_out"] or "",
                         STATUS_LABEL.get(r["status"], r["status"]),
                         "yes" if r["half_day"] else "",
                         r["minutes"] or 0,
                         round((r["minutes"] or 0) / 60.0, 2),
                         "yes" if r["corrected"] else "", r["note"] or ""])
    db.audit(c, "report.export", actor=auth.current_user(request)["u"],
             detail="%s..%s" % (start, end), ip=client_ip(request))
    return PlainTextResponse(
        buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition":
                 'attachment; filename="attendance_%s_%s.csv"' % (start, end)})


# ---------------------------------------------------------- health & settings

@app.get("/admin/health", response_class=HTMLResponse)
def health_page(request: Request):
    require(request, "viewer")
    c = conn()
    health = dict(c.execute("SELECT * FROM health WHERE id=1").fetchone() or {})
    return render(request, "health.html", health=health, up=recognizer_up(c),
                  audit=[dict(r) for r in c.execute(
                      "SELECT * FROM audit ORDER BY id DESC LIMIT 40").fetchall()])


@app.get("/admin/settings", response_class=HTMLResponse)
def settings_page(request: Request, saved: str = ""):
    require(request, "admin")
    return render(request, "settings.html", cfg=config.load(force=True),
                  saved=bool(saved))


@app.post("/admin/settings")
def settings_save(request: Request, csrf: str = Form(...),
                  site_name: str = Form(...), threshold: float = Form(...),
                  votes: int = Form(...), shift_start: str = Form(...),
                  shift_end: str = Form(...), grace: int = Form(...),
                  rescan_gap: int = Form(...), min_work: int = Form(...),
                  store_thumbs: str = Form(""),
                  blur_preview: str = Form(""), log_unknowns: str = Form(""),
                  liveness_on: str = Form(""), liveness_threshold: float = Form(...),
                  motion_on: str = Form(""), roi: str = Form("")):
    user = require(request, "admin")
    check_csrf(request, csrf)
    cfg = config.load(force=True)
    cfg["site"]["name"] = site_name.strip()
    cfg["match"]["threshold"] = float(threshold)
    cfg["match"]["votes_required"] = int(votes)
    cfg["attendance"]["shift_start"] = shift_start.strip()
    cfg["attendance"]["shift_end"] = shift_end.strip()
    cfg["attendance"]["grace_minutes"] = int(grace)
    # Duplicate handling. Clamped rather than trusted: a zero rescan gap
    # reintroduces exactly the behaviour these settings exist to prevent.
    cfg["attendance"]["min_rescan_gap_s"] = max(10, min(900, int(rescan_gap)))
    cfg["attendance"]["min_work_minutes"] = max(0, min(720, int(min_work)))
    cfg.setdefault("privacy", {})
    cfg["privacy"]["store_event_thumbs"] = bool(store_thumbs)
    cfg["privacy"]["blur_kiosk_preview"] = bool(blur_preview)
    cfg["privacy"]["log_unknown_faces"] = bool(log_unknowns)
    cfg["liveness"]["enabled"] = bool(liveness_on)
    cfg["liveness"]["threshold"] = float(liveness_threshold)
    cfg["motion"]["enabled"] = bool(motion_on)
    roi = roi.strip()
    if roi:
        try:
            parts = [int(v) for v in roi.replace(" ", "").split(",")]
            cfg["camera"]["roi"] = parts if len(parts) == 4 else None
        except ValueError:
            cfg["camera"]["roi"] = None
    else:
        cfg["camera"]["roi"] = None
    config.save(cfg)
    c = conn()
    db.audit(c, "settings.save", actor=user["u"], ip=client_ip(request),
             detail="threshold=%s votes=%s liveness=%s rescan_gap=%s "
                    "min_work=%s store_thumbs=%s"
                    % (threshold, votes, bool(liveness_on), rescan_gap,
                       min_work, bool(store_thumbs)))
    c.execute("INSERT INTO commands(kind, created_at) VALUES ('reload', ?)",
              (db.now(),))
    return RedirectResponse("/admin/settings?saved=1", status_code=303)


@app.get("/admin/account", response_class=HTMLResponse)
def account_page(request: Request, msg: str = ""):
    user = require(request, "viewer")
    c = conn()
    row = c.execute("SELECT * FROM users WHERE id=?", (user["uid"],)).fetchone()
    return render(request, "account.html", me=dict(row), msg=msg,
                  totp_uri=auth.totp_uri(c, user["uid"]))


@app.post("/admin/account/password")
def change_password(request: Request, new1: str = Form(...), new2: str = Form(...),
                    csrf: str = Form(...)):
    user = require(request, "viewer")
    check_csrf(request, csrf)
    if new1 != new2:
        return RedirectResponse("/admin/account?msg=Passwords+did+not+match",
                                status_code=303)
    if len(new1) < 10:
        return RedirectResponse(
            "/admin/account?msg=Use+at+least+10+characters", status_code=303)
    auth.set_password(conn(), user["uid"], new1, actor=user["u"])
    return RedirectResponse("/admin/account?msg=Password+updated", status_code=303)


@app.post("/admin/account/totp")
def setup_totp(request: Request, csrf: str = Form(...)):
    user = require(request, "admin")
    check_csrf(request, csrf)
    auth.enable_totp(conn(), user["uid"], actor=user["u"])
    return RedirectResponse(
        "/admin/account?msg=Scan+the+key+below+in+your+authenticator+app",
        status_code=303)


# ----------------------------------------------------------------- media files

@app.get("/media/{path:path}")
def media(request: Request, path: str):
    """Serve face thumbnails and unknown snapshots - login required, and the
    path is confined to the data directory."""
    require(request, "viewer")
    base = os.path.realpath(config.abspath("data"))
    target = os.path.realpath(os.path.join(base, path))
    if not target.startswith(base + os.sep) or not os.path.isfile(target):
        raise HTTPException(404, "Not found")
    return FileResponse(target, headers={"Cache-Control": "private, max-age=3600"})

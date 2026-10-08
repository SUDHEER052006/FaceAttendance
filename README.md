# Face Attendance — Raspberry Pi 5

Face-recognition attendance for a single Raspberry Pi 5 with Camera Module 3,
CPU only, designed to run 24/7 with a secured dashboard on a 7" touchscreen.

- **Detection** — YuNet (~0.3 MB, 8–12 ms per 640×480 frame)
- **Recognition** — ArcFace `w600k_mbf` / MobileFaceNet, 512-d embeddings
- **Anti-spoofing** — passive liveness, rejects printed photos and phone replays
- **Accuracy** — IOU tracking with multi-frame voting, not per-frame guessing
- **One person, one record** — three layers of de-duplication, so somebody
  standing at the door produces a single attendance row, not forty
- **Private by default** — no face images stored or displayed unless you
  deliberately turn them on
- **Interface** — light Material-styled dashboard, built for a 7" touchscreen
- **Storage** — SQLite WAL, embeddings not photos, DPDP-compliant erasure

---

## Quick start

```bash
git clone <your-repo-url> face-attendance
cd face-attendance
chmod +x run.sh
./run.sh dev          # first run: foreground, so you can watch the logs
```

`run.sh` installs apt packages, creates the venv, installs Python deps,
downloads both models, creates the database, and prints a generated admin
password **once**. Write it down.

Then open the dashboard:

- On the Pi screen — `http://localhost:8000`
- From your laptop — `http://<pi-ip>:8000`

Once it looks right, install it as a pair of services so it survives reboots:

```bash
./run.sh            # installs + enables fa-recognizer and fa-web, then starts
./run.sh status
./run.sh logs
```

### All commands

| Command | What it does |
|---|---|
| `./run.sh` | Install if needed, then install + start both systemd services |
| `./run.sh dev` | Run both processes in the foreground (Ctrl-C stops) |
| `./run.sh install` | Dependencies and models only, start nothing |
| `./run.sh models` | Re-download the models |
| `./run.sh bench` | Measure real FPS and per-stage timings on this board |
| `./run.sh tune` | Compute `match.threshold` from your own enrolled faces |
| `./run.sh test` | Self-test the attendance and de-duplication logic (no camera or models needed; safe on a live device) |
| `./run.sh harden` | Watchdog, journald size cap, no screen blanking (asks first) |
| `./run.sh kiosk` | Install the fullscreen Chromium autostart for the 7" panel |
| `./run.sh logs` / `stop` / `restart` / `status` | Service control |

---

## How attendance is decided

This is the part that makes or breaks the product, so it is worth
understanding before you deploy.

A doorway camera does not see one neat event per person per day. It sees the
same face dozens of times: while they wait for the door, when they turn their
head, on the way to the canteen, coming back from a smoke, standing in the
corridor talking. Logging all of that is what turns an attendance system into
noise.

**De-duplication happens at three levels, and all three are needed:**

| Level | Where | What it stops |
|---|---|---|
| 1. Per track | `tracker.py` | A resolved track is never embedded again, so one person standing in frame costs a handful of inferences rather than hundreds |
| 2. Per person | `recognizer_service.py` | A short cooldown keyed on person id. Tracks die and respawn constantly — a head turn, a dropped detection, the motion gate going idle — and without this each respawn logged the same person again |
| 3. Per policy | `attendance.py` | The final say. Reports `seen` or `duplicate` for anything that is not a real punch |

**An event row and a face image are written only for a genuine check-in or
check-out.** Everything else leaves no trace beyond `last_seen`, which is why
the activity feed stays readable and the disk does not fill with the same face.

The rules, all driven from `config.yaml`:

```
First recognition of the day                           -> check_in
Seen again within  attendance.min_rescan_gap_s   (90s) -> ignored entirely
Seen again, but <  attendance.min_work_minutes   (90m)
                   after check-in                      -> a sighting, not a punch
Seen again after that                                  -> check_out (latest wins)
```

That third rule is the one people miss. Without it, somebody spotted on their
way to the canteen at 09:25 was recorded as having gone home, and the report
showed a twenty-minute working day. A sighting only becomes a departure once
the person has been at work long enough for leaving to be plausible.

Both numbers are editable on **Settings** and clamped server-side — a zero
rescan gap would reintroduce exactly the behaviour they exist to prevent.

`./run.sh test` asserts all of this against the real code: a person recognised
by 25 consecutive tracks must produce exactly one event row and zero stored
images, and 30 tracks of one stranger must produce one snapshot. Run it after
touching `attendance.py`, `tracker.py`, or `commit()`.

`Status` and `half_day` stay separate columns on purpose: somebody who arrives
at 10:45 and leaves at 13:30 is **both** late and a half day, and payroll needs
to see both.

The day sheet shows a `(4×)` marker next to **Last seen** when a person was
sighted more than once, so you can tell "the camera saw them repeatedly and
collapsed it" from "the camera only saw them once".

---

## The dashboard

Two separate interfaces, both on the same 7" screen. Light Material styling —
Google blue on white, an icon rail rather than a tab strip, 48 px minimum
touch targets, and no emoji (half of them rendered as empty boxes on a Pi with
no emoji font installed).

### 1. Kiosk view — `/`

What is on the panel all day. No login needed to watch it.

- Live camera with boxes: cyan while scanning, green on a match, red on unknown
  (faces can be blurred here — see `privacy.blur_kiosk_preview`)
- Giant clock and date
- Three counters — **On site / Late / Absent**
- Rolling activity feed showing **name, department and an In/Out chip**. No
  face images: this screen needs no login and is usually mounted where the
  whole lobby can read it
- Fullscreen confirmation card when somebody is recorded ("Welcome, Priya" /
  "Goodbye, Priya")
- The on-screen banner also announces spoof rejections, "already recorded"
  for a repeat sighting, and enrollment progress

### 2. Admin dashboard — login or PIN

| Page | What it is for |
|---|---|
| **Overview** `/admin` | **On site now** / present / late / absent tiles, today's activity feed, recognizer status, enrolled counts, and a plain statement of what is and is not being stored |
| **Today** `/admin/today` | Every active person for any chosen date — in, out, **last seen**, sighting count, status, hours. Inline **Fix** panel to correct a punch (reason mandatory, written to the audit log) |
| **Register** `/admin/register` | Name, code, department, phone, how many angles, and a mandatory biometric-consent checkbox → then a live capture screen with a progress bar and real-time coaching ("too dark", "turn your head slightly", "captured left 3/7") |
| **People** `/admin/people` | Searchable roster with template counts and last-seen. Flags anyone enrolled with **no face**. Per-person page shows captured poses, 30-day history, re-capture, enroll-from-photos, mark inactive, and DPDP deletion |
| **Unknown** `/admin/unknowns` | Gallery of faces seen repeatedly but not matched. One click turns a snapshot into an enrolled person, or dismiss it. Auto-purges on the retention schedule |
| **Reports** `/admin/reports` | Any date range, any person or everyone. Totals for records / late / half-days / hours, plus **CSV export** |
| **Health** `/admin/health` | Recognizer up-or-down, CPU temperature, FPS, mode (idle/active), uptime, load, disk free, and the last 40 audit entries |
| **Settings** `/admin/settings` | Site name, **duplicate handling** (rescan gap, earliest possible check-out), **privacy** (store face images, blur the preview, log unknowns), match threshold, votes required, shift start/end/grace, liveness on/off + threshold, motion gating, and the detection ROI box |
| **Account** `/admin/account` | Change password, enable TOTP two-factor |

Three roles: **viewer** (read only), **hr** (register, correct attendance),
**admin** (settings, delete biometric data, 2FA).

**PIN unlock** — the 4-digit `web.admin_pin` works only from the device itself.
A PIN stolen off the wall is useless over the network.

---

## 24/7 checklist

Work through this before you call it deployed. Everything here is a failure
mode that actually kills always-on Pi installs.

- [ ] **Boot from NVMe or a USB SSD, not an SD card.** The single most common
      cause of death. The PCIe slot is free since you are not using an AI HAT.
- [ ] **Active Cooler fitted.** Sustained inference throttles a passively
      cooled Pi 5 at 80 °C and halves your FPS.
- [ ] `./run.sh harden` — hardware watchdog, 200 MB journald cap, no blanking.
- [ ] **DS3231 RTC module** (~₹100). A Pi that reboots with no network has no
      idea what time it is, and wrong attendance timestamps are unforgivable.
- [ ] **Motion gating on** (default). An empty corridor at 3 AM should cost ~2%
      CPU, not 100%.
- [ ] **Set the ROI** to just the doorway on the Settings page. Often 3–4×
      faster for free.
- [ ] `./run.sh bench` and confirm you are getting the FPS you expect.
- [ ] `./run.sh tune` after enrolling real people, and apply the recommendation.
- [ ] Change the admin password and the PIN. Enable 2FA.
- [ ] Decide retention (`retention.unknown_days`) and record consent at every
      enrollment — the Register page enforces the checkbox.
- [ ] Back up `data/attendance.db` somewhere off the Pi, on a schedule.

### Remote access

Do **not** port-forward 8000. Either:

- **Cloudflare Tunnel** — `cloudflared tunnel --url http://localhost:8000`, or
- **Tailscale** — `curl -fsSL https://tailscale.com/install.sh | sh`

Both give you HTTPS and an identity layer without opening anything inbound.
If you do expose it publicly, put Caddy in front for TLS and set
`web.https_only: true` in `config.yaml` so session cookies become secure-only.

---

## Tuning accuracy

The order that actually matters:

1. **Enrollment quality beats everything.** 7 angles, even light, no backlight,
   no mask. The capture screen coaches for this; trust its complaints.
2. **`./run.sh tune`** — never guess the threshold. It builds genuine and
   impostor score distributions from your own faces and recommends a number.
   Re-run it after every batch of enrollments.
3. **`match.votes_required`** — raise it if you ever see a wrong name; lower it
   if recognition feels slow at the door. 4 is a good doorway default.
4. **`detect.min_face_px`** — raise it to stop the system trying to identify
   someone 5 m down the corridor from 30 pixels of face.
5. If a specific person is unreliable, open their page and **re-capture with
   "delete existing templates" ticked**. One bad sample poisons a whole
   identity.

### Adding a liveness model

Out of the box the heuristic engine runs (moiré/FFT banding, saturation
collapse, specular blobs, micro-texture). It will catch a casually held-up
phone or printout, and it is honestly not as good as a trained model.

To upgrade, convert a MiniFASNet checkpoint from the
[Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
project to ONNX and save it as `models/minifasnet.onnx`. It is picked up
automatically on restart — `/admin/health` shows which engine is active.
Expected input is NCHW float, class 1 = real.

### Manual model download

If `./run.sh models` cannot reach GitHub, fetch these by hand into `models/`:

| File | Source |
|---|---|
| `face_detection_yunet_2023mar.onnx` | `opencv/opencv_zoo` → `models/face_detection_yunet/` |
| `w600k_mbf.onnx` | `deepinsight/insightface` releases → `v0.7/buffalo_s.zip`, extract this one file |

---

## How it is put together

Two processes, on purpose. If the dashboard crashes or someone hammers a
report query, **attendance keeps recording**. If the recognizer dies, systemd
restarts it and the dashboard tells you it was down.

```
fa-recognizer.service        cores 0-1      fa-web.service       core 2
┌──────────────────────────────────┐        ┌───────────────────────────┐
│ picamera2: lores 640x480 (detect)│        │ FastAPI + Uvicorn         │
│            main 1536x864 (crop)  │        │ Jinja templates, no build │
│ motion gate -> ROI -> YuNet      │        │ MJPEG from /dev/shm       │
│ IOU tracker -> vote over N frames│        │ auth, reports, CSV        │
│ ArcFace embed (unresolved only)  │        │ writes to `commands`      │
│ cosine match + margin check      │        └───────────┬───────────────┘
│ MiniFASNet liveness on candidates│                    │
│ -> events, attendance, unknowns  │◄───────────────────┘
│ -> /dev/shm/fa_frame.jpg         │   SQLite (WAL) + command table
└──────────────────────────────────┘
```

The recognizer owns the camera, so the dashboard can never grab it. To enroll
somebody the web process writes a row into `commands`; the recognizer picks it
up within a second, captures the samples, and reports progress back through the
same row. That is what the live progress bar is polling.

### Files

```
run.sh                       install + start, everything
config.yaml                  every tunable
app/
  recognizer_service.py      process A: camera, inference, enrollment
  web.py                     process B: dashboard routes
  pipeline.py                detector, aligner, quality gate, embedder, matcher
  liveness.py                anti-spoofing engines
  tracker.py                 IOU tracker + vote accumulation
  attendance.py              shift policy, check-in/out, de-duplication, reports
  auth.py                    argon2, sessions, TOTP, PIN, rate limiting
  db.py                      schema + in-place column migrations
  camera.py                  picamera2 with OpenCV fallback
  templates/                 dashboard UI (7" first); icons.html holds the
                             inlined Material icon set
  static/app.css             light Material theme, one file, no build step
scripts/
  benchmark.py               real FPS on this board
  tune_threshold.py          threshold from your own faces
  selftest.py                proves the de-duplication rules still hold
  systemd/                   the two unit files
```

---

## Troubleshooting

**Camera not found** — `rpicam-hello --list-cameras` should print `imx708`.
If `picamera2` will not import, the venv was built without
`--system-site-packages`; delete `.venv` and `.installed` and re-run.

**Kiosk shows CAMERA OFFLINE** — the recognizer is not publishing frames:
`sudo systemctl status fa-recognizer` and `./run.sh logs`.

**Recognises nobody** — check `/admin/people` for anyone showing **No face**,
then confirm templates exist. A fresh install matches nothing until you enroll.

**Wrong person matched** — raise `match.threshold` and `match.votes_required`,
then run `./run.sh tune`. Also check for a bad enrollment sample.

**Slow / low FPS** — confirm motion gating is on, set an ROI, check CPU
temperature on `/admin/health`, and verify the Active Cooler is running.

**Everyone marked late** — `attendance.shift_start` and `grace_minutes` on the
Settings page, and confirm the clock is right (`timedatectl`). This is what the
RTC module is for.

**Same person logged repeatedly** — raise `attendance.min_rescan_gap_s` on the
Settings page. If it is already high and duplicates still appear, they are
seconds apart in the `events` table; check that only one `fa-recognizer` is
running (`systemctl status fa-recognizer`), because two instances keep separate
cooldown maps and will each log the same person.

**Working days far too short** — `attendance.min_work_minutes` is too low, so a
mid-morning sighting is being read as a departure. Set it to something longer
than the longest plausible gap between arriving and genuinely leaving.

**Check-out never recorded** — the opposite: `min_work_minutes` is longer than
the actual shift, or the camera cannot see people on their way out. A single
doorway camera sees both directions, so the last sighting of the day is the
departure; confirm people pass within range when leaving.

---

## Privacy and compliance

Biometric data is sensitive personal data under India's DPDP Act 2023.

- Face **templates** are stored, not photos. A template is a 512-number
  vector; you cannot reconstruct a face from it.
- **No face image is captured with an attendance event by default.**
  `privacy.store_event_thumbs` is off, so a normal day's attendance adds rows
  to the database and nothing to the image store.
- **The activity feed never shows a captured face**, on any screen, whatever
  `store_event_thumbs` is set to. `/api/live` feeds the kiosk, which needs no
  login, so that payload carries names and initials only. If you do turn
  storage on, the image is reachable only from that person's own page, behind
  authentication.
- `privacy.blur_kiosk_preview` blurs faces in the live camera preview while
  keeping the tracking boxes and names, so an operator can still see the
  system is working.
- Enrollment images are kept only to show you what was captured, and only on
  that person's page.
- Unknown snapshots are de-duplicated by embedding similarity as well as by
  time, so one stranger at the door produces one snapshot to review rather
  than a folder full of the same face.
- Consent is recorded per person at enrollment and timestamped; the Register
  page will not proceed without it.
- Deletion destroys templates and face images and removes the person from the
  index. Attendance history is retained for headcount but the biometrics are
  gone and unrecoverable.
- Unknown-visitor snapshots auto-purge on a schedule you set.
- Every enrollment, deletion, attendance correction, settings change, export
  and login attempt is written to an append-only audit log, visible on
  `/admin/health`.

Keep `data/` out of git — `.gitignore` already does this. If the device can be
physically stolen, encrypt the partition with LUKS; the database holds
biometric templates.

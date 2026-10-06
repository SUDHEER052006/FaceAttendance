# Face Attendance — Raspberry Pi 5

Face-recognition attendance for a single Raspberry Pi 5 with Camera Module 3,
CPU only, designed to run 24/7 with a secured dashboard on a 7" touchscreen.

- **Detection** — YuNet (~0.3 MB, 8–12 ms per 640×480 frame)
- **Recognition** — ArcFace `w600k_mbf` / MobileFaceNet, 512-d embeddings
- **Anti-spoofing** — passive liveness, rejects printed photos and phone replays
- **Accuracy** — IOU tracking with multi-frame voting, not per-frame guessing
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
| `./run.sh harden` | Watchdog, journald size cap, no screen blanking (asks first) |
| `./run.sh kiosk` | Install the fullscreen Chromium autostart for the 7" panel |
| `./run.sh logs` / `stop` / `restart` / `status` | Service control |

---

## The dashboard

Two separate interfaces, both on the same 7" screen.

### 1. Kiosk view — `/`

What is on the panel all day. No login needed to watch it.

- Live camera with boxes: cyan while scanning, green on a match, red on unknown
- Giant clock and date
- Three counters — **In / Late / Out**
- Rolling feed of the last recognitions with face thumbnails
- Fullscreen green greeting card when somebody is recognised ("Welcome, Priya")
- The on-screen banner also announces spoof rejections and enrollment progress

### 2. Admin dashboard — login or PIN

| Page | What it is for |
|---|---|
| **Overview** `/admin` | Present / absent / late / turnout tiles, recognizer status, enrolled counts, latest recognitions with scores |
| **Today** `/admin/today` | Every active person for any chosen date — in, out, status, hours. Inline **Fix** panel to correct a punch (reason mandatory, written to the audit log) |
| **Register** `/admin/register` | Name, code, department, phone, how many angles, and a mandatory biometric-consent checkbox → then a live capture screen with a progress bar and real-time coaching ("too dark", "turn your head slightly", "captured left 3/7") |
| **People** `/admin/people` | Searchable roster with template counts and last-seen. Flags anyone enrolled with **no face**. Per-person page shows captured poses, 30-day history, re-capture, enroll-from-photos, mark inactive, and DPDP deletion |
| **Unknown** `/admin/unknowns` | Gallery of faces seen repeatedly but not matched. One click turns a snapshot into an enrolled person, or dismiss it. Auto-purges on the retention schedule |
| **Reports** `/admin/reports` | Any date range, any person or everyone. Totals for records / late / half-days / hours, plus **CSV export** |
| **Health** `/admin/health` | Recognizer up-or-down, CPU temperature, FPS, mode (idle/active), uptime, load, disk free, and the last 40 audit entries |
| **Settings** `/admin/settings` | Site name, match threshold, votes required, shift start/end/grace, liveness on/off + threshold, motion gating, and the detection ROI box |
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
  attendance.py              shift policy, check-in/out, reports
  auth.py                    argon2, sessions, TOTP, PIN, rate limiting
  db.py                      schema
  camera.py                  picamera2 with OpenCV fallback
  templates/ static/         dashboard UI (7" first)
scripts/
  benchmark.py               real FPS on this board
  tune_threshold.py          threshold from your own faces
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

---

## Privacy and compliance

Biometric data is sensitive personal data under India's DPDP Act 2023.

- Face **templates** are stored, not photos, wherever possible. Enrollment
  images are kept only to show you what was captured.
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

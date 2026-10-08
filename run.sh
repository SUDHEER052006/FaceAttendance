#!/usr/bin/env bash
# =============================================================================
#  Face Attendance - one script to install everything and start it.
#
#    ./run.sh              install if needed, then install+start both services
#    ./run.sh dev          run in the foreground (Ctrl-C to stop) - use this
#                          the first time so you can watch the logs
#    ./run.sh install      dependencies + models only, do not start
#    ./run.sh models       (re)download the models
#    ./run.sh stop         stop both services
#    ./run.sh logs         follow both service logs
#    ./run.sh status       show what is running
#    ./run.sh bench        measure real FPS on this board
#    ./run.sh tune         compute a match threshold from your enrolled faces
#    ./run.sh test         self-test the attendance + de-duplication logic
#    ./run.sh harden       24/7 hardening: watchdog, log caps, swap (asks first)
#    ./run.sh kiosk        install the fullscreen browser autostart
#
#  Safe to re-run at any time. Nothing here is destructive.
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
STAMP="$ROOT/.installed"
SERVICE_USER="${SUDO_USER:-$(id -un)}"

C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_0=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$C_OK"   "$C_0" "$*"; }
warn() { printf '%s==>%s %s\n' "$C_WARN" "$C_0" "$*"; }
die()  { printf '%s==>%s %s\n' "$C_ERR"  "$C_0" "$*" >&2; exit 1; }
note() { printf '%s    %s%s\n' "$C_DIM" "$*" "$C_0"; }

trap 'die "failed at line $LINENO. Scroll up for the actual error."' ERR

need_sudo() {
  if [ "$(id -u)" -eq 0 ]; then SUDO=""; else
    command -v sudo >/dev/null || die "sudo not found and not running as root"
    SUDO="sudo"
  fi
}

# ---------------------------------------------------------------- environment

detect_board() {
  BOARD="unknown"
  if [ -r /proc/device-tree/model ]; then
    BOARD="$(tr -d '\0' < /proc/device-tree/model)"
  fi
  IS_PI=0
  case "$BOARD" in *"Raspberry Pi"*) IS_PI=1 ;; esac
}

# ------------------------------------------------------------------- packages

apt_deps() {
  need_sudo
  say "installing system packages"
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  local pkgs=(
    python3 python3-venv python3-dev python3-pip
    build-essential pkg-config
    curl unzip ca-certificates
    libatlas-base-dev libopenblas0
    libgl1 libglib2.0-0
  )
  if [ "$IS_PI" -eq 1 ]; then
    # picamera2 comes from apt, never pip - that is why the venv below is
    # created with --system-site-packages.
    pkgs+=(python3-picamera2 libcamera-apps python3-libcamera python3-kms++)
  fi
  $SUDO apt-get install -y -qq "${pkgs[@]}" || warn "some packages failed; continuing"
}

make_venv() {
  if [ ! -x "$PY" ]; then
    say "creating virtualenv (with system site packages, for picamera2)"
    python3 -m venv --system-site-packages "$VENV"
  fi
  say "installing Python packages"
  "$PIP" install --quiet --upgrade pip wheel
  "$PIP" install --quiet -r requirements.txt
}

# --------------------------------------------------------------------- models

MODEL_DIR="$ROOT/models"
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
BUFFALO_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"

fetch() {
  local url="$1" out="$2"
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

download_models() {
  mkdir -p "$MODEL_DIR" "$MODEL_DIR/_tmp"

  if [ ! -s "$MODEL_DIR/face_detection_yunet_2023mar.onnx" ]; then
    say "downloading YuNet face detector (~230 KB)"
    fetch "$YUNET_URL" "$MODEL_DIR/face_detection_yunet_2023mar.onnx" \
      || die "YuNet download failed. See README > Manual model download."
  else
    note "YuNet already present"
  fi

  if [ ! -s "$MODEL_DIR/w600k_mbf.onnx" ]; then
    say "downloading ArcFace buffalo_s (~16 MB)"
    if fetch "$BUFFALO_URL" "$MODEL_DIR/_tmp/buffalo_s.zip"; then
      unzip -o -q "$MODEL_DIR/_tmp/buffalo_s.zip" -d "$MODEL_DIR/_tmp"
      local found
      found="$(find "$MODEL_DIR/_tmp" -name 'w600k_mbf.onnx' -print -quit)"
      [ -n "$found" ] || die "w600k_mbf.onnx not inside the archive. See README > Manual model download."
      cp "$found" "$MODEL_DIR/w600k_mbf.onnx"
      rm -rf "$MODEL_DIR/_tmp"
      say "ArcFace ready"
    else
      die "ArcFace download failed. See README > Manual model download."
    fi
  else
    note "ArcFace already present"
  fi

  if [ ! -s "$MODEL_DIR/minifasnet.onnx" ]; then
    warn "no liveness model (models/minifasnet.onnx) - the heuristic engine"
    note "will be used. See README > Adding a liveness model."
  fi

  # Tiny placeholder shown by the dashboard when the recognizer is down.
  if [ ! -s "$ROOT/app/static/offline.jpg" ]; then
    "$PY" - <<'PYEOF' || true
import cv2, numpy as np, os
# Light, to match the dashboard - a black panel reads as a dead screen.
img = np.full((480, 640, 3), 241, dtype=np.uint8)
cv2.rectangle(img, (0, 0), (639, 479), (218, 214, 211), 2)
cv2.putText(img, "Camera offline", (150, 245), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (104, 99, 95), 2, cv2.LINE_AA)
cv2.putText(img, "systemctl status fa-recognizer", (138, 290),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 134, 128), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join("app", "static", "offline.jpg"), img)
PYEOF
  fi
}

# --------------------------------------------------------------------- install

requirements_hash() {
  if command -v sha256sum >/dev/null; then sha256sum requirements.txt | cut -d" " -f1
  else shasum -a 256 requirements.txt | cut -d" " -f1; fi
}

do_install() {
  detect_board
  say "board: $BOARD"
  if [ "$IS_PI" -eq 0 ]; then
    warn "not a Raspberry Pi - installing anyway for development."
    note "set camera.source: opencv in config.yaml to use a USB webcam."
  fi

  local want have=""
  want="$(requirements_hash)"
  [ -f "$STAMP" ] && have="$(cat "$STAMP")"

  if [ "$want" != "$have" ] || [ ! -x "$PY" ]; then
    apt_deps
    make_venv
    echo "$want" > "$STAMP"
  else
    note "dependencies already current (delete .installed to force a reinstall)"
    [ -x "$PY" ] || make_venv
  fi

  download_models

  say "initialising database"
  "$PY" -m app.bootstrap || warn "bootstrap reported a problem (see above)"

  if [ "$IS_PI" -eq 1 ]; then
    if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
      note "tip: enable I2C and fit a DS3231 RTC so timestamps survive a"
      note "     reboot with no network. See README > 24/7 checklist."
    fi
  fi
  say "install complete"
}

# ----------------------------------------------------------------- run modes

run_dev() {
  say "starting in the foreground. Ctrl-C stops both."
  note "dashboard:  http://$(hostname -I 2>/dev/null | awk '{print $1}'):$(awk '/^  port:/{print $2}' config.yaml | head -1)"
  "$PY" -m app.recognizer_service &
  local rec=$!
  "$PY" -m uvicorn app.web:app \
      --host "$(awk '/^  host:/{print $2}' config.yaml | head -1)" \
      --port "$(awk '/^  port:/{print $2}' config.yaml | head -1)" \
      --workers 1 --log-level warning &
  local web=$!
  trap 'kill $rec $web 2>/dev/null || true' INT TERM
  wait -n $rec $web || true
  kill $rec $web 2>/dev/null || true
  wait 2>/dev/null || true
  say "stopped"
}

install_services() {
  need_sudo
  say "installing systemd units (user: $SERVICE_USER)"
  local port host
  port="$(awk '/^  port:/{print $2}' config.yaml | head -1)"
  host="$(awk '/^  host:/{print $2}' config.yaml | head -1)"

  for unit in fa-recognizer fa-web; do
    sed -e "s#@ROOT@#$ROOT#g" \
        -e "s#@USER@#$SERVICE_USER#g" \
        -e "s#@PORT@#$port#g" \
        -e "s#@HOST@#$host#g" \
        "scripts/systemd/$unit.service" | $SUDO tee "/etc/systemd/system/$unit.service" >/dev/null
  done

  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now fa-recognizer fa-web
  sleep 2
  show_status
  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo
  say "running. Dashboard:"
  note "  on the Pi screen:  http://localhost:$port"
  note "  from your laptop:  http://${ip:-<pi-ip>}:$port"
  note "logs: ./run.sh logs"
}

show_status() {
  need_sudo
  for unit in fa-recognizer fa-web; do
    if $SUDO systemctl is-active --quiet "$unit"; then
      printf '  %s%-16s running%s\n' "$C_OK" "$unit" "$C_0"
    else
      printf '  %s%-16s NOT running%s\n' "$C_ERR" "$unit" "$C_0"
    fi
  done
}

# ------------------------------------------------------------------ hardening

do_harden() {
  need_sudo
  warn "this changes system settings on this Pi:"
  note "  - enables the hardware watchdog (auto-reboot on a hard hang)"
  note "  - caps journald at 200 MB so logs cannot fill the disk"
  note "  - disables screen blanking so the 7 inch panel stays on"
  read -r -p "Continue? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { say "skipped"; return 0; }

  $SUDO mkdir -p /etc/systemd/journald.conf.d
  printf '[Journal]\nSystemMaxUse=200M\nSystemMaxFileSize=20M\n' \
    | $SUDO tee /etc/systemd/journald.conf.d/fa-cap.conf >/dev/null
  $SUDO systemctl restart systemd-journald || true

  if ! grep -q "RuntimeWatchdogSec" /etc/systemd/system.conf; then
    printf '\nRuntimeWatchdogSec=15\nRebootWatchdogSec=2min\n' \
      | $SUDO tee -a /etc/systemd/system.conf >/dev/null
  fi
  if [ -f /boot/firmware/config.txt ] && ! grep -q "dtparam=watchdog=on" /boot/firmware/config.txt; then
    printf 'dtparam=watchdog=on\n' | $SUDO tee -a /boot/firmware/config.txt >/dev/null
    warn "watchdog enabled in config.txt - takes effect after a reboot"
  fi

  if command -v xset >/dev/null; then
    $SUDO raspi-config nonint do_blanking 1 2>/dev/null || true
  fi

  say "hardening applied. Reboot when convenient."
  warn "still on an SD card? Move to NVMe or a USB SSD before going 24/7."
  note "SD wear-out is the most common way an always-on Pi dies."
}

do_kiosk() {
  say "installing fullscreen browser autostart for the 7 inch screen"
  local port autostart
  port="$(awk '/^  port:/{print $2}' config.yaml | head -1)"
  autostart="$HOME/.config/autostart"
  mkdir -p "$autostart"
  cat > "$autostart/fa-kiosk.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Face Attendance Kiosk
Exec=chromium-browser --kiosk --noerrdialogs --disable-infobars --incognito \\
  --check-for-update-interval=31536000 --app=http://localhost:$port
X-GNOME-Autostart-enabled=true
EOF
  say "done - it will launch on the next desktop login"
  note "to test now: chromium-browser --kiosk --app=http://localhost:$port"
}

# ----------------------------------------------------------------------- main

case "${1:-up}" in
  up)
    do_install
    detect_board
    if [ "$IS_PI" -eq 1 ] && command -v systemctl >/dev/null; then
      install_services
    else
      warn "no systemd - running in the foreground instead"
      run_dev
    fi
    ;;
  dev)      do_install; run_dev ;;
  install)  do_install ;;
  models)   [ -x "$PY" ] || { do_install; exit 0; }; download_models ;;
  stop)     need_sudo; $SUDO systemctl stop fa-recognizer fa-web; say "stopped" ;;
  restart)  need_sudo; $SUDO systemctl restart fa-recognizer fa-web; show_status ;;
  status)   show_status ;;
  logs)     need_sudo; $SUDO journalctl -u fa-recognizer -u fa-web -f -n 60 ;;
  bench)    "$PY" scripts/benchmark.py ;;
  tune)     "$PY" scripts/tune_threshold.py ;;
  test)     "$PY" scripts/selftest.py ;;
  harden)   do_harden ;;
  kiosk)    do_kiosk ;;
  *)        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
esac

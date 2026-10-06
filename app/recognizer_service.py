"""Process A: owns the camera, does all inference, writes to the database.

Runs as its own systemd unit, pinned to cores 0-1. If the dashboard crashes,
this keeps recording attendance. If this crashes, systemd restarts it and the
dashboard will show you that it died.

Loop shape:

    grab lores frame
      -> motion gate (skip everything if the room is empty)
      -> YuNet detect
      -> IOU tracker associates boxes to tracks
      -> for unresolved tracks only: align from the main stream, quality gate,
         ArcFace embed, cosine match
      -> push a vote; once N frames agree, run liveness, write the event and
         the attendance row
      -> publish an annotated JPEG to /dev/shm for the dashboard to stream
"""
import json
import os
import shutil
import signal
import sys
import time

import cv2
import numpy as np

from . import attendance, camera, config, db, liveness, pipeline
from .tracker import Tracker

RUNNING = True
START_TS = time.time()


def _stop(signum, frame):
    global RUNNING
    RUNNING = False
    print("[recognizer] stop signal %s" % signum)


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return round(int(fh.read().strip()) / 1000.0, 1)
    except Exception:
        return None


def publish_frame(path, bgr, quality=72):
    """Atomic write so the web process never reads a half-finished JPEG."""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            fh.write(buf.tobytes())
        os.replace(tmp, path)
    except OSError:
        pass


def save_jpg(path, bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


class Service:
    def __init__(self):
        self.conn = db.init()
        self.cam = camera.open_camera()
        self.detector = pipeline.Detector(self.cam.lores_size)
        self.embedder = pipeline.Embedder()
        self.matcher = pipeline.Matcher()
        self.liveness = liveness.build()
        self.tracker = Tracker()

        n = self.matcher.load(self.conn)
        print("[recognizer] loaded %d templates for %d people"
              % (n, len(self.matcher.names)))

        self.frame_path = config.g("paths.frame", "/dev/shm/fa_frame.jpg")
        self.scale = self.cam.main_size[0] / float(self.cam.lores_size[0])
        self.prev_gray = None
        self.last_motion = 0.0
        self.fps = 0.0
        self.state = "idle"
        self.last_health = 0.0
        self.last_cmd_poll = 0.0
        self.last_purge = 0.0
        self.banner = None          # (text, expires_at) overlaid on the preview

    # ---------------------------------------------------------------- helpers

    def roi_slice(self, lores):
        roi = config.g("camera.roi")
        if not roi:
            return lores, (0, 0)
        x, y, w, h = [int(v) for v in roi]
        h_img, w_img = lores.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(16, min(w, w_img - x))
        h = max(16, min(h, h_img - y))
        return lores[y:y + h, x:x + w], (x, y)

    def crop_face(self, main, pts, offset):
        """Map lores landmarks into the main stream and align from real pixels."""
        ox, oy = offset
        pts_main = (pts + np.array([ox, oy], dtype=np.float32)) * self.scale
        return pipeline.align(main, pts_main), pts_main

    def say(self, text, seconds=3.0):
        self.banner = (text, time.time() + seconds)

    # ------------------------------------------------------------------ events

    def commit(self, track, person_id, score, crop, votes):
        live_score = None
        if self.liveness is not None:
            live_score = self.liveness.score(crop)
            if live_score < float(config.g("liveness.threshold", 0.55)):
                track.resolved = True
                track.liveness = live_score
                self.say("SPOOF REJECTED (%.2f)" % live_score, 4.0)
                db.audit(self.conn, "liveness.reject",
                         actor="recognizer", target=str(person_id),
                         detail="score=%.3f" % live_score)
                print("[recognizer] spoof rejected for person %s (%.3f)"
                      % (person_id, live_score))
                return

        ts = db.now()
        thumb_rel = os.path.join(
            "thumbs", ts[:10], "%s_%s.jpg" % (ts[11:].replace(":", ""), person_id))
        save_jpg(os.path.join(config.abspath("data"), thumb_rel), crop)

        self.conn.execute(
            "INSERT INTO events(person_id, ts, score, liveness, votes, thumb) "
            "VALUES (?,?,?,?,?,?)",
            (person_id, ts, score, live_score, votes, thumb_rel))

        kind, _row = attendance.record(self.conn, person_id, ts)
        track.resolved = True
        track.person_id = person_id
        track.liveness = live_score
        track.thumb = thumb_rel

        name = self.matcher.names.get(person_id, "#%s" % person_id)
        if kind == "check_in":
            self.say("Welcome, %s" % name, 4.0)
        elif kind == "check_out":
            self.say("Bye, %s" % name, 4.0)
        print("[recognizer] %s %s score=%.3f votes=%d live=%s"
              % (kind, name, score, votes, live_score))

    def log_unknown(self, track, crop, score):
        ts = db.now()
        rel = os.path.join("unknowns", ts[:10],
                           "%s.jpg" % ts[11:].replace(":", ""))
        save_jpg(os.path.join(config.abspath("data"), rel), crop)
        self.conn.execute(
            "INSERT INTO unknowns(ts, image, score) VALUES (?,?,?)",
            (ts, rel, score))
        track.resolved = True
        self.say("Unknown face logged", 3.0)

    # --------------------------------------------------------------- main loop

    def step(self):
        t0 = time.time()
        lores, main = self.cam.read()
        frame, offset = self.roi_slice(lores)

        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        if config.g("motion.enabled", True):
            score = camera.motion_score(self.prev_gray, gray)
            self.prev_gray = gray
            if score >= float(config.g("motion.threshold", 2.2)):
                self.last_motion = t0
            active = (t0 - self.last_motion) < float(
                config.g("motion.cooldown_s", 6))
        else:
            self.prev_gray = gray
            active = True
        self.state = "active" if active else "idle"

        faces = self.detector.detect(frame) if active else []
        boxes = [f["box"] for f in faces]
        pairs = self.tracker.update(boxes)

        # Only embed tracks that still need an identity. A person standing in
        # frame for 30 s costs a handful of embeddings, not hundreds.
        todo = [(t, faces[i]) for t, i in pairs
                if i is not None and not t.resolved]

        crops, metas = [], []
        for track, face in todo:
            crop, _ = self.crop_face(main, face["pts"], offset)
            if crop is None:
                continue
            ok, _blur, reason = pipeline.quality(crop, face["pts"])
            if not ok:
                track.push_vote(None, 0.0)
                continue
            crops.append(crop)
            metas.append((track, crop, reason))

        if crops:
            embeddings = self.embedder.embed(crops)
            for (track, crop, _r), (pid, score) in zip(
                    metas, self.matcher.query(embeddings)):
                track.push_vote(pid, score)
                verdict_pid, mean_score, votes = track.verdict()
                if verdict_pid is not None:
                    self.commit(track, verdict_pid, mean_score, crop, votes)
                elif (len(track.votes) >= config.g("match.vote_window", 10)
                      and track.miss_ratio() > 0.8):
                    self.log_unknown(track, crop, score)

        self.draw(lores, faces, pairs, offset)

        # Adaptive frame rate: idle at 5 fps, jump to 15 on motion.
        target = float(config.g("camera.active_fps", 15) if active
                       else config.g("camera.idle_fps", 5))
        elapsed = time.time() - t0
        self.fps = 1.0 / max(elapsed, 1e-6)
        sleep = (1.0 / target) - elapsed
        if sleep > 0:
            time.sleep(sleep)

    def draw(self, lores, faces, pairs, offset):
        img = lores.copy()
        ox, oy = offset
        roi = config.g("camera.roi")
        if roi:
            x, y, w, h = [int(v) for v in roi]
            cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)

        by_index = {i: t for t, i in pairs if i is not None}
        for i, face in enumerate(faces):
            x, y, w, h = [int(v) for v in face["box"]]
            x, y = x + ox, y + oy
            track = by_index.get(i)
            if track is not None and track.person_id:
                colour = (80, 220, 100)
                label = self.matcher.names.get(track.person_id, "?")
            elif track is not None and track.resolved:
                colour = (70, 90, 240)
                label = "unknown"
            else:
                colour = (0, 190, 240)
                _p, _s, votes = track.verdict() if track else (None, 0, 0)
                label = "scanning %d" % votes
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
            cv2.rectangle(img, (x, y + h), (x + w, y + h + 20), colour, -1)
            cv2.putText(img, label[:18], (x + 4, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1,
                        cv2.LINE_AA)

        bar = "%s  %.1f fps  %s" % (
            time.strftime("%H:%M:%S"), self.fps, self.state)
        cv2.putText(img, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)

        if self.banner and time.time() < self.banner[1]:
            text = self.banner[0]
            h = img.shape[0]
            cv2.rectangle(img, (0, h - 44), (img.shape[1], h), (25, 25, 30), -1)
            cv2.putText(img, text, (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (120, 230, 160), 2, cv2.LINE_AA)
        elif self.banner:
            self.banner = None

        publish_frame(self.frame_path, img)

    # ---------------------------------------------------------------- commands

    def poll_commands(self):
        row = self.conn.execute(
            "SELECT * FROM commands WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            return
        self.conn.execute(
            "UPDATE commands SET status='running', message=? WHERE id=?",
            ("starting", row["id"]))
        try:
            payload = json.loads(row["payload"] or "{}")
            if row["kind"] == "enroll_live":
                self.enroll_live(row["id"], payload)
            elif row["kind"] == "enroll_files":
                self.enroll_files(row["id"], payload)
            elif row["kind"] == "reload":
                n = self.matcher.load(self.conn)
                self.tracker.reset()
                self.finish(row["id"], "reloaded %d templates" % n)
            else:
                self.fail(row["id"], "unknown command %s" % row["kind"])
        except Exception as exc:  # keep the service alive no matter what
            self.fail(row["id"], "%s: %s" % (type(exc).__name__, exc))

    def finish(self, cmd_id, message):
        self.conn.execute(
            "UPDATE commands SET status='done', progress=100, message=? WHERE id=?",
            (message, cmd_id))

    def fail(self, cmd_id, message):
        self.conn.execute(
            "UPDATE commands SET status='error', message=? WHERE id=?",
            (message, cmd_id))
        print("[recognizer] command %s failed: %s" % (cmd_id, message))

    def progress(self, cmd_id, pct, message):
        self.conn.execute(
            "UPDATE commands SET progress=?, message=? WHERE id=?",
            (int(pct), message, cmd_id))

    def enroll_live(self, cmd_id, payload):
        """Capture N good, *varied* samples straight from the camera.

        Variety matters more than count: seven near-identical frontal shots
        are worth less than four taken at different angles, so samples that
        are too similar to one already captured are rejected.
        """
        person_id = int(payload["person_id"])
        want = int(payload.get("samples", 7))
        timeout = float(payload.get("timeout", 75))
        name = payload.get("name", "#%s" % person_id)

        kept, poses = [], []
        deadline = time.time() + timeout
        self.say("Enrolling %s - look at the camera" % name, 5.0)

        while time.time() < deadline and len(kept) < want and RUNNING:
            lores, main = self.cam.read()
            frame, offset = self.roi_slice(lores)
            faces = self.detector.detect(frame)
            if len(faces) != 1:
                msg = "no face" if not faces else "more than one face in frame"
                self.progress(cmd_id, 100 * len(kept) / want,
                              "%s (%d/%d)" % (msg, len(kept), want))
                self.draw(lores, faces, [], offset)
                time.sleep(0.08)
                continue

            face = faces[0]
            crop, _ = self.crop_face(main, face["pts"], offset)
            if crop is None:
                continue
            ok, blur, reason = pipeline.quality(crop, face["pts"])
            if not ok:
                self.progress(cmd_id, 100 * len(kept) / want,
                              "%s (%d/%d)" % (reason, len(kept), want))
                self.draw(lores, faces, [], offset)
                time.sleep(0.08)
                continue

            emb = self.embedder.embed([crop])[0]
            if kept and float(np.max(np.stack(kept) @ emb)) > 0.985:
                self.progress(cmd_id, 100 * len(kept) / want,
                              "turn your head slightly (%d/%d)"
                              % (len(kept), want))
                self.draw(lores, faces, [], offset)
                time.sleep(0.12)
                continue

            pose = pipeline.pose_label(face["pts"])
            kept.append(emb)
            poses.append(pose)
            self.conn.execute(
                "INSERT INTO templates(person_id, embedding, quality, pose, "
                "created_at) VALUES (?,?,?,?,?)",
                (person_id, emb.astype(np.float32).tobytes(), blur, pose,
                 db.now()))
            save_jpg(os.path.join(config.abspath(config.g("paths.faces")),
                                  str(person_id), "%s_%d.jpg" % (pose, len(kept))),
                     crop)
            self.progress(cmd_id, 100 * len(kept) / want,
                          "captured %s (%d/%d)" % (pose, len(kept), want))
            self.draw(lores, faces, [], offset)
            time.sleep(0.25)

        if not kept:
            self.fail(cmd_id, "no usable frames captured - check lighting")
            self.say("Enrollment failed", 4.0)
            return

        n = self.matcher.load(self.conn)
        self.tracker.reset()
        self.finish(cmd_id, "enrolled %d samples (%s); index now %d templates"
                    % (len(kept), ", ".join(sorted(set(poses))), n))
        self.say("Enrolled %s" % name, 4.0)

    def enroll_files(self, cmd_id, payload):
        """Enroll from uploaded photos. Same quality gate as live capture."""
        person_id = int(payload["person_id"])
        paths = payload.get("paths", [])
        kept = 0
        for idx, rel in enumerate(paths):
            path = rel if os.path.isabs(rel) else config.abspath(rel)
            img = cv2.imread(path)
            if img is None:
                continue
            faces = self.detector.detect(img)
            if len(faces) != 1:
                continue
            face = faces[0]
            crop = pipeline.align(img, face["pts"])
            if crop is None:
                continue
            ok, blur, _reason = pipeline.quality(crop, face["pts"])
            if not ok:
                continue
            emb = self.embedder.embed([crop])[0]
            pose = pipeline.pose_label(face["pts"])
            self.conn.execute(
                "INSERT INTO templates(person_id, embedding, quality, pose, "
                "created_at) VALUES (?,?,?,?,?)",
                (person_id, emb.astype(np.float32).tobytes(), blur, pose,
                 db.now()))
            kept += 1
            self.progress(cmd_id, 100 * (idx + 1) / max(len(paths), 1),
                          "%d usable of %d" % (kept, idx + 1))
        if kept == 0:
            self.fail(cmd_id, "no usable faces in the uploaded images")
            return
        n = self.matcher.load(self.conn)
        self.finish(cmd_id, "enrolled %d photos; index now %d templates" % (kept, n))

    # ----------------------------------------------------------- housekeeping

    def heartbeat(self):
        try:
            usage = shutil.disk_usage(config.abspath("data"))
            free = usage.free
        except Exception:
            free = None
        try:
            load1 = os.getloadavg()[0]
        except Exception:
            load1 = None
        self.conn.execute(
            "UPDATE health SET ts=?, cpu_temp=?, fps=?, state=?, faces=?, "
            "uptime_s=?, disk_free=?, load1=?, detail=? WHERE id=1",
            (db.now(), cpu_temp(), round(self.fps, 1), self.state,
             len(self.tracker.tracks), int(time.time() - START_TS), free,
             load1,
             "liveness=%s templates=%d" % (
                 getattr(self.liveness, "name", "off"), len(self.matcher.owners))))

    def purge(self):
        """Retention. Unknown snapshots and old events do not live forever -
        this is a DPDP requirement, not a nice-to-have."""
        udays = int(config.g("retention.unknown_days", 14))
        edays = int(config.g("retention.event_days", 400))
        cutoff_u = time.strftime(
            "%Y-%m-%d", time.localtime(time.time() - udays * 86400))
        cutoff_e = time.strftime(
            "%Y-%m-%d", time.localtime(time.time() - edays * 86400))

        stale = self.conn.execute(
            "SELECT id, image FROM unknowns WHERE ts < ?", (cutoff_u,)).fetchall()
        for row in stale:
            path = os.path.join(config.abspath("data"), row["image"])
            try:
                os.remove(path)
            except OSError:
                pass
        self.conn.execute("DELETE FROM unknowns WHERE ts < ?", (cutoff_u,))
        self.conn.execute("DELETE FROM events WHERE ts < ?", (cutoff_e,))
        if stale:
            print("[recognizer] purged %d unknown snapshots" % len(stale))

    def run(self):
        print("[recognizer] running - camera %s, lores %s, main %s"
              % (type(self.cam).__name__, self.cam.lores_size, self.cam.main_size))
        while RUNNING:
            try:
                self.step()
            except camera.CameraError as exc:
                print("[recognizer] camera error: %s - retrying in 2s" % exc)
                time.sleep(2)
                continue
            except Exception as exc:
                print("[recognizer] loop error: %s: %s" % (type(exc).__name__, exc))
                time.sleep(0.5)

            now = time.time()
            if now - self.last_cmd_poll > 1.0:
                self.last_cmd_poll = now
                try:
                    self.poll_commands()
                except Exception as exc:
                    print("[recognizer] command poll failed: %s" % exc)
            if now - self.last_health > 10.0:
                self.last_health = now
                try:
                    self.heartbeat()
                except Exception:
                    pass
            if now - self.last_purge > 3600.0:
                self.last_purge = now
                try:
                    self.purge()
                except Exception:
                    pass

        self.cam.close()
        self.conn.close()
        print("[recognizer] stopped cleanly")


def main():
    try:
        Service().run()
    except FileNotFoundError as exc:
        print("[recognizer] %s" % exc)
        sys.exit(2)


if __name__ == "__main__":
    main()

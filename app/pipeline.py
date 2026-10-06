"""Detect -> align -> quality-gate -> embed -> match.

Model choices are deliberate for a CPU-only Pi 5:
  YuNet      ~0.3 MB, ~8-12 ms per 640x480 frame
  ArcFace    w600k_mbf (MobileFaceNet, 512-d), ~25-40 ms per face

Everything sits behind the Detector/Embedder classes so swapping in a Hailo
backend later is a one-file change, not a rewrite.
"""
import os

import cv2
import numpy as np

from . import config

# Canonical 5-point template ArcFace was trained against, at 112x112.
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


class Detector:
    """OpenCV's bundled YuNet. Returns boxes + 5 landmarks + score."""

    def __init__(self, size):
        model = config.abspath(config.g("detect.model"))
        if not os.path.exists(model):
            raise FileNotFoundError(
                "detector model missing: %s - run ./run.sh models" % model)
        self.net = cv2.FaceDetectorYN.create(
            model=model,
            config="",
            input_size=size,
            score_threshold=float(config.g("detect.score_threshold", 0.75)),
            nms_threshold=float(config.g("detect.nms_threshold", 0.3)),
            top_k=int(config.g("detect.top_k", 20)),
        )
        self.size = size

    def set_size(self, size):
        if size != self.size:
            self.net.setInputSize(size)
            self.size = size

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        self.set_size((w, h))
        _, faces = self.net.detect(bgr)
        out = []
        if faces is None:
            return out
        min_px = float(config.g("detect.min_face_px", 70))
        for f in faces:
            x, y, fw, fh = f[0:4]
            if fw < min_px or fh < min_px:
                continue
            pts = np.array(f[4:14], dtype=np.float32).reshape(5, 2)
            # YuNet emits (right-eye, left-eye, nose, right-mouth, left-mouth).
            # ArcFace's template is ordered by increasing x, so normalise by x
            # rather than trusting the naming - this survives mirrored input.
            if pts[0][0] > pts[1][0]:
                pts[[0, 1]] = pts[[1, 0]]
            if pts[3][0] > pts[4][0]:
                pts[[3, 4]] = pts[[4, 3]]
            out.append({
                "box": (float(x), float(y), float(fw), float(fh)),
                "pts": pts,
                "score": float(f[14]),
            })
        return out


def align(bgr, pts, size=112):
    """Similarity-transform the face onto ArcFace's canonical landmarks."""
    dst = ARCFACE_DST * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(pts, dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def quality(crop, pts):
    """Return (ok, blur_score, reason).

    Garbage in is the single biggest cause of wrong matches, so both
    enrollment and live inference run through this same gate.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())

    eye_mid = (pts[0] + pts[1]) / 2.0
    eye_dist = float(np.linalg.norm(pts[1] - pts[0])) or 1.0
    yaw_ratio = float(abs(pts[2][0] - eye_mid[0]) / eye_dist)

    if blur < float(config.g("quality.min_blur", 55.0)):
        return False, blur, "too blurry - hold still or add light"
    if bright < float(config.g("quality.min_brightness", 45)):
        return False, blur, "too dark"
    if bright > float(config.g("quality.max_brightness", 215)):
        return False, blur, "overexposed - move out of direct light"
    if yaw_ratio > float(config.g("quality.max_yaw_ratio", 0.38)):
        return False, blur, "turn to face the camera"
    return True, blur, "ok"


def pose_label(pts):
    """Rough pose bucket, used to label enrollment samples."""
    eye_mid = (pts[0] + pts[1]) / 2.0
    eye_dist = float(np.linalg.norm(pts[1] - pts[0])) or 1.0
    yaw = float((pts[2][0] - eye_mid[0]) / eye_dist)
    mouth_mid = (pts[3] + pts[4]) / 2.0
    pitch = float((pts[2][1] - (eye_mid[1] + mouth_mid[1]) / 2.0) / eye_dist)
    if yaw < -0.15:
        return "left"
    if yaw > 0.15:
        return "right"
    if pitch < -0.10:
        return "up"
    if pitch > 0.22:
        return "down"
    return "front"


class Embedder:
    """ArcFace via ONNX Runtime. 512-d, L2-normalised output."""

    def __init__(self):
        import onnxruntime as ort

        model = config.abspath(config.g("embed.model"))
        if not os.path.exists(model):
            raise FileNotFoundError(
                "embedding model missing: %s - run ./run.sh models" % model)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(config.g("embed.threads", 2))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else 112

    def _prep(self, crop):
        if crop.shape[0] != self.size:
            crop = cv2.resize(crop, (self.size, self.size))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        return np.transpose(rgb, (2, 0, 1))

    def embed(self, crops):
        """crops: list of aligned BGR 112x112. Returns (N, 512) normalised."""
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch = np.stack([self._prep(c) for c in crops]).astype(np.float32)
        out = self.sess.run(None, {self.input_name: batch})[0]
        out = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class Matcher:
    """Brute-force cosine match against the template matrix.

    For anything under ~5000 people a single NumPy dot product is measured in
    microseconds. Do not add a vector database here.
    """

    def __init__(self):
        self.matrix = np.zeros((0, 512), dtype=np.float32)
        self.owners = []        # row index -> person_id
        self.names = {}         # person_id -> name

    def load(self, conn):
        rows = conn.execute(
            "SELECT t.person_id, t.embedding FROM templates t "
            "JOIN people p ON p.id = t.person_id WHERE p.active = 1"
        ).fetchall()
        if rows:
            self.matrix = np.stack(
                [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            self.owners = [r["person_id"] for r in rows]
        else:
            self.matrix = np.zeros((0, 512), dtype=np.float32)
            self.owners = []
        self.names = {
            r["id"]: r["name"]
            for r in conn.execute("SELECT id, name FROM people").fetchall()
        }
        return len(self.owners)

    def query(self, embeddings):
        """Returns one (person_id|None, score) per input embedding.

        A match must clear the absolute threshold AND beat the best competing
        identity by match.margin. The margin check is what stops look-alikes
        and siblings from flip-flopping between each other.
        """
        thresh = float(config.g("match.threshold", 0.40))
        margin = float(config.g("match.margin", 0.05))
        if self.matrix.shape[0] == 0 or embeddings.shape[0] == 0:
            return [(None, 0.0)] * int(embeddings.shape[0])

        sims = embeddings @ self.matrix.T          # (N, n_templates)
        owners = np.asarray(self.owners)
        results = []
        for row in sims:
            # Collapse per-template scores down to the best score per person.
            best_by_person = {}
            for pid, score in zip(owners, row):
                pid = int(pid)
                if score > best_by_person.get(pid, -1.0):
                    best_by_person[pid] = float(score)
            ranked = sorted(best_by_person.items(), key=lambda kv: -kv[1])
            top_pid, top_score = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else -1.0
            if top_score >= thresh and (top_score - runner) >= margin:
                results.append((top_pid, top_score))
            else:
                results.append((None, top_score))
        return results

"""Passive anti-spoofing.

Two engines:

1. OnnxLiveness - if models/minifasnet.onnx exists it is used. This is the one
   you want in production. MiniFASNet is ~1.9 MB and runs in single-digit ms.
   See README ("Adding a liveness model") for how to drop one in.

2. HeuristicLiveness - the always-available fallback. It scores classical
   replay-attack cues:
     * moire / high-frequency banding  -> phone or monitor replay
     * colour saturation collapse       -> printed photo
     * specular highlight blob          -> glossy screen or photo paper
     * flat texture (low LBP variance)  -> paper
   It is genuinely useful against a casually held-up phone or printout, and it
   is NOT a substitute for a trained model. Treat it as a speed bump until you
   add the ONNX model, and keep the threshold conservative.

Either way, liveness is only evaluated on frames that already produced a
candidate identity match - never on every frame. It is cheap, but not free.
"""
import os

import cv2
import numpy as np

from . import config


class HeuristicLiveness:
    name = "heuristic"

    def score(self, crop):
        """Return 0.0 (likely spoof) .. 1.0 (likely live)."""
        img = cv2.resize(crop, (112, 112))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- moire / screen banding -------------------------------------
        # A re-photographed display leaves periodic energy in the mid-high
        # frequency band that real skin does not have.
        f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        mag = np.log1p(np.abs(f))
        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        yy, xx = np.ogrid[:mag.shape[0], :mag.shape[1]]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mid = mag[(radius > 18) & (radius < 42)].mean()
        low = mag[radius <= 18].mean() or 1.0
        moire_ratio = float(mid / low)
        moire_ok = 1.0 - np.clip((moire_ratio - 0.42) / 0.25, 0.0, 1.0)

        # --- colour richness --------------------------------------------
        # Prints and cheap screens lose saturation spread across the face.
        sat_std = float(hsv[:, :, 1].std())
        colour_ok = np.clip(sat_std / 28.0, 0.0, 1.0)

        # --- specular blob ----------------------------------------------
        # Glossy paper and glass produce one bright saturated-white region.
        v = hsv[:, :, 2]
        spec = float(((v > 245) & (hsv[:, :, 1] < 30)).mean())
        spec_ok = 1.0 - np.clip(spec / 0.06, 0.0, 1.0)

        # --- micro-texture ----------------------------------------------
        # Real skin has fine local variance; paper is comparatively flat.
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        tex_ok = np.clip(lap / 180.0, 0.0, 1.0)

        return float(np.clip(
            0.34 * moire_ok + 0.26 * colour_ok + 0.20 * spec_ok + 0.20 * tex_ok,
            0.0, 1.0))


class OnnxLiveness:
    name = "onnx"

    def __init__(self, model_path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else 80

    def score(self, crop):
        img = cv2.resize(crop, (self.size, self.size))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch = np.transpose(rgb, (2, 0, 1))[None, ...]
        out = np.asarray(self.sess.run(None, {self.input_name: batch})[0]).ravel()
        if out.size == 1:
            return float(1.0 / (1.0 + np.exp(-out[0])))
        exp = np.exp(out - out.max())
        probs = exp / exp.sum()
        # MiniFASNet convention: class 1 = real face, 0/2 = print/replay attack.
        return float(probs[1] if probs.size > 1 else probs[0])


def build():
    """Pick the best available engine. Logged once at startup."""
    if not config.g("liveness.enabled", True):
        return None
    model = config.abspath(config.g("liveness.model", ""))
    if model and os.path.exists(model):
        try:
            engine = OnnxLiveness(model)
            print("[liveness] using ONNX model: %s" % model)
            return engine
        except Exception as exc:
            print("[liveness] ONNX model failed to load (%s); using heuristic" % exc)
    else:
        print("[liveness] no ONNX model found; using heuristic engine "
              "(see README: Adding a liveness model)")
    return HeuristicLiveness()

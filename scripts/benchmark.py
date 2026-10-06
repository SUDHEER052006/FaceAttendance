"""Measure what this board actually does. Run it before you tune anything.

    ./run.sh bench

Reports per-stage timings and the frame rate you can realistically expect with
1 and 2 faces in view. Numbers from a thermally-throttled Pi are meaningless,
so it prints CPU temperature before and after.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from app import config, pipeline
from app.recognizer_service import cpu_temp


def timeit(fn, n):
    fn()                      # warm up, exclude first-call graph setup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    lores = tuple(config.g("camera.lores", [640, 480]))
    print("=" * 62)
    print(" Face Attendance benchmark")
    print("=" * 62)
    print(" CPU temp before : %s C" % cpu_temp())
    print(" OpenCV          : %s" % cv2.__version__)
    print(" detect resolution: %dx%d" % lores)
    print("")

    detector = pipeline.Detector(lores)
    embedder = pipeline.Embedder()
    print(" embed threads   : %s" % config.g("embed.threads", 2))
    print("")

    # A synthetic frame with structure - a flat grey image detects nothing and
    # would make YuNet look unrealistically fast.
    rng = np.random.default_rng(7)
    frame = rng.integers(40, 210, (lores[1], lores[0], 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    crop = cv2.resize(frame[:112, :112], (112, 112))

    det_ms = timeit(lambda: detector.detect(frame), 30)
    emb1_ms = timeit(lambda: embedder.embed([crop]), 20)
    emb2_ms = timeit(lambda: embedder.embed([crop, crop]), 20)

    print(" YuNet detect    : %6.1f ms/frame" % det_ms)
    print(" ArcFace 1 face  : %6.1f ms" % emb1_ms)
    print(" ArcFace 2 faces : %6.1f ms  (%.1f ms/face)" % (emb2_ms, emb2_ms / 2))
    print("")

    for faces, embed_ms in ((0, 0.0), (1, emb1_ms), (2, emb2_ms)):
        total = det_ms + embed_ms
        print(" %d face(s): %5.1f ms/frame  ->  %4.1f fps"
              % (faces, total, 1000.0 / total))

    print("")
    print(" NOTE: embedding only runs for tracks without an identity yet, so")
    print("       steady-state FPS sits close to the 0-face number once people")
    print("       in frame have been recognised.")
    print("")
    print(" CPU temp after  : %s C" % cpu_temp())
    temp = cpu_temp()
    if temp and temp > 78:
        print(" WARNING: above 78 C - the Pi is throttling. Fit the Active")
        print("          Cooler or these numbers are your floor, not your ceiling.")
    print("=" * 62)


if __name__ == "__main__":
    main()

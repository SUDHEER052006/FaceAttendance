"""Camera abstraction.

Two streams on purpose:
  lores  - small, fed to the detector every frame (the hot path)
  main   - larger, faces are cropped from here so the embedder sees real pixels

Falls back to a plain OpenCV capture so you can develop on a laptop.
"""
import cv2
import numpy as np

from . import config


class CameraError(RuntimeError):
    pass


class Picam:
    def __init__(self):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # pragma: no cover - Pi only
            raise CameraError(
                "picamera2 not importable. Install with: sudo apt install -y "
                "python3-picamera2, and make sure the venv was created with "
                "--system-site-packages (run.sh does this)."
            ) from exc

        lores = tuple(config.g("camera.lores", [640, 480]))
        main = tuple(config.g("camera.main", [1536, 864]))

        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(
            main={"size": main, "format": "RGB888"},
            lores={"size": lores, "format": "YUV420"},
            buffer_count=4,
        )
        transform = {}
        if config.g("camera.hflip", False):
            transform["hflip"] = 1
        if config.g("camera.vflip", False):
            transform["vflip"] = 1
        if transform:
            from libcamera import Transform
            cfg["transform"] = Transform(**transform)
        self.cam.configure(cfg)
        self.cam.start()
        self.lores_size = lores
        self.main_size = main

    def read(self):
        main, lores = self.cam.capture_arrays(["main", "lores"])
        # picamera2's "RGB888" hands back BGR byte order in the numpy array.
        # That is what OpenCV wants, so no conversion here.
        lores_bgr = cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)
        return lores_bgr, main

    def close(self):
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass


class OpenCVCam:
    """Laptop / USB-webcam fallback. Downscales a single stream into two."""

    def __init__(self):
        idx = config.g("camera.opencv_index", 0)
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise CameraError("could not open OpenCV capture index %s" % idx)
        self.main_size = tuple(config.g("camera.main", [1536, 864]))
        self.lores_size = tuple(config.g("camera.lores", [640, 480]))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.main_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.main_size[1])
        self.hflip = config.g("camera.hflip", False)
        self.vflip = config.g("camera.vflip", False)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise CameraError("capture read failed")
        if self.hflip:
            frame = cv2.flip(frame, 1)
        if self.vflip:
            frame = cv2.flip(frame, 0)
        main = cv2.resize(frame, self.main_size, interpolation=cv2.INTER_AREA)
        lores = cv2.resize(main, self.lores_size, interpolation=cv2.INTER_AREA)
        return lores, main

    def close(self):
        self.cap.release()


def open_camera():
    source = config.g("camera.source", "picamera2")
    if source == "opencv":
        return OpenCVCam()
    try:
        return Picam()
    except CameraError as exc:
        print("[camera] picamera2 unavailable (%s); falling back to OpenCV" % exc)
        return OpenCVCam()


def motion_score(prev_gray, gray):
    """Mean absolute difference. Deliberately dumb and ~0.3ms."""
    if prev_gray is None or prev_gray.shape != gray.shape:
        return 999.0
    return float(np.mean(cv2.absdiff(prev_gray, gray)))

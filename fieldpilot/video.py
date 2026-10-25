"""Video helpers: H.264 writer with fallback, and the chase-cam | robot-cam recorder."""
from pathlib import Path

import cv2
import numpy as np

CAM_W, CAM_H = 640, 480


def open_writer(path: Path, fps: float, size):
    """Prefer H.264 ('avc1', plays in QuickTime/browsers); fall back to MPEG-4 Part 2 ('mp4v')."""
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            print(f"video: {path.name} codec={codec}")
            return w
    raise RuntimeError(f"Could not open a video writer for {path}")


class Recorder:
    """Call once per control step; writes every `every`-th frame (default 20 Hz / 2 = 10 fps)."""

    def __init__(self, path: Path | None, sim, every: int = 2):
        self.sim, self.every, self.n, self.writer = sim, every, 0, None
        if path is not None:
            fps = round(1.0 / (sim.control_dt * every))
            self.writer = open_writer(path, fps, (2 * CAM_W, CAM_H))

    def __call__(self, label: str):
        self.n += 1
        if self.writer is None or self.n % self.every:
            return
        frame = np.hstack([self.sim.render("chase", CAM_W, CAM_H), self.sim.render("robot_cam", CAM_W, CAM_H)])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"t={self.sim.time:5.1f}s  {label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()

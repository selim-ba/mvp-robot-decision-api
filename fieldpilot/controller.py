"""Deterministic go-to-pose controller (baseline, and Phase 1 movement testing).

Two phases per target: (1) turn toward the target point and drive to it, (2) rotate in place
to the target yaw. Hysteresis keeps it from flip-flopping between phases near the goal.
"""
import numpy as np

from .sim import wrap_angle


class GoToPoseController:
    def __init__(self, pos_tol=0.10, yaw_tol=np.deg2rad(5), k_lin=1.0, k_ang=2.5, max_v=0.6, max_w=1.5):
        self.pos_tol, self.yaw_tol = pos_tol, yaw_tol
        self.k_lin, self.k_ang = k_lin, k_ang
        self.max_v, self.max_w = max_v, max_w
        self.reset()

    def reset(self):
        self._at_position = False

    def __call__(self, pose, target):
        """pose, target: (x, y, yaw). target yaw may be None. Returns (v, w, done)."""
        x, y, yaw = pose
        tx, ty, tyaw = target
        dist = float(np.hypot(tx - x, ty - y))

        if self._at_position and dist > 2 * self.pos_tol:
            self._at_position = False          # pushed away: drive again
        if not self._at_position and dist <= self.pos_tol:
            self._at_position = True

        if not self._at_position:
            err = wrap_angle(np.arctan2(ty - y, tx - x) - yaw)
            w = float(np.clip(self.k_ang * err, -self.max_w, self.max_w))
            v = 0.0 if abs(err) > np.pi / 3 else min(self.k_lin * dist, self.max_v) * np.cos(err)
            return float(v), w, False

        if tyaw is None:
            return 0.0, 0.0, True
        err = wrap_angle(tyaw - yaw)
        if abs(err) < self.yaw_tol:
            return 0.0, 0.0, True
        return 0.0, float(np.clip(self.k_ang * err, -self.max_w, self.max_w)), False

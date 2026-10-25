"""Goal-conditioned navigation environment (Gymnasium) for PPO.

Task: drive the robot to a target pose (x, y, yaw) and stop there — position within 0.15 m,
heading within 10°, forward speed < 0.1 m/s and yaw rate < 0.3 rad/s.
Inspection waypoints need the heading so the camera faces the panel; PPO handles it end to end.

Observation (15, float32), all in roughly [-1, 1]:
    target in robot frame (dx, dy) / 5, distance / 5,
    sin/cos of bearing to target (robot frame), sin/cos of heading error,
    forward speed, yaw rate / 3, previous action (2),
    robot x / 6, y / 2, sin/cos yaw   (absolute pose lets the policy learn the static panel layout)
Action (2): left/right wheel speed in [-1, 1], scaled to the actuator limit (12 rad/s).
Reward: potential-based progress (distance, plus heading within 1 m, plus low speed within 0.5 m), small time and
action-smoothness penalties, +10 on success, -10 on collision / leaving the area / tipping over.

Training never calls the Decisions API and never uses vegetation labels.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .sim import FieldSim, wrap_angle
from .waypoints import inspection_waypoints

SUCCESS_POS = 0.15                 # m
SUCCESS_YAW = np.deg2rad(10)       # rad
SUCCESS_V = 0.10                   # m/s  — must actually stop at the waypoint (steady camera capture)
SUCCESS_W = 0.30                   # rad/s


class FieldNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    SAMPLE_X, SAMPLE_Y = (-5.5, 5.5), (-1.5, 1.5)   # where random starts/targets are drawn
    BOUNDS_X, BOUNDS_Y = (-6.5, 6.5), (-2.5, 1.9)   # leaving this box = failure (y>1.9 is too close to panels)

    def __init__(self, render_mode=None, max_steps: int = 600, waypoint_prob: float = 0.5):
        self.sim = FieldSim()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.waypoint_prob = waypoint_prob
        self.waypoint_poses = [wp.pose for pair in inspection_waypoints(self.sim.panels()).values()
                               for wp in pair.values()]
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (15,), np.float32)
        self.target = np.zeros(3)
        self.prev_action = np.zeros(2)
        self.steps = 0

    # ------------------------------------------------------------- helpers
    def _random_pose(self):
        r = self.np_random
        return (float(r.uniform(*self.SAMPLE_X)), float(r.uniform(*self.SAMPLE_Y)), float(r.uniform(-np.pi, np.pi)))

    def _sample_target(self):
        x, y, _ = self.sim.pose()
        for _ in range(100):
            if self.np_random.random() < self.waypoint_prob:
                t = self.waypoint_poses[self.np_random.integers(len(self.waypoint_poses))]
            else:
                t = self._random_pose()
            if np.hypot(t[0] - x, t[1] - y) >= 0.5:
                return t
        return t

    def errors(self):
        """(distance, bearing in robot frame, heading error, target dx/dy in robot frame)."""
        x, y, yaw = self.sim.pose()
        dx, dy = self.target[0] - x, self.target[1] - y
        c, s = np.cos(yaw), np.sin(yaw)
        bx, by = c * dx + s * dy, -s * dx + c * dy
        return float(np.hypot(dx, dy)), float(np.arctan2(by, bx)), wrap_angle(self.target[2] - yaw), bx, by

    def _potential(self, dist, yaw_err):
        v, w = self.sim.velocity()
        heading = 0.5 * abs(yaw_err) * max(0.0, 1.0 - dist / 1.0)
        slow_down = 0.3 * (abs(v) + 0.2 * abs(w)) * max(0.0, 1.0 - dist / 0.5)
        return -dist - heading - slow_down

    def _obs(self):
        dist, bearing, yaw_err, bx, by = self.errors()
        x, y, yaw = self.sim.pose()
        v, w = self.sim.velocity()
        return np.array([bx / 5, by / 5, min(dist, 10.0) / 5, np.sin(bearing), np.cos(bearing),
                         np.sin(yaw_err), np.cos(yaw_err), v, w / 3, *self.prev_action,
                         x / 6, y / 2, np.sin(yaw), np.cos(yaw)], dtype=np.float32)

    def at_target(self) -> bool:
        """Same success test as step(): close, aligned and stopped."""
        dist, _, yaw_err, _, _ = self.errors()
        v, w = self.sim.velocity()
        return bool(dist < SUCCESS_POS and abs(yaw_err) < SUCCESS_YAW and abs(v) < SUCCESS_V and abs(w) < SUCCESS_W)

    def _failed(self):
        x, y, _ = self.sim.pose()
        out = not (self.BOUNDS_X[0] < x < self.BOUNDS_X[1] and self.BOUNDS_Y[0] < y < self.BOUNDS_Y[1])
        return bool(out or self.sim.in_collision() or not self.sim.is_upright())

    def _info(self, dist, yaw_err, success=False, failed=False):
        v, w = self.sim.velocity()
        return {"is_success": success, "failed": failed, "dist": dist, "yaw_err_deg": float(np.degrees(abs(yaw_err))),
                "speed": abs(v), "yaw_rate": abs(w)}

    # --------------------------------------------------------------- API
    def reset(self, *, seed=None, options=None):
        """options: start=(x,y,yaw), target=(x,y,yaw), keep_robot=True (chain targets without teleporting)."""
        super().reset(seed=seed)
        options = options or {}
        if not options.get("keep_robot"):
            self.sim.reset(*(options.get("start") or self._random_pose()))
        self.target = np.array(options["target"] if "target" in options else self._sample_target(), dtype=float)
        self.prev_action = np.zeros(2)
        self.steps = 0
        dist, _, yaw_err, _, _ = self.errors()
        self._phi = self._potential(dist, yaw_err)
        return self._obs(), self._info(dist, yaw_err)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        m = self.sim.max_wheel_speed
        self.sim.step(a[0] * m, a[1] * m)
        self.steps += 1

        dist, _, yaw_err, _, _ = self.errors()
        phi = self._potential(dist, yaw_err)
        reward = 5.0 * (phi - self._phi) - 0.005 - 0.02 * float(np.sum((a - self.prev_action) ** 2))
        self._phi = phi
        self.prev_action = a

        success = self.at_target()
        failed = self._failed()
        if success:
            reward += 10.0
        elif failed:
            reward -= 10.0
        terminated = bool(success or failed)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        return self._obs(), float(reward), terminated, truncated, self._info(dist, yaw_err, bool(success), bool(failed and not success))

    def render(self):
        if self.render_mode == "rgb_array":
            return self.sim.render("chase", 640, 480)
        return None

    def close(self):
        self.sim.close()

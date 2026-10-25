"""PPO vs rule-based "ghost race" helpers (used by scripts/compare.py and scripts/demo.py).

Each Racer drives the same 7-stop patrol in its own FieldNavEnv. The PPO racer's sim is the one
shown; the baseline racer's pose is mirrored onto the visual-only ghost robot.
"""
import numpy as np

from .controller import GoToPoseController
from .nav_env import FieldNavEnv
from .waypoints import END_POSE, START_POSE, inspection_waypoints


def patrol_targets(sim):
    wps = inspection_waypoints(sim.panels())
    return [pair[k].pose for pair in wps.values() for k in ("normal", "close")] + [END_POSE]


class Racer:
    def __init__(self, name, model=None, verbose=True):
        self.name, self.model, self.verbose = name, model, verbose
        self.env = FieldNavEnv()
        self.sim = self.env.sim
        self.targets = patrol_targets(self.sim)
        self.ctrl = GoToPoseController(max_v=0.9)   # same baseline settings as eval_nav.py
        self.i, self.finish_time = 0, None
        self.env.reset(options={"start": START_POSE, "target": self.targets[0]})

    def tick(self):
        """Wheel speeds for the next control step."""
        if self.i >= len(self.targets):
            return 0.0, 0.0
        if self.env.at_target():
            self.i += 1
            if self.verbose:
                print(f"  {self.name:<10} reached stop {self.i}/{len(self.targets)} at t={self.sim.time:5.1f}s")
            if self.i == len(self.targets):
                self.finish_time = self.sim.time
                return 0.0, 0.0
            self.env.reset(options={"keep_robot": True, "target": self.targets[self.i]})
            self.ctrl.reset()
        if self.model is not None:
            a = np.clip(self.model.predict(self.env._obs(), deterministic=True)[0], -1, 1)
            self.env.prev_action = a
            return tuple(a * self.sim.max_wheel_speed)
        v, w, _ = self.ctrl(self.sim.pose(), self.targets[self.i])
        return self.sim.twist_to_wheels(v, w)

    def status(self):
        if self.finish_time:
            return f"{self.name}: finished in {self.finish_time:.1f}s"
        return f"{self.name}: stop {min(self.i + 1, len(self.targets))}/{len(self.targets)}"


def make_racers(model, verbose=True):
    return Racer("PPO", model, verbose), Racer("Rule-based", None, verbose)

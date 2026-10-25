"""Interactive viewer: watch the robot patrol the panels.

    python scripts/view.py            # deterministic baseline controller
    python scripts/view.py --ppo      # trained PPO policy (models/ppo_nav.zip) in control

Uses MuJoCo's managed viewer with a control callback, so it runs with plain `python` on macOS
(no `mjpython` needed). Close the window to exit. Backspace in the viewer restarts the patrol.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fieldpilot.controller import GoToPoseController  # noqa: E402
from fieldpilot.nav_env import FieldNavEnv  # noqa: E402
from fieldpilot.waypoints import END_POSE, START_POSE, inspection_waypoints  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", action="store_true", help="drive with the trained PPO policy")
    args = ap.parse_args()

    env = FieldNavEnv()          # owns the sim; also provides PPO observations and the success test
    sim = env.sim
    wps = inspection_waypoints(sim.panels())
    targets = [pair[k].pose for pair in wps.values() for k in ("normal", "close")] + [END_POSE]
    env.reset(options={"start": START_POSE, "target": targets[0]})

    if args.ppo:
        from stable_baselines3 import PPO
        model = PPO.load(ROOT / "models" / "ppo_nav.zip", device="cpu")
    ctrl = GoToPoseController()
    state = {"i": 0, "next_t": 0.0}
    print(f"Driving with: {'PPO policy' if args.ppo else 'deterministic baseline'}")

    def control(m, d):
        """Called by MuJoCo before every physics step; runs the policy at 20 Hz."""
        if d.time < state["next_t"] - 2 * sim.control_dt:      # viewer was reset
            state.update(i=0, next_t=0.0)
            env.reset(options={"keep_robot": True, "target": targets[0]})
            ctrl.reset()
        if d.time < state["next_t"]:
            return
        state["next_t"] = d.time + sim.control_dt
        left = right = 0.0
        if state["i"] < len(targets):
            if env.at_target():
                print(f"  reached target {state['i'] + 1}/{len(targets)} at t={d.time:.1f}s")
                state["i"] += 1
                ctrl.reset()
                if state["i"] < len(targets):
                    env.reset(options={"keep_robot": True, "target": targets[state["i"]]})
            if state["i"] < len(targets):
                if args.ppo:
                    a = np.clip(model.predict(env._obs(), deterministic=True)[0], -1, 1)
                    env.prev_action = a
                    left, right = a * sim.max_wheel_speed
                else:
                    v, w, _ = ctrl(sim.pose(), targets[state["i"]])
                    left, right = sim.twist_to_wheels(v, w)
        d.ctrl[:] = [left, right]      # actuator order in field.xml: left_motor, right_motor

    mujoco.set_mjcb_control(control)
    try:
        mujoco.viewer.launch(sim.model, sim.data)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()

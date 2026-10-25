"""PPO vs rule-based controller, side by side in one MuJoCo scene ("ghost race").

The orange robot is driven by PPO (real physics). A second, hidden simulation runs the
deterministic baseline on the same patrol; its robot pose is mirrored onto the blue,
semi-transparent ghost robot each control step. Both start together at the same pose.

    python scripts/compare.py            # live MuJoCo viewer
    python scripts/compare.py --video    # record outputs/phase2/compare.mp4 (overview camera)
"""
import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from fieldpilot.race import make_racers  # noqa: E402
from fieldpilot.video import open_writer  # noqa: E402


def run_video(ppo, rule):
    out = ROOT / "outputs" / "phase2" / "compare.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    writer = open_writer(out, 20, (w, h))
    sim = ppo.sim
    done_steps = 0
    for _ in range(2000):                         # 100 s cap
        sim.step(*ppo.tick())
        rule.sim.step(*rule.tick())
        sim.set_ghost(rule.sim.base_qpos())
        mujoco.mj_forward(sim.model, sim.data)    # refresh ghost placement before rendering
        frame = cv2.cvtColor(sim.render("overview", w, h), cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"t={sim.time:5.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(frame, f"orange = {ppo.status()}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 110, 230), 2, cv2.LINE_AA)
        cv2.putText(frame, f"blue   = {rule.status()}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 110, 30), 2, cv2.LINE_AA)
        writer.write(frame)
        if ppo.finish_time and rule.finish_time:
            done_steps += 1
            if done_steps > 30:                   # hold the final frame 1.5 s
                break
    writer.release()
    print(f"\nSaved {out}")


def run_viewer(ppo, rule):
    sim = ppo.sim
    state = {"next_t": 0.0}

    def control(m, d):
        if d.time < state["next_t"]:
            return
        state["next_t"] = d.time + sim.control_dt
        d.ctrl[:] = ppo.tick()                     # PPO drives the real (orange) robot
        rule.sim.step(*rule.tick())                # baseline advances in its own sim ...
        sim.set_ghost(rule.sim.base_qpos())        # ... and is shown as the blue ghost

    mujoco.set_mjcb_control(control)
    try:
        mujoco.viewer.launch(sim.model, sim.data)
    finally:
        mujoco.set_mjcb_control(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="record outputs/phase2/compare.mp4 instead of opening the viewer")
    args = ap.parse_args()
    ppo, rule = make_racers(PPO.load(ROOT / "models" / "ppo_nav.zip", device="cpu"))
    print("orange = PPO (real physics)   blue ghost = rule-based controller (mirrored from a 2nd sim)")
    if args.video:
        run_video(ppo, rule)
    else:
        run_viewer(ppo, rule)
    for r in (ppo, rule):
        print(f"{r.name:<10} finished patrol in {r.finish_time:.1f}s" if r.finish_time else f"{r.name:<10} did not finish")


if __name__ == "__main__":
    main()

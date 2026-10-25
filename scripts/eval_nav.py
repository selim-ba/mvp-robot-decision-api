"""Evaluate the PPO policy against the deterministic baseline.

1. 20 held-out episodes (seeds 10000+, never used in training): random start, target is either an
   inspection waypoint or a random pose. PPO and baseline get the exact same start/target.
2. Full patrol with PPO in control: start -> normal + close-up waypoint of each panel -> end.

Usage: python scripts/eval_nav.py [--episodes 20] [--no-video]
Output: outputs/phase2/{results.json, results.md, ppo_patrol.mp4}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from fieldpilot.controller import GoToPoseController  # noqa: E402
from fieldpilot.nav_env import FieldNavEnv  # noqa: E402
from fieldpilot.video import Recorder  # noqa: E402
from fieldpilot.waypoints import END_POSE, START_POSE, inspection_waypoints  # noqa: E402

MODEL_PATH = ROOT / "models" / "ppo_nav.zip"
OUT = ROOT / "outputs" / "phase2"
HELDOUT_SEED0 = 10_000


def ppo_policy(model):
    def act(obs, env):
        return model.predict(obs, deterministic=True)[0]
    return act


def baseline_policy():
    # Top speed 0.9 m/s (PPO can reach ~0.96 m/s at full wheel speed) so the comparison is about strategy.
    ctrl = GoToPoseController(max_v=0.9)

    def act(obs, env):
        if env.steps == 0:
            ctrl.reset()
        v, w, _ = ctrl(env.sim.pose(), tuple(env.target))
        left, right = env.sim.twist_to_wheels(v, w)
        return np.array([left, right], dtype=np.float32) / env.sim.max_wheel_speed
    return act


def run_leg(env, policy, rec=None, label=""):
    """Run until success, failure or timeout from the env's current state."""
    obs = env._obs()
    path, (px, py, _) = 0.0, env.sim.pose()
    while True:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        x, y, _ = env.sim.pose()
        path, px, py = path + float(np.hypot(x - px, y - py)), x, y
        if rec:
            rec(label)
        if term or trunc:
            break
    return {"success": bool(info["is_success"]), "failed": bool(info["failed"]), "timeout": bool(trunc),
            "time_s": round(env.steps * env.sim.control_dt, 2), "path_m": round(path, 2),
            "final_dist": round(info["dist"], 3), "final_yaw_err_deg": round(info["yaw_err_deg"], 1),
            "final_speed": round(info["speed"], 3)}


def summarize(rows):
    ok = [r for r in rows if r["success"]]
    return {"success_rate": len(ok) / len(rows), "successes": len(ok), "episodes": len(rows),
            "failures(collision/out)": sum(r["failed"] for r in rows), "timeouts": sum(r["timeout"] for r in rows),
            "mean_time_s(success)": round(float(np.mean([r["time_s"] for r in ok])), 2) if ok else None,
            "mean_path_m(success)": round(float(np.mean([r["path_m"] for r in ok])), 2) if ok else None}


def patrol(env, policy, rec=None):
    wps = inspection_waypoints(env.sim.panels())
    legs = [(wp.name, wp.pose) for pair in wps.values() for wp in pair.values()] + [("end", END_POSE)]
    out = []
    for i, (name, target) in enumerate(legs):
        opts = {"start": START_POSE, "target": target} if i == 0 else {"keep_robot": True, "target": target}
        env.reset(options=opts)
        res = run_leg(env, policy, rec, f"PPO -> {name}")
        out.append({"leg": name, **res})
        for _ in range(10):                       # brake and hold 0.5 s at the waypoint (camera capture point)
            env.sim.step(0.0, 0.0)
            if rec:
                rec(f"PPO at {name}")
        if not res["success"]:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = PPO.load(MODEL_PATH, device="cpu")
    env = FieldNavEnv()
    policies = {"ppo": ppo_policy(model), "baseline": baseline_policy()}

    episodes = {k: [] for k in policies}
    for i in range(args.episodes):
        seed = HELDOUT_SEED0 + i
        for name, pol in policies.items():
            env.reset(seed=seed)
            start, target = env.sim.pose(), tuple(float(t) for t in env.target)
            res = run_leg(env, pol)
            episodes[name].append({"seed": seed, "start": [round(s, 2) for s in start],
                                   "target": [round(t, 2) for t in target], **res})
        p, b = episodes["ppo"][-1], episodes["baseline"][-1]
        print(f"ep {i:2d}  PPO: {'OK ' if p['success'] else 'FAIL'} {p['time_s']:5.1f}s {p['path_m']:5.2f}m   "
              f"baseline: {'OK ' if b['success'] else 'FAIL'} {b['time_s']:5.1f}s {b['path_m']:5.2f}m")

    rec = None if args.no_video else Recorder(OUT / "ppo_patrol.mp4", env.sim)
    patrols = {"ppo": patrol(env, policies["ppo"], rec), "baseline": patrol(env, policies["baseline"])}
    if rec:
        rec.close()

    summary = {k: summarize(v) for k, v in episodes.items()}
    for k, legs in patrols.items():
        summary[k]["patrol_legs_reached"] = f"{sum(leg['success'] for leg in legs)}/7"
        summary[k]["patrol_time_s"] = round(sum(leg["time_s"] for leg in legs), 1)
    accept = summary["ppo"]["success_rate"] >= 0.8
    results = {"summary": summary, "acceptance_ppo_success>=0.8": accept, "episodes": episodes, "patrol": patrols}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    keys = ["success_rate", "failures(collision/out)", "timeouts", "mean_time_s(success)",
            "mean_path_m(success)", "patrol_legs_reached", "patrol_time_s"]
    md = ["| metric | PPO | deterministic baseline |", "|---|---|---|"]
    md += [f"| {k} | {summary['ppo'][k]} | {summary['baseline'][k]} |" for k in keys]
    (OUT / "results.md").write_text(f"Held-out episodes: {args.episodes} (seeds {HELDOUT_SEED0}+)\n\n" + "\n".join(md) + "\n")

    print("\n" + "\n".join(md))
    print("\nPPO patrol (leg, reached, time_s, final_dist, yaw_err_deg, speed):")
    for leg in patrols["ppo"]:
        print(f"  {leg['leg']:<15} {leg['success']!s:<5} {leg['time_s']:5.2f} {leg['final_dist']:.3f} "
              f"{leg['final_yaw_err_deg']:5.1f} {leg['final_speed']:.3f}")
    print(f"\nAcceptance (PPO success >= 80% on held-out): {'PASS' if accept else 'FAIL'}")
    env.close()
    sys.exit(0 if accept else 1)


if __name__ == "__main__":
    main()

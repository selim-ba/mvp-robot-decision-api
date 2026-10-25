"""Validate the navigation env, then train PPO (Stable-Baselines3) and save the policy.

Usage: python scripts/train_ppo.py [--steps 2000000] [--n-envs 8] [--seed 0]
Output: models/ppo_nav.zip, outputs/phase2/train_info.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gymnasium.utils.env_checker import check_env as gym_check_env  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_checker import check_env as sb3_check_env  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from fieldpilot.nav_env import FieldNavEnv  # noqa: E402

MODEL_PATH = ROOT / "models" / "ppo_nav.zip"
OUT = ROOT / "outputs" / "phase2"


def validate_env():
    env = FieldNavEnv()
    gym_check_env(env)
    env.close()
    env = FieldNavEnv()
    sb3_check_env(env, warn=True)
    env.close()
    print("env validation: gymnasium check_env PASSED, SB3 check_env PASSED")


class Progress(BaseCallback):
    """Compact progress line every few PPO iterations."""

    def __init__(self, every: int = 5):
        super().__init__()
        self.every, self.it, self.t0, self.history = every, 0, time.time(), []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        self.it += 1
        if self.it % self.every:
            return
        succ = float(np.mean(self.model.ep_success_buffer)) if self.model.ep_success_buffer else float("nan")
        eps = self.model.ep_info_buffer
        rew = float(np.mean([e["r"] for e in eps])) if eps else float("nan")
        length = float(np.mean([e["l"] for e in eps])) if eps else float("nan")
        fps = self.num_timesteps / (time.time() - self.t0)
        self.history.append({"timesteps": self.num_timesteps, "success_rate": succ, "ep_rew": rew, "ep_len": length})
        print(f"steps {self.num_timesteps:>9,}  success {succ:5.2f}  ep_rew {rew:7.2f}  ep_len {length:6.1f}  "
              f"{fps:6.0f} steps/s  {time.time() - self.t0:6.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    validate_env()

    vec = make_vec_env(FieldNavEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy", vec,
        n_steps=1024, batch_size=512, n_epochs=10, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128]),   # standard SB3 MLP, just sized
        device="cpu", seed=args.seed, verbose=0,
    )
    cb = Progress()
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    wall = time.time() - t0
    model.save(MODEL_PATH)
    vec.close()

    info = {"timesteps": args.steps, "n_envs": args.n_envs, "seed": args.seed, "wall_time_s": round(wall, 1),
            "final_train_success_rate": cb.history[-1]["success_rate"] if cb.history else None,
            "history": cb.history}
    (OUT / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved {MODEL_PATH} after {wall / 60:.1f} min")


if __name__ == "__main__":
    main()

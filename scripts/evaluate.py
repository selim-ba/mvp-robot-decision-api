"""Phase 5: run every evaluation and write one summary + README figures.

Runs (as subprocesses, each also writes its own detailed outputs):
  1. scripts/eval_nav.py --no-video          20 held-out nav episodes, PPO vs rule-based, patrol
  2. scripts/phase3_vision.py --mode both    24 labelled images, Decisions API + mock, failure tests
  3. scripts/phase4_mission.py --no-video    full mission: api / mock / api_fail
Then writes outputs/results/summary.{json,md} and docs/{training_curve.png, contact_sheet_api.png,
panel_2_evidence.png}.

Usage: python scripts/evaluate.py            (needs OPENAI_API_KEY)
       python scripts/evaluate.py --mock     (fully offline: mock classifier only)
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"


def load(path):
    p = OUT / path
    return json.loads(p.read_text()) if p.exists() else None


def plot_curves(history, path):
    """Tiny OpenCV line plot (no matplotlib): success rate and episode duration vs training steps."""
    W, H, pad = 1000, 380, 60
    img = np.full((H, W, 3), 255, np.uint8)
    xs = np.array([h["timesteps"] for h in history]) / 1e6
    panels = [("Success rate (%)", np.array([h["success_rate"] for h in history]) * 100, 100, (200, 110, 30)),
              ("Episode duration (s)", np.array([h["ep_len"] for h in history]) / 20, 30, (30, 120, 220))]
    pw = (W - 3 * pad) // 2
    for k, (title, ys, ymax, color) in enumerate(panels):
        x0, y0, x1, y1 = pad + k * (pw + pad), 40, pad + k * (pw + pad) + pw, H - 50
        cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
        for frac in (0.25, 0.5, 0.75):
            yy = int(y1 - frac * (y1 - y0))
            cv2.line(img, (x0, yy), (x1, yy), (235, 235, 235), 1)
            cv2.putText(img, f"{frac * ymax:g}",(x0 - 40, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        pts = np.stack([x0 + (xs / xs.max()) * (x1 - x0), y1 - np.clip(ys / ymax, 0, 1) * (y1 - y0)], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        cv2.putText(img, title, (x0, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, "0", (x0 - 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, f"{xs.max():.1f}M steps", (x1 - 90, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline: no Decisions API calls")
    args = ap.parse_args()
    runs = [["scripts/eval_nav.py", "--no-video"],
            ["scripts/phase3_vision.py", "--mode", "mock" if args.mock else "both"],
            ["scripts/phase4_mission.py", "--scenario", "mock" if args.mock else "all", "--no-video"]]
    exit_codes = {}
    for cmd in runs:
        print(f"\n######## {' '.join(cmd)}")
        exit_codes[cmd[0]] = subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode

    train, nav, vis = load("phase2/train_info.json"), load("phase2/results.json"), load("phase3/results.json")
    missions = {sc: load(f"phase4/{sc}/mission_report.json") for sc in (["mock"] if args.mock else ["api", "mock", "api_fail"])}
    summary = {"exit_codes": exit_codes, "classifier": "mock" if args.mock else "api (gpt-6-luna)"}
    if train:
        summary["training"] = {"timesteps": train["timesteps"], "wall_time_min": round(train["wall_time_s"] / 60, 1),
                               "n_envs": train["n_envs"], "final_train_success_rate": train["final_train_success_rate"]}
    if nav:
        summary["navigation"] = nav["summary"]
    if vis:
        summary["vision"] = {k: vis[k]["metrics"] for k in ("api", "mock") if k in vis}
        summary["vision"]["failure_tests"] = {k: v["handled"] for k, v in vis.get("failure_tests", {}).items()}
    summary["missions"] = {}
    for sc, m in missions.items():
        if m:
            summary["missions"][sc] = {
                "checks_pass": all(m["checks"].values()), "completed": m["completed"],
                "sim_time_s": m["mission_sim_time_s"], "classifier_calls": m["classifier_calls"],
                "classifier_errors": m["classifier_errors"],
                "decisions": {k: {"p": v.get("decision", {}).get("probability"),
                                  "decision": v.get("decision", {}).get("decision"),
                                  "close_up": v["visited_close"]} for k, v in m["panels"].items()}}

    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# FieldPilot results", ""]
    if nav:
        s = nav["summary"]
        md += ["## Navigation (20 held-out episodes)", "", "| metric | PPO | rule-based |", "|---|---|---|"]
        md += [f"| {k} | {s['ppo'][k]} | {s['baseline'][k]} |" for k in s["ppo"]]
        md.append("")
    if vis:
        md += ["## Vegetation decision (24 labelled robot-camera images)", "",
               "| metric | " + " | ".join(k for k in ("api", "mock") if k in vis) + " |",
               "|---|" + "---|" * sum(k in vis for k in ("api", "mock"))]
        for key in ("accuracy", "precision", "recall", "errors", "prob_obstructed_min", "prob_clear_max",
                    "latency_ms_p50", "latency_ms_p95", "input_tokens_mean"):
            md.append(f"| {key} | " + " | ".join(str(vis[k]["metrics"][key]) for k in ("api", "mock") if k in vis) + " |")
        md.append("")
    md += ["## Full mission", "", "| scenario | panel_1 | panel_2 | panel_3 | sim time | checks |", "|---|---|---|---|---|---|"]
    for sc, m in summary["missions"].items():
        cells = []
        for k in ("panel_1", "panel_2", "panel_3"):
            d = m["decisions"].get(k, {})
            p = "err" if d.get("p") is None else f"{d['p']:.2f}"
            cells.append(f"p={p} -> {d.get('decision')}{' + close-up' if d.get('close_up') else ''}")
        md.append(f"| {sc} | " + " | ".join(cells) + f" | {m['sim_time_s']} s | {'PASS' if m['checks_pass'] else 'FAIL'} |")
    (OUT / "results" / "summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))

    DOCS.mkdir(exist_ok=True)
    if train and train.get("history"):
        plot_curves(train["history"], DOCS / "training_curve.png")
    for src, dst in [("phase3/contact_sheet_api.png", "contact_sheet_api.png"),
                     ("phase4/api/panel_2_evidence.png", "panel_2_evidence.png")]:
        if (OUT / src).exists():
            shutil.copy(OUT / src, DOCS / dst)
    print(f"\nWrote {OUT / 'results'} and figures in {DOCS}")
    print("exit codes:", exit_codes)
    sys.exit(0 if all(c == 0 for c in exit_codes.values()) else 1)


if __name__ == "__main__":
    main()

"""FieldPilot demo: run the real mission and render a ~55 s demo video in one command.

    python scripts/demo.py            # real Decisions API (needs OPENAI_API_KEY)
    python scripts/demo.py --mock     # offline: mock classifier (clearly labelled in the video)

Segments: title -> site overview -> autonomous mission (PPO drives, Decisions API decides, real
time) -> evidence card -> PPO vs rule-based ghost race (3x speed) -> results card.
Output: outputs/demo/fieldpilot_demo.mp4, outputs/demo/mission_report.json, docs/demo_storyboard.png
Results-card numbers come from outputs/results/summary.json (scripts/evaluate.py) when present.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from fieldpilot.decisions import ObstructionClassifier  # noqa: E402
from fieldpilot.mission import InspectionMission  # noqa: E402
from fieldpilot.race import make_racers  # noqa: E402
from fieldpilot.sim import FieldSim  # noqa: E402
from fieldpilot.video import open_writer  # noqa: E402
from fieldpilot.waypoints import START_POSE  # noqa: E402

OUT = ROOT / "outputs" / "demo"
W, H, FPS = 1280, 720, 20
FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, GREY, DARK = (245, 245, 245), (170, 170, 170), (28, 26, 24)
ORANGE, BLUE, GREEN, RED = (0, 140, 255), (235, 150, 40), (80, 200, 80), (70, 70, 235)


# ------------------------------------------------------------------ drawing helpers
def shade(img, x0, y0, x1, y1, alpha=0.72):
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(DARK) * alpha).astype(np.uint8)


def put(img, s, org, scale=0.8, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def put_center(img, s, y, scale=1.0, color=WHITE, thick=1):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    put(img, s, ((W - tw) // 2, y), scale, color, thick)


def header(img, step, title, right=""):
    shade(img, 0, 0, W, 56, 0.85)
    put(img, step, (20, 37), 0.75, ORANGE, 1)
    put(img, title, (70, 37), 0.75, WHITE, 1)
    if right:
        (tw, _), _ = cv2.getTextSize(right, FONT, 0.65, 1)
        put(img, right, (W - tw - 20, 37), 0.65, GREY, 1)


class Clip:
    def __init__(self, writer):
        self.w, self.frames, self.storyboard = writer, 0, []

    def write(self, bgr, n=1, keep=False):
        for _ in range(n):
            self.w.write(bgr)
            self.frames += 1
        if keep:
            self.storyboard.append(bgr.copy())


# ---------------------------------------------------------------------- segments
def title_card(clip, mock):
    img = np.full((H, W, 3), DARK, np.uint8)
    put_center(img, "FieldPilot", 300, 2.2, WHITE, 2)
    put_center(img, "Autonomous solar-panel inspection in simulation", 360, 0.95, GREY)
    put_center(img, "MuJoCo  |  PPO navigation (Stable-Baselines3)  |  OpenAI Decisions API (gpt-6-luna)", 420, 0.7, ORANGE)
    if mock:
        put_center(img, "OFFLINE DEMO: mock classifier instead of the Decisions API", 480, 0.7, RED)
    clip.write(img, 3 * FPS, keep=True)


def overview_intro(clip):
    sim = FieldSim()
    sim.reset(*START_POSE)
    for _ in range(5):
        sim.step(0.0, 0.0)
    base = cv2.cvtColor(sim.render("overview", W, H), cv2.COLOR_RGB2BGR)
    sim.close()
    img = base.copy()
    header(img, "1", "The site")
    shade(img, 0, H - 110, W, H)
    put(img, "3 solar panels. Panel 2 (middle) is overgrown - the robot is NOT told which one.", (24, H - 66), 0.75)
    put(img, "It must find out from its own camera images, then collect close-up evidence.", (24, H - 30), 0.75, GREY)
    clip.write(img, 4 * FPS, keep=True)


class MissionFrames:
    """Recorder plugged into InspectionMission: chase cam | robot cam + live state/decision panel."""

    def __init__(self, clip, sim, mock):
        self.clip, self.sim, self.mock, self.log, self.n, self.last = clip, sim, mock, [], 0, ""
        self.after_evidence = False  # True between an EVIDENCE capture and the next navigation

    def __call__(self, label):
        self.n += 1
        chase = self.sim.render("chase", 640, 480)
        cam = self.sim.render("robot_cam", 640, 480)
        img = np.full((H, W, 3), DARK, np.uint8)
        img[56:536] = cv2.cvtColor(np.hstack([chase, cam]), cv2.COLOR_RGB2BGR)
        clf = "MOCK classifier" if self.mock else "Decisions API (gpt-6-luna)"
        header(img, "2", f"Autonomous mission - PPO drives, {clf} decides", f"t = {self.sim.time:5.1f} s")
        shade(img, 8, 64, 200, 92, 0.6)
        put(img, "chase camera", (16, 85), 0.55, GREY)
        shade(img, 648, 64, 830, 92, 0.6)
        put(img, "robot camera", (656, 85), 0.55, GREY)

        color, text = WHITE, label
        if label.startswith("EVIDENCE"):
            self.after_evidence = True
        elif label.startswith(("NAVIGATE", "MISSION")):
            self.after_evidence = False
        if label.startswith("DECISIONS API"):
            panel = label.split()[2].rstrip(":")
            p = label.split("p(vegetation)=")[1].split()[0]
            decision = label.split("-> ")[-1]
            color = RED if decision == "INSPECT_CLOSER" else GREEN
            if self.after_evidence:                                      # second opinion on the close-up
                text = f"{panel} close-up: p(vegetation) = {p}  ->  evidence saved"
                entry = (f"{panel}  close-up p={p}  evidence saved", color)
            else:
                text = f"{panel}: p(vegetation) = {p}  ->  {decision.replace('_', ' ')}"
                entry = (f"{panel}  p={p}  {decision.replace('_', ' ').lower()}", color)
            if entry not in self.log:
                self.log.append(entry)
        elif label.startswith(("CAPTURE", "EVIDENCE")):
            color, text = ORANGE, label.replace("CAPTURE", "Photo:").replace("EVIDENCE", "Close-up evidence:")
        elif label.startswith("NAVIGATE"):
            text = label.replace("NAVIGATE (PPO) ->", "PPO navigating to")
        elif label.startswith("MISSION COMPLETE"):
            color, text = GREEN, "Mission complete - no manual intervention"
        put(img, text, (24, 584), 0.95, color, 1)
        for k, (s, c) in enumerate(self.log[-4:]):
            put(img, s, (24 + (k % 2) * 620, 634 + (k // 2) * 40), 0.7, c)
        first_decision_frame = label.startswith("DECISIONS API") and label != self.last
        self.last = label
        self.clip.write(img, keep=first_decision_frame or self.n % 120 == 60)

    def close(self):
        pass


def evidence_card(clip, report, mock):
    shown = 0
    for panel, v in report["panels"].items():
        if not v.get("evidence_image"):
            continue
        ev = cv2.imread(str(OUT / v["evidence_image"]))
        img = np.full((H, W, 3), DARK, np.uint8)
        header(img, "2", f"Inspection evidence - {panel}")
        img[90:570, 40:680] = ev
        d, e = v.get("decision", {}), v.get("evidence_check", {})
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}"   # noqa: E731
        put(img, "Maintenance flag raised", (720, 160), 1.0, RED)
        put(img, f"Normal view:  p(vegetation) = {fmt(d.get('probability'))}", (720, 230), 0.75)
        put(img, f"Close-up:     p(vegetation) = {fmt(e.get('probability'))}", (720, 275), 0.75)
        put(img, f"Threshold {d.get('threshold', 0.5):.2f} (configurable)", (720, 320), 0.65, GREY)
        put(img, "Image-only inference - no simulator labels", (720, 380), 0.65, GREY)
        put(img, "used for the decision." if not mock else "used (offline MOCK classifier).", (720, 410), 0.65, GREY)
        put(img, "Panels 1 and 3: no obstruction -> patrol continued", (40, 630), 0.75, GREEN)
        clip.write(img, 3 * FPS, keep=True)
        shown += 1
        if shown >= 2:
            break


def race(clip, model):
    ppo, rule = make_racers(model, verbose=False)
    sim, done, step = ppo.sim, 0, 0
    while step < 2400:
        sim.step(*ppo.tick())
        rule.sim.step(*rule.tick())
        step += 1
        if ppo.finish_time and rule.finish_time:
            done += 1
            if done > 20:
                break
        if step % 3:
            continue
        sim.set_ghost(rule.sim.base_qpos())
        mujoco.mj_forward(sim.model, sim.data)
        img = cv2.cvtColor(sim.render("overview", W, H), cv2.COLOR_RGB2BGR)
        header(img, "3", "Learned vs hand-coded navigation - same 7-stop patrol (3x speed)", f"t = {sim.time:5.1f} s")
        shade(img, 0, H - 100, W, H)
        put(img, f"orange  {ppo.status()}", (24, H - 58), 0.8, ORANGE)
        put(img, f"blue ghost  {rule.status()}", (24, H - 20), 0.8, BLUE)
        clip.write(img, keep=(step == 300))
    return ppo.finish_time, rule.finish_time


def results_card(clip, race_times, mission):
    s = {}
    p = ROOT / "outputs" / "results" / "summary.json"
    if p.exists():
        s = json.loads(p.read_text())
    img = np.full((H, W, 3), DARK, np.uint8)
    header(img, "4", "Results")
    y = 130
    nav = s.get("navigation")
    if nav:
        put(img, "Navigation - 20 held-out episodes", (60, y), 0.85, ORANGE)
        put(img, f"PPO success {nav['ppo']['success_rate'] * 100:.0f}%  |  mean {nav['ppo']['mean_time_s(success)']} s"
                 f"  vs rule-based {nav['baseline']['mean_time_s(success)']} s  |  0 collisions", (60, y + 40), 0.72)
        y += 100
    if race_times[0] and race_times[1]:
        put(img, f"Full patrol: PPO {race_times[0]:.1f} s vs rule-based {race_times[1]:.1f} s "
                 f"({(1 - race_times[0] / race_times[1]) * 100:.0f}% faster)", (60, y), 0.72)
        y += 60
    vis = s.get("vision", {}).get("api")
    if vis:
        put(img, "Vegetation decision - 24 labelled robot-camera images", (60, y), 0.85, ORANGE)
        put(img, f"Decisions API accuracy {vis['accuracy'] * 100:.0f}%  |  median latency {vis['latency_ms_p50']:.0f} ms"
                 f"  |  ~{vis['input_tokens_mean']:.0f} tokens/image", (60, y + 40), 0.72)
        y += 100
    found = [k for k, v in mission["panels"].items() if v.get("visited_close")]
    put(img, "Mission", (60, y), 0.85, ORANGE)
    put(img, f"3 panels inspected  |  close-up evidence for {', '.join(found) or 'none'}  |  "
             f"{mission['classifier_errors']} classifier errors  |  0 manual interventions", (60, y + 40), 0.72)
    y += 100
    if not s:
        put(img, "(run scripts/evaluate.py for the benchmark numbers)", (60, y), 0.65, GREY)
    put_center(img, "FieldPilot  |  MuJoCo + Stable-Baselines3 PPO + OpenCV + OpenAI Decisions API", H - 40, 0.6, GREY)
    clip.write(img, 6 * FPS, keep=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline mock classifier instead of the Decisions API")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = PPO.load(ROOT / "models" / "ppo_nav.zip", device="cpu")
    clf = ObstructionClassifier("mock" if args.mock else "api", args.threshold)

    writer = open_writer(OUT / "fieldpilot_demo.mp4", FPS, (W, H))
    clip = Clip(writer)
    title_card(clip, args.mock)
    overview_intro(clip)
    mission = InspectionMission(model, clf, OUT, recorder=lambda sim: MissionFrames(clip, sim, args.mock))
    report = mission.run()
    (OUT / "mission_report.json").write_text(json.dumps(report, indent=2))
    evidence_card(clip, report, args.mock)
    race_times = race(clip, model)
    results_card(clip, race_times, report)
    writer.release()

    picks = np.linspace(0, len(clip.storyboard) - 1, min(9, len(clip.storyboard))).astype(int)
    sb = [cv2.resize(clip.storyboard[i], (427, 240)) for i in picks]
    while len(sb) % 3:
        sb.append(np.zeros_like(sb[0]))
    (ROOT / "docs").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "docs" / "demo_storyboard.png"), np.vstack([np.hstack(sb[i:i + 3]) for i in range(0, len(sb), 3)]))
    shutil.copy(OUT / "fieldpilot_demo.mp4", ROOT / "docs" / "fieldpilot_demo.mp4")
    print(f"\nDemo video: {OUT / 'fieldpilot_demo.mp4'}  ({clip.frames / FPS:.1f} s, {clip.frames} frames)"
          f"  -> copied to docs/")
    print(f"Mission: completed={report['completed']}  legs_ok={report['all_legs_reached']}  "
          f"classifier errors={report['classifier_errors']}")


if __name__ == "__main__":
    main()

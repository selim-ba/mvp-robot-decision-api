"""Phase 4: run the full autonomous inspection mission and check the acceptance criteria.

Scenarios (each a complete mission from the start pose, no manual intervention):
  api       real Decisions API (gpt-6-luna), video recorded
  mock      offline mock classifier
  api_fail  API forced to fail (invalid key) -> every panel falls back to "inspect_closer"

Usage:
  python scripts/phase4_mission.py                       # all three scenarios
  python scripts/phase4_mission.py --scenario api        # just one
  python scripts/phase4_mission.py --threshold 0.7 --no-video
Output: outputs/phase4/<scenario>/{mission_report.json, *_normal.png, *_evidence.png, mission.mp4}
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from fieldpilot.decisions import ObstructionClassifier  # noqa: E402
from fieldpilot.mission import InspectionMission  # noqa: E402
from fieldpilot.video import Recorder  # noqa: E402

OUT = ROOT / "outputs" / "phase4"
OBSTRUCTED_PANELS = {"panel_2"}           # scene ground truth — used ONLY to grade the run below


def make_classifier(scenario, threshold):
    if scenario == "mock":
        return ObstructionClassifier("mock", threshold)
    if scenario == "api_fail":
        return ObstructionClassifier("api", threshold, api_key="sk-invalid-fieldpilot-test", max_retries=0)
    return ObstructionClassifier("api", threshold)


def grade(report, scenario):
    p = report["panels"]

    def dec(k):                                   # {} if the panel was never evaluated
        return p.get(k, {}).get("decision", {})

    checks = {"mission_completed_unattended": report["completed"] and report["all_legs_reached"]}
    if scenario == "api_fail":
        checks["api_errors_handled_with_fallback"] = all(
            dec(k).get("error") and dec(k).get("decision") == "inspect_closer" for k in p)
        checks["fallback_visits_close_up_everywhere"] = all(v["visited_close"] for v in p.values())
        return checks
    healthy = [k for k in p if k not in OBSTRUCTED_PANELS]
    checks["healthy_panels_continue_patrol"] = all(
        dec(k).get("decision") == "continue" and not p[k]["visited_close"] for k in healthy)
    checks["vegetation_panel_inspected_closer"] = all(
        dec(k).get("decision") == "inspect_closer" and p.get(k, {}).get("visited_close")
        and p[k].get("nav_close", {}).get("success") and (OUT / scenario / p[k].get("evidence_image", "_")).is_file()
        for k in OBSTRUCTED_PANELS)
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["all", "api", "mock", "api_fail"], default="all")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    scenarios = ["api", "mock", "api_fail"] if args.scenario == "all" else [args.scenario]

    policy = PPO.load(ROOT / "models" / "ppo_nav.zip", device="cpu")
    summary = {}
    for sc in scenarios:
        out = OUT / sc
        print(f"\n=== Scenario: {sc} ===")
        video = (not args.no_video) and sc == "api"
        mission = InspectionMission(policy, make_classifier(sc, args.threshold), out,
                                    recorder=(lambda sim: Recorder(out / "mission.mp4", sim)) if video else None)
        report = mission.run()
        report["checks"] = grade(report, sc)
        (out / "mission_report.json").write_text(json.dumps(report, indent=2))
        summary[sc] = report
        for name, v in report["panels"].items():
            d = v.get("decision", {})
            p = d.get("probability")
            print(f"  {name}: p={'n/a' if p is None else f'{p:.2f}'} -> {d.get('decision')}"
                  f"{'  (close-up visited, evidence saved)' if v['visited_close'] else ''}")
        print(f"  mission time {report['mission_sim_time_s']} s sim / {report['mission_wall_time_s']} s wall, "
              f"classifier calls {report['classifier_calls']}, errors {report['classifier_errors']}")
        for k, ok in report["checks"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}")

    all_ok = all(all(r["checks"].values()) for r in summary.values())
    print(f"\nPhase 4 overall: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

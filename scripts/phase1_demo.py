"""Phase 1 acceptance run.

1. Movement tests: drive straight, turn in place (robot must move physically).
2. Patrol with the deterministic controller: visit the normal and close-up waypoint of every
   panel and save robot-camera images at each.
3. Save overview images, a contact sheet, an optional video, and outputs/phase1/report.json.

The green/blue pixel fractions below are a *rendering sanity check only* (is the panel in view,
is vegetation visible?). They are never used for inspection decisions.

Usage: python scripts/phase1_demo.py [--no-video]
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fieldpilot.controller import GoToPoseController  # noqa: E402
from fieldpilot.sim import FieldSim, wrap_angle  # noqa: E402
from fieldpilot.video import CAM_H, CAM_W, Recorder  # noqa: E402
from fieldpilot.waypoints import END_POSE, START_POSE, inspection_waypoints  # noqa: E402

OUT = ROOT / "outputs" / "phase1"


def save_rgb(path: Path, rgb: np.ndarray):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def pixel_fractions(rgb: np.ndarray):
    """Fraction of saturated blue (panel cells) and green (vegetation) pixels. Sanity check only."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h >= 95) & (h <= 130) & (s >= 140) & (v >= 25)
    green = (h >= 35) & (h <= 85) & (s >= 100) & (v >= 40)
    return float(blue.mean()), float(green.mean())


# ------------------------------------------------------------------ movement tests
def test_straight(sim: FieldSim):
    sim.reset(*START_POSE)
    x0, y0, _ = sim.pose()
    for _ in range(40):                      # 2 s at 20 Hz, both wheels 6 rad/s (~0.48 m/s)
        sim.step(6.0, 6.0)
    x1, y1, yaw1 = sim.pose()
    v, _ = sim.velocity()
    res = {"dx": x1 - x0, "dy": y1 - y0, "final_yaw_deg": float(np.degrees(yaw1)), "speed": v,
           "upright": sim.is_upright()}
    res["pass"] = bool(res["dx"] > 0.5 and abs(res["dy"]) < 0.1 and res["upright"])
    return res


def test_turn(sim: FieldSim):
    sim.reset(*START_POSE)
    x0, y0, yaw0 = sim.pose()
    for _ in range(20):                      # 1 s spin in place
        sim.step(-4.0, 4.0)
    x1, y1, yaw1 = sim.pose()
    res = {"dyaw_deg": float(np.degrees(wrap_angle(yaw1 - yaw0))),
           "drift": float(np.hypot(x1 - x0, y1 - y0)), "upright": sim.is_upright()}
    res["pass"] = bool(res["dyaw_deg"] > 30 and res["drift"] < 0.1 and res["upright"])
    return res


# ------------------------------------------------------------------------ patrol
def drive_to(sim, ctrl, target, label, rec, timeout_s=40.0):
    ctrl.reset()
    t0, collided, done = sim.time, False, False
    while sim.time - t0 < timeout_s:
        v, w, done = ctrl(sim.pose(), target)
        if done:
            break
        sim.step(*sim.twist_to_wheels(v, w))
        collided |= sim.in_collision()
        rec(label)
    x, y, yaw = sim.pose()
    return {
        "target": [round(t, 3) for t in target],
        "reached": bool(done),
        "pos_err": float(np.hypot(target[0] - x, target[1] - y)),
        "yaw_err_deg": float(np.degrees(abs(wrap_angle(target[2] - yaw)))),
        "time_s": round(sim.time - t0, 2),
        "collision": bool(collided),
        "upright": sim.is_upright(),
    }


def contact_sheet(images: dict, path: Path):
    """Grid: one column per panel, rows = normal / close."""
    tw, th = 320, 240
    panels = sorted({k.rsplit("_", 1)[0] for k in images})
    rows = []
    for kind in ("normal", "close"):
        tiles = []
        for p in panels:
            tile = cv2.resize(cv2.cvtColor(images[f"{p}_{kind}"], cv2.COLOR_RGB2BGR), (tw, th))
            cv2.putText(tile, f"{p} {kind}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sim = FieldSim()
    report = {"platform": f"{platform.system()} {platform.machine()}", "python": platform.python_version(),
              "mujoco": mujoco.__version__,
              "model": {"nbody": sim.model.nbody, "ngeom": sim.model.ngeom, "nu": sim.model.nu,
                        "timestep": sim.model.opt.timestep, "substeps": sim.n_substeps,
                        "wheel_radius": sim.wheel_radius, "half_track": sim.half_track,
                        "max_wheel_speed": sim.max_wheel_speed}}

    # 1. Movement tests
    report["test_straight"] = test_straight(sim)
    report["test_turn"] = test_turn(sim)
    print("straight:", report["test_straight"])
    print("turn    :", report["test_turn"])

    # 2. Patrol: start -> (normal, close) for each panel -> end
    sim.reset(*START_POSE)
    save_rgb(OUT / "overview_start.png", sim.render("overview", 1280, 720))
    wps = inspection_waypoints(sim.panels())
    ctrl = GoToPoseController()
    rec = Recorder(None if args.no_video else OUT / "phase1_patrol.mp4", sim)
    images, legs = {}, []
    wall0 = time.perf_counter()
    for panel, pair in wps.items():
        for kind in ("normal", "close"):
            wp = pair[kind]
            leg = drive_to(sim, ctrl, wp.pose, wp.name, rec)
            for _ in range(10):              # settle 0.5 s before capturing
                sim.step(0.0, 0.0)
                rec(wp.name)
            img = sim.render("robot_cam", CAM_W, CAM_H)
            images[wp.name] = img
            save_rgb(OUT / f"{wp.name}.png", img)
            blue, green = pixel_fractions(img)
            leg.update(name=wp.name, panel_blue_frac=round(blue, 4), green_frac=round(green, 4))
            legs.append(leg)
            print(f"{wp.name:<16} reached={leg['reached']} pos_err={leg['pos_err']:.3f} "
                  f"yaw_err={leg['yaw_err_deg']:.1f}° t={leg['time_s']}s collision={leg['collision']} "
                  f"blue={blue:.3f} green={green:.3f}")
        save_rgb(OUT / f"overview_{panel}.png", sim.render("overview", 1280, 720))
    legs.append({**drive_to(sim, ctrl, END_POSE, "end", rec), "name": "end"})
    wall = time.perf_counter() - wall0
    rec.close()
    save_rgb(OUT / "overview_end.png", sim.render("overview", 1280, 720))
    contact_sheet(images, OUT / "contact_sheet.png")

    report["patrol"] = legs
    report["patrol_sim_time_s"] = round(sim.time, 2)
    report["patrol_wall_time_s"] = round(wall, 2)

    # 3. Acceptance checks
    g = {leg["name"]: leg["green_frac"] for leg in legs if "green_frac" in leg}
    checks = {
        "robot_moves_straight": report["test_straight"]["pass"],
        "robot_turns_in_place": report["test_turn"]["pass"],
        "all_waypoints_reached": all(leg["reached"] for leg in legs),
        "no_collisions": not any(leg["collision"] for leg in legs),
        "stayed_upright": all(leg["upright"] for leg in legs),
        "panel_visible_in_every_image": all(leg["panel_blue_frac"] > 0.02 for leg in legs if "panel_blue_frac" in leg),
        "vegetation_more_visible_on_panel_2": g["panel_2_normal"] > 2 * max(g["panel_1_normal"], g["panel_3_normal"]),
        "images_saved": all((OUT / f"{n}.png").exists() for n in images),
    }
    report["checks"] = checks
    report["all_pass"] = all(checks.values())
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    sim.close()

    print("\nPhase 1 checks:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nOutputs in {OUT}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()

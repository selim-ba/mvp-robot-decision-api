"""Inspection mission: a small finite-state machine.

    NAVIGATE_TO_PANEL --(arrived)--> CAPTURE --> EVALUATE
        EVALUATE --(p >= threshold, or API failure)--> NAVIGATE_CLOSER --> CAPTURE_EVIDENCE --> NEXT_PANEL
        EVALUATE --(p <  threshold)----------------------------------------------------------> NEXT_PANEL
    NEXT_PANEL --> NAVIGATE_TO_PANEL (next panel) | RETURN_HOME (no panels left) --> DONE
    Navigation failure (collision / timeout) --> NEXT_PANEL (logged; the mission keeps going).

All driving is done by the trained PPO policy (no deterministic fallback). Decisions come only
from the classifier on the robot-camera image; simulator labels are never read here.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from .nav_env import FieldNavEnv
from .waypoints import END_POSE, START_POSE, inspection_waypoints

HOLD_STEPS = 10          # 0.5 s brake/settle at each waypoint before taking a photo
PAUSE_FRAMES = 30        # video only: freeze ~1.5 s on capture/decision captions


class InspectionMission:
    def __init__(self, policy, classifier, out_dir: Path, recorder=None):
        self.policy, self.clf, self.out = policy, classifier, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env = FieldNavEnv()
        self.sim = self.env.sim
        self.rec = recorder(self.sim) if recorder else None
        self.waypoints = inspection_waypoints(self.sim.panels())
        self.log, self.panels, self.legs = [], {}, []
        self._started = False

    # ---------------------------------------------------------------- helpers
    def _frame(self, caption, n=1):
        if self.rec:
            for _ in range(n):
                self.rec(caption)

    def _transition(self, src, dst, **info):
        entry = {"t_sim": round(self.sim.time, 2), "from": src, "to": dst, **info}
        self.log.append(entry)
        extra = "  " + ", ".join(f"{k}={v}" for k, v in info.items()) if info else ""
        print(f"  [{entry['t_sim']:6.2f}s] {src:<17} -> {dst:<17}{extra}")

    def _navigate(self, name, target):
        """PPO drives to `target` (x, y, yaw) and stops. Returns the leg record."""
        opts = {"target": target} if self._started else {"start": START_POSE, "target": target}
        obs, _ = self.env.reset(options={**opts, "keep_robot": self._started})
        self._started = True
        t0 = self.sim.time
        while True:
            action = self.policy.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = self.env.step(action)
            self._frame(f"NAVIGATE (PPO) -> {name}")
            if term or trunc:
                break
        for _ in range(HOLD_STEPS):
            self.sim.step(0.0, 0.0)
            self._frame(f"ARRIVED {name}" if info["is_success"] else f"NAV FAILED {name}")
        leg = {"leg": name, "success": bool(info["is_success"]), "failed": bool(info["failed"]),
               "timeout": bool(trunc), "time_s": round(self.sim.time - t0, 2),
               "final_dist": round(info["dist"], 3), "final_yaw_err_deg": round(info["yaw_err_deg"], 1)}
        self.legs.append(leg)
        return leg

    def _capture(self, filename, caption):
        img = self.sim.render("robot_cam", 640, 480)
        cv2.imwrite(str(self.out / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self._frame(caption, PAUSE_FRAMES // 2)
        return img

    def _evaluate(self, img, panel, tag):
        r = self.clf.classify(img)
        p = "n/a (API error)" if r.probability is None else f"{r.probability:.2f}"
        self._frame(f"DECISIONS API {panel}: p(vegetation)={p} thr={r.threshold:.2f} -> {r.decision.upper()}",
                    PAUSE_FRAMES)
        rec = r.to_dict()
        rec["crop_box"] = list(rec["crop_box"])
        self.panels[panel][tag] = rec
        return r

    # -------------------------------------------------------------------- run
    def run(self):
        wall0 = time.perf_counter()
        names = list(self.waypoints)
        i, state, img = 0, "NAVIGATE_TO_PANEL", None
        print(f"Mission start: {len(names)} panels, classifier mode={self.clf.mode}, threshold={self.clf.threshold}")
        while state != "DONE":
            panel = names[i] if i < len(names) else None
            if state == "NAVIGATE_TO_PANEL":
                self.panels[panel] = {"visited_close": False}
                leg = self._navigate(f"{panel} (normal)", self.waypoints[panel]["normal"].pose)
                nxt = "CAPTURE" if leg["success"] else "NEXT_PANEL"
                self.panels[panel]["nav_normal"] = leg
                self._transition(state, nxt, panel=panel, arrived=leg["success"], t_leg=leg["time_s"])
            elif state == "CAPTURE":
                img = self._capture(f"{panel}_normal.png", f"CAPTURE {panel}")
                nxt = "EVALUATE"
                self._transition(state, nxt, image=f"{panel}_normal.png")
            elif state == "EVALUATE":
                r = self._evaluate(img, panel, "decision")
                nxt = "NAVIGATE_CLOSER" if r.decision == "inspect_closer" else "NEXT_PANEL"
                self._transition(state, nxt, p=None if r.probability is None else round(r.probability, 3),
                                 decision=r.decision, latency_ms=round(r.latency_s * 1000), error=r.error)
            elif state == "NAVIGATE_CLOSER":
                leg = self._navigate(f"{panel} (close-up)", self.waypoints[panel]["close"].pose)
                self.panels[panel]["visited_close"] = True
                self.panels[panel]["nav_close"] = leg
                nxt = "CAPTURE_EVIDENCE" if leg["success"] else "NEXT_PANEL"
                self._transition(state, nxt, arrived=leg["success"], t_leg=leg["time_s"])
            elif state == "CAPTURE_EVIDENCE":
                img = self._capture(f"{panel}_evidence.png", f"EVIDENCE {panel}")
                self.panels[panel]["evidence_image"] = f"{panel}_evidence.png"
                r = self._evaluate(img, panel, "evidence_check")      # second opinion on the close-up
                nxt = "NEXT_PANEL"
                self._transition(state, nxt, image=f"{panel}_evidence.png",
                                 p_closeup=None if r.probability is None else round(r.probability, 3))
            elif state == "NEXT_PANEL":
                i += 1
                nxt = "NAVIGATE_TO_PANEL" if i < len(names) else "RETURN_HOME"
                self._transition(state, nxt)
            elif state == "RETURN_HOME":
                leg = self._navigate("end", END_POSE)
                nxt = "DONE"
                self._transition(state, nxt, arrived=leg["success"])
            state = nxt
        self._frame("MISSION COMPLETE", PAUSE_FRAMES)
        if self.rec:
            self.rec.close()
        self.env.close()

        return {
            "navigator": "PPO (models/ppo_nav.zip)",
            "classifier_mode": self.clf.mode, "threshold": self.clf.threshold,
            "completed": state == "DONE",
            "all_legs_reached": all(leg["success"] for leg in self.legs),
            "mission_sim_time_s": round(self.sim.time, 2),
            "mission_wall_time_s": round(time.perf_counter() - wall0, 2),
            "classifier_calls": sum(("decision" in p) + ("evidence_check" in p) for p in self.panels.values()),
            "classifier_errors": sum(bool(p.get(k, {}).get("error")) for p in self.panels.values()
                              for k in ("decision", "evidence_check")),
            "panels": self.panels, "legs": self.legs, "transitions": self.log,
        }

# FieldPilot — project guide for Claude

Portfolio MVP for **Lucé Robotics** (autonomous robots for outdoor industrial infrastructure).
A simulated wheeled robot patrols 3 solar panels, navigates with a trained **PPO** policy,
captures RGB images from its own camera, asks the **OpenAI Decisions API** whether vegetation
is visibly obstructing each panel, and moves closer to collect evidence when it is.

Goal: a **simple, working, reproducible end-to-end MVP** — not a production robotics system.
Deliverables: a GitHub repo + a 45–60 s demo video for a robotics CTO.
Hardware: MacBook Pro M4 Pro (macOS, Apple Silicon). Budget: ~15–20 h total.

---

## Frozen MVP architecture

### Simulation — MuJoCo
- Small field with **3 solar panels**.
- One **differential-drive** robot with a forward-facing **RGB camera** on a mast.
- **One panel partially obstructed by visible vegetation** (panel_2).
- Predefined **normal** and **close-up** inspection waypoints per panel.

### Reinforcement learning — Stable-Baselines3 PPO
- Goal-conditioned navigation policy.
- Input: robot state + relative target position. Output: wheel velocity commands.
- Reward: progress toward target, arrival bonus, collision penalty.
- **PPO must actually drive the robot in the final demo.**
- Keep a simple deterministic navigation baseline for comparison.

### Computer vision — RGB camera
- Images captured directly from the simulated robot camera.
- OpenCV only for simple crop/preprocessing.
- No YOLO, SAM, or any other vision model.

### OpenAI Decisions API — GPT-6 Luna
- Analyse the captured panel image: "Is vegetation visibly obstructing this solar panel?"
- Return an obstruction probability; a configurable threshold decides closer inspection vs. continue.
- **Real image inference only — never simulator ground-truth labels.**
- Never call the API during PPO training.
- Mock mode for offline testing; graceful handling of API failures.
- Use the current official `openai` Python SDK (verify the exact Decisions API call in Phase 3).

### Mission controller (finite-state machine)
1. Navigate to panel (PPO).
2. Capture camera image.
3. Evaluate image (Decisions API).
4. If obstruction suspected → navigate closer (PPO).
5. Capture inspection evidence.
6. Otherwise → continue to next panel.
7. Finish mission.

---

## Strictly out of scope — do NOT implement
Robotic arms/manipulation · ROS, SLAM, mapping · multiple anomaly types · extra CV models ·
custom NN architectures · web interfaces · Docker/cloud deployment · complex 3D assets ·
LLM-based low-level robot control.

Avoid overengineering. Prefer simple, understandable Python.

---

## Roadmap (one phase at a time — STOP after each phase for approval)

| Phase | Content | Acceptance |
|---|---|---|
| 1 Simulation | MuJoCo world, robot, panels, vegetation, RGB camera, deterministic controller | Runs on the Mac; robot moves physically; camera renders healthy + obstructed panels; example images saved |
| 2 RL navigation | Gymnasium env + SB3 PPO | `check_env` passes; PPO reaches ≥80% of targets over 20 held-out episodes; policy saved; compared with deterministic controller |
| 3 Vision + Decisions API | Capture images, call API via official SDK | Real classification works; ≥10 labelled examples tested; outputs, errors, latency recorded; mock mode + failure handling |
| 4 Integration | FSM connecting decisions to PPO navigation | Healthy → continue; vegetation → go closer; full mission with no manual intervention |
| 5 Evaluation + demo | Demo command, metrics, video, README | Reproducible demo; nav + decision results; 45–60 s recording; concise README (architecture, results, limitations) |

After each phase report: what was implemented · files created/modified · tests run + results ·
remaining problems · next recommended action.

---

## Development rules
- Work on ONE phase at a time; stop and wait for approval after each.
- Execute and test code. **Never claim something works without running it.**
- Prefer the simplest implementation; debug before expanding the architecture.
- **Never silently substitute the deterministic controller for PPO.**
- Keep API keys out of source control (`.env` is git-ignored; see `.env.example`).
- No unnecessary dependencies; no features outside the agreed MVP.

---

## Repository layout
```
fieldpilot/
  assets/field.xml   MuJoCo world (robot, 3 panels, vegetation, cameras)
  sim.py             FieldSim: load/reset/step/pose/collision/render
  controller.py      Deterministic go-to-pose baseline controller
  waypoints.py       Panel list + normal/close-up waypoints (derived from the model)
  nav_env.py         Gymnasium env: goal-conditioned pose navigation for PPO
  video.py           Chase-cam | robot-cam mp4 recorder
  vision.py          OpenCV: crop around the blue panel blob, JPEG/base64 encode, green-fraction (mock only)
  race.py            Racer helpers for the PPO vs rule-based ghost race
  mission.py         InspectionMission FSM (PPO navigation + camera + classifier), transition log, report
  decisions.py       ObstructionClassifier: Decisions API predicate -> probability -> threshold decision; mock mode; failure fallback
scripts/
  check_env.py       Print library versions + offscreen render smoke test
  phase1_demo.py     Phase 1 acceptance: movement tests, patrol, save images/report/video
  view.py            Interactive viewer (plain `python`; uv's Python breaks `mjpython`)
  train_ppo.py       Env validation + SB3 PPO training -> models/ppo_nav.zip
  eval_nav.py        20 held-out episodes PPO vs baseline + PPO patrol video
  phase3_vision.py   24-image labelled dataset, API + mock classification, latency/errors, failure tests
  phase4_mission.py  Full mission in 3 scenarios (api / mock / api_fail) + acceptance checks + video
  evaluate.py        Phase 5: runs all evals -> outputs/results/summary.{json,md} + docs/ figures
  demo.py            Phase 5: one-command mission + ~55 s demo video (outputs/demo/, docs/)
  compare.py         "Ghost race": PPO robot + blue mocap ghost mirroring the baseline (viewer or --video)
models/              Trained policy ppo_nav.zip (committed, ~480 KB)
docs/                README figures + demo video (committed)
outputs/             Generated images, videos, reports (git-ignored)
```

## Phase 3 design (Decisions API)
- SDK: `openai` 3.37.0, `client.decisions.create(model="gpt-6-luna", input=[user message with
  input_text + input_image (base64 data URL)], questions=[{"type": "predicate", ...}])`.
  Response: `answers[0].probability` (predicate) or `type == "refusal"`; `usage.input_tokens`.
- Predicate (not choice) format: community reports choice-format probabilities are overconfident.
- Decision: probability >= threshold (default 0.5) -> inspect_closer, else continue.
- Any API failure/refusal -> probability None, error recorded, fallback decision `inspect_closer`.
- Mock mode = green-pixel heuristic on the same image (offline testing only; never sim labels).

## Conventions
- World frame: x along the panel row, y toward the panels, z up. Panels at y = 2.5, facing −y.
- Robot yaw 0 = facing +x. Inspection waypoints face the panel (yaw = +90°).
- Control rate 20 Hz (physics dt = 2 ms, 25 substeps per control step).
- Actions are wheel angular velocities [rad/s], clipped to the actuator ctrlrange.
- Ground truth (`panel_2` has vegetation) may be used for **evaluation labels only**, never for decisions.

## Commands (macOS)
```bash
uv venv -p 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
python scripts/check_env.py
python scripts/phase1_demo.py        # writes outputs/phase1/
python scripts/view.py               # interactive viewer (managed viewer; no mjpython needed)
python scripts/train_ppo.py          # Phase 2 training (~2M steps, CPU)
python scripts/eval_nav.py           # writes outputs/phase2/
python scripts/phase3_vision.py      # writes outputs/phase3/ (needs OPENAI_API_KEY; --mode mock offline)
python scripts/phase4_mission.py     # writes outputs/phase4/<scenario>/ (--scenario mock for offline)
python scripts/evaluate.py           # all benchmarks -> outputs/results/ (--mock offline)
python scripts/demo.py               # demo video -> outputs/demo/fieldpilot_demo.mp4 (--mock offline)
```

## Phase 2 design (navigation env)
- Goal = target pose (x, y, yaw). Success: within 0.15 m and 10°, AND stopped (|v| < 0.1 m/s,
  |w| < 0.3 rad/s). PPO handles heading and stopping, so it arrives facing the panel — no
  deterministic "final turn". (v1 had no stop condition and flew through targets at ~1 m/s.)
- Obs (15): target in robot frame, distance, bearing, heading error, speeds, previous action, absolute pose.
- Action (2): normalized left/right wheel speeds. 20 Hz.
- Reward: potential-based progress (distance + heading within 1 m + low speed within 0.5 m), time/smoothness penalties,
  +10 success, −10 collision / out of area (y > 1.9 is too close to panels) / tipping.
- Starts and targets sampled in x∈[−5.5, 5.5], y∈[−1.5, 1.5]; 50% of targets are inspection waypoints.
- Held-out eval: seeds 10000+ (fresh start/target pairs), same tasks for PPO and baseline.
  Baseline runs with max_v = 0.9 m/s in eval so the speed comparison is fair.

## Status
- Phase 1: done and approved (all 8 checks pass on macOS 15.6 / M4 Pro, mujoco 3.17.0).
  Decision: keep the wheeled robot (quadruped considered and rejected for scope).
- Phase 2: done, awaiting approval. PPO v2 (stop-at-target): 20/20 held-out, mean 5.27 s vs baseline
  7.67 s (baseline max_v 0.9), 0 collisions; PPO patrol 7/7, final errors ≤ 0.06 m / 5.8°, speed < 0.1 m/s.
  Trained 2M steps in 2.8 min on CPU (~11.8k steps/s). Policy: models/ppo_nav.zip (v1 fly-through model: ppo_nav_v1.zip, unused).
  Ghost race (scripts/compare.py): PPO 18.7 s vs rule-based 30.9 s for the 7-stop patrol.
- Phase 3: done, awaiting approval. 24 labelled images (8 obstructed): API 24/24 correct, 0 errors;
  obstructed p = 0.94–1.00, clear p = 0.00 (incl. low grass tufts at panel_1). Latency p50 266 ms,
  mean 481 ms, max 2.4 s (first calls slower). ~363 input tokens/image. Failure tests pass
  (invalid key -> 401 handled, 1 ms timeout handled, mock without key). Caveat: synthetic scene is easy.
- Phase 5: done. evaluate.py: all 3 sub-evals exit 0 (nav 20/20 PPO 5.27 s vs 7.67 s; API 24/24,
  p50 508 ms / p95 941 ms; missions api/mock/api_fail PASS). demo.py: 55.1 s H.264 1280x720 video,
  mission completed, 0 classifier errors. requirements.txt pinned. Public repo:
  https://github.com/selim-ba/mvp-robot-decision-api
- Phase 4: done and approved. All 3 scenarios PASS. api: panel_1 p=0.00 continue, panel_2 p=0.99
  inspect_closer (close-up p=1.00, evidence saved), panel_3 p=0.00 continue; 17.95 s sim / 5.0 s wall,
  4 API calls, 0 errors. mock: same decisions. api_fail: 6/6 errors handled, close-up at every panel, completes.
  FSM: NAVIGATE_TO_PANEL -> CAPTURE -> EVALUATE -> (NAVIGATE_CLOSER ->
  CAPTURE_EVIDENCE) -> NEXT_PANEL -> ... -> RETURN_HOME -> DONE. Nav failure -> NEXT_PANEL.
  API failure -> fallback inspect_closer. Close-up evidence gets a second classifier opinion (logged only).

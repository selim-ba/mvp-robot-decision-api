"""Phase 3 acceptance: robot-camera images -> OpenCV crop -> OpenAI Decisions API (gpt-6-luna).

1. Build a labelled dataset: 8 viewpoints x 3 panels = 24 robot-camera images captured in MuJoCo.
   Labels come from how the scene was built (panel_2 = obstructed) and are used ONLY to score
   the classifier, never as classifier input.
2. Classify every image with the real API and with the offline mock; record probability,
   decision, latency, tokens and errors.
3. Failure-handling tests: invalid API key, forced timeout, mock mode without any key.
4. Write outputs/phase3/{results.json, results.md, contact_sheet_api.png, crops/...}.

Usage: python scripts/phase3_vision.py [--mode both|api|mock] [--threshold 0.5] [--detail auto]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fieldpilot.decisions import ObstructionClassifier  # noqa: E402
from fieldpilot.sim import FieldSim  # noqa: E402

OUT = ROOT / "outputs" / "phase3"
OBSTRUCTED_PANELS = {"panel_2"}           # scene ground truth — evaluation labels only
# (lateral offset dx [m], standoff from panel base [m], extra yaw [deg])
VIEWPOINTS = [(0.0, 2.0, 0), (0.0, 1.2, 0), (-0.6, 1.6, 0), (0.6, 1.6, 0),
              (-0.6, 2.4, 0), (0.6, 2.4, 0), (0.0, 2.0, 10), (0.0, 1.6, -10)]


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def build_dataset():
    (OUT / "dataset").mkdir(parents=True, exist_ok=True)
    sim = FieldSim()
    items = []
    for name, px, py in sim.panels():
        for k, (dx, standoff, dyaw) in enumerate(VIEWPOINTS):
            x, y = px + dx, py - standoff
            yaw = math.atan2(py - y, px - x) + math.radians(dyaw)
            sim.reset(x, y, yaw)
            for _ in range(5):                 # let the robot settle on its wheels
                sim.step(0.0, 0.0)
            img = sim.render("robot_cam", 640, 480)
            fn = f"{name}_v{k}.png"
            save_rgb(OUT / "dataset" / fn, img)
            items.append({"file": fn, "panel": name, "label": int(name in OBSTRUCTED_PANELS),
                          "viewpoint": {"dx": dx, "standoff": standoff, "dyaw_deg": dyaw}, "image": img})
    sim.close()
    (OUT / "dataset" / "labels.json").write_text(json.dumps(
        [{k: v for k, v in it.items() if k != "image"} for it in items], indent=2))
    return items


def run(clf, items, tag):
    rows = []
    for it in items:
        r = clf.classify(it["image"])
        pred = None if r.probability is None else int(r.decision == "inspect_closer")
        rows.append({"file": it["file"], "label": it["label"], "pred": pred, **r.to_dict()})
        p = "  n/a" if r.probability is None else f"{r.probability:.3f}"
        flag = "ERR " if r.error else ("ok  " if pred == it["label"] else "MISS")
        print(f"[{tag}] {it['file']:<14} label={it['label']} p={p} -> {r.decision:<14} "
              f"{r.latency_s * 1000:7.0f} ms {flag}{(' ' + r.error) if r.error else ''}")
    return rows


def metrics(rows):
    ok = [r for r in rows if r["pred"] is not None]
    tp = sum(r["pred"] == 1 and r["label"] == 1 for r in ok)
    tn = sum(r["pred"] == 0 and r["label"] == 0 for r in ok)
    fp = sum(r["pred"] == 1 and r["label"] == 0 for r in ok)
    fn = sum(r["pred"] == 0 and r["label"] == 1 for r in ok)
    lat = np.array([r["latency_s"] for r in ok]) * 1000 if ok else np.array([np.nan])
    pos = [r["probability"] for r in ok if r["label"] == 1]
    neg = [r["probability"] for r in ok if r["label"] == 0]
    toks = [r["input_tokens"] for r in ok if r["input_tokens"]]
    return {
        "n": len(rows), "answered": len(ok), "errors": len(rows) - len(ok),
        "accuracy": round((tp + tn) / len(ok), 3) if ok else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "prob_obstructed_min": round(min(pos), 3) if pos else None,
        "prob_clear_max": round(max(neg), 3) if neg else None,
        "latency_ms_mean": round(float(np.mean(lat)), 1), "latency_ms_p50": round(float(np.median(lat)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1), "latency_ms_max": round(float(np.max(lat)), 1),
        "input_tokens_mean": round(float(np.mean(toks)), 1) if toks else None,
    }


def failure_tests(img, threshold):
    """Each test passes if classify() returns (no exception) with an error and the fallback decision."""
    tests = {}

    def check(name, make):
        try:
            r = make().classify(img)
            tests[name] = {"handled": bool(r.error) and r.decision == "inspect_closer",
                           "error": r.error, "decision": r.decision, "latency_ms": round(r.latency_s * 1000, 1)}
        except Exception as e:  # noqa: BLE001
            tests[name] = {"handled": False, "error": f"UNCAUGHT {type(e).__name__}: {e}"}
        print(f"[fail-test] {name:<16} handled={tests[name]['handled']}  {tests[name]['error']}")

    check("invalid_api_key", lambda: ObstructionClassifier("api", threshold, api_key="sk-invalid-fieldpilot-test", max_retries=0))
    check("timeout_1ms", lambda: ObstructionClassifier("api", threshold, timeout_s=0.001, max_retries=0))

    saved = os.environ.pop("OPENAI_API_KEY", None)        # mock must work with no key at all
    try:
        r = ObstructionClassifier("mock", threshold).classify(img)
        tests["mock_without_key"] = {"handled": r.error is None and r.probability is not None,
                                     "error": r.error, "decision": r.decision}
    except Exception as e:  # noqa: BLE001
        tests["mock_without_key"] = {"handled": False, "error": f"UNCAUGHT {type(e).__name__}: {e}"}
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved
    print(f"[fail-test] mock_without_key handled={tests['mock_without_key']['handled']}")
    return tests


def contact_sheet(items, rows, path):
    tiles = []
    for it, r in zip(items, rows):
        t = cv2.cvtColor(it["image"], cv2.COLOR_RGB2BGR).copy()
        x0, y0, x1, y1 = r["crop_box"]
        cv2.rectangle(t, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
        t = cv2.resize(t, (240, 180))
        correct = r["pred"] == r["label"]
        color = (0, 200, 0) if correct else (0, 0, 255)
        p = "ERR" if r["probability"] is None else f"p={r['probability']:.2f}"
        cv2.rectangle(t, (0, 0), (240, 22), (0, 0, 0), -1)
        cv2.putText(t, f"{it['file'][:-4]} y={it['label']} {p}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        tiles.append(t)
    cols = len(VIEWPOINTS)
    grid = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(grid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "api", "mock"], default="both")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--detail", default="auto", choices=["low", "high", "auto", "original"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crops").mkdir(exist_ok=True)

    items = build_dataset()
    print(f"dataset: {len(items)} images ({sum(i['label'] for i in items)} obstructed) -> {OUT / 'dataset'}")
    results = {"threshold": args.threshold, "detail": args.detail, "n_images": len(items)}

    if args.mode in ("both", "mock"):
        rows = run(ObstructionClassifier("mock", args.threshold), items, "mock")
        results["mock"] = {"metrics": metrics(rows), "rows": rows}

    if args.mode in ("both", "api"):
        clf = ObstructionClassifier("api", args.threshold, detail=args.detail)
        rows = run(clf, items, "api")
        results["api"] = {"metrics": metrics(rows), "rows": rows}
        for it, r in zip(items, rows):           # save exactly what the API saw
            x0, y0, x1, y1 = r["crop_box"]
            save_rgb(OUT / "crops" / it["file"], it["image"][y0:y1, x0:x1])
        contact_sheet(items, rows, OUT / "contact_sheet_api.png")
        results["failure_tests"] = failure_tests(items[0]["image"], args.threshold)

    m_api = results.get("api", {}).get("metrics")
    checks = {}
    if m_api:
        checks["api_answered_at_least_10_labelled"] = m_api["answered"] >= 10
        checks["api_accuracy_at_least_0.8"] = (m_api["accuracy"] or 0) >= 0.8
        checks["latency_recorded"] = not math.isnan(m_api["latency_ms_mean"])
        checks["failure_handling"] = all(t["handled"] for t in results["failure_tests"].values())
    if "mock" in results:
        checks["mock_mode_runs"] = results["mock"]["metrics"]["answered"] == len(items)
    results["checks"] = checks
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    md = [f"Images: {len(items)} (8 viewpoints x 3 panels; obstructed = panel_2). Threshold {args.threshold}.", "",
          "| metric | " + " | ".join(k for k in ("api", "mock") if k in results) + " |",
          "|---|" + "---|" * sum(k in results for k in ("api", "mock"))]
    for key in ("answered", "errors", "accuracy", "precision", "recall", "confusion", "prob_obstructed_min",
                "prob_clear_max", "latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "latency_ms_max",
                "input_tokens_mean"):
        md.append(f"| {key} | " + " | ".join(str(results[k]["metrics"][key]) for k in ("api", "mock") if k in results) + " |")
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print("\nPhase 3 checks:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    sys.exit(0 if checks and all(checks.values()) else 1)


if __name__ == "__main__":
    main()

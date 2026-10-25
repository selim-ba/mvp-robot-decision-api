"""Vegetation-obstruction check via the OpenAI Decisions API (model: gpt-6-luna).

One predicate question per image -> calibrated probability in [0, 1]. A configurable threshold
turns it into a mission decision:  probability >= threshold -> "inspect_closer", else "continue".

Modes
- "api":  real image inference through `client.decisions.create(...)` (official `openai` SDK).
- "mock": offline stand-in for testing without network/key. Uses a crude green-pixel heuristic on
          the SAME image (never simulator labels). Clearly marked as mock in every result.

Failure handling: timeouts, network errors, auth/rate-limit/HTTP errors and model refusals never
crash the caller. They return a result with `error` set, `probability=None`, and the configured
fallback decision (default "inspect_closer": when unsure, look closer rather than miss a defect).

The API is only ever called from inspection code — never during PPO training.
"""
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .vision import crop_to_panel, encode_data_url, green_fraction

MODEL = "gpt-6-luna"
QUESTION_NAME = "vegetation_obstruction"
CONTEXT = ("Image from the forward RGB camera of a ground inspection robot in a solar farm. "
           "The robot is facing a solar panel to check whether it needs maintenance.")
QUESTION = ("Vegetation (bushes, tall grass, vines or leaves) is visibly obstructing or covering "
            "part of the solar panel in this image.")


@dataclass
class InspectionResult:
    probability: float | None          # None when the API failed or refused
    decision: str                      # "inspect_closer" | "continue"
    mode: str                          # "api" | "mock"
    latency_s: float
    threshold: float
    error: str | None = None
    input_tokens: int | None = None
    crop_box: tuple = field(default_factory=tuple)

    def to_dict(self):
        return asdict(self)


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass


class ObstructionClassifier:
    def __init__(self, mode: str = "api", threshold: float = 0.5, timeout_s: float = 15.0,
                 max_retries: int = 1, detail: str = "auto", on_failure: str = "inspect_closer",
                 api_key: str | None = None):
        if mode not in ("api", "mock"):
            raise ValueError("mode must be 'api' or 'mock'")
        self.mode, self.threshold, self.detail, self.on_failure = mode, threshold, detail, on_failure
        self.client = None
        if mode == "api":
            _load_dotenv()
            from openai import OpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set (see .env.example) — use mode='mock' for offline runs")
            self.client = OpenAI(api_key=key,
                                 timeout=timeout_s, max_retries=max_retries)

    def _decide(self, p):
        return "inspect_closer" if p >= self.threshold else "continue"

    def classify(self, rgb: np.ndarray) -> InspectionResult:
        crop, box = crop_to_panel(rgb)
        t0 = time.perf_counter()
        if self.mode == "mock":
            p = float(np.clip(green_fraction(crop) * 10.0, 0.02, 0.98))
            return InspectionResult(p, self._decide(p), "mock", time.perf_counter() - t0, self.threshold, crop_box=box)

        try:
            resp = self.client.decisions.create(
                model=MODEL,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": CONTEXT},
                    {"type": "input_image", "image_url": encode_data_url(crop), "detail": self.detail},
                ]}],
                questions=[{"type": "predicate", "name": QUESTION_NAME, "instructions": QUESTION}],
            )
        except Exception as e:  # noqa: BLE001 — any SDK/network error must not crash the mission
            return InspectionResult(None, self.on_failure, "api", time.perf_counter() - t0, self.threshold,
                                    error=f"{type(e).__name__}: {str(e)[:200]}", crop_box=box)
        latency = time.perf_counter() - t0
        tokens = getattr(getattr(resp, "usage", None), "input_tokens", None)
        ans = resp.answers[0] if resp.answers else None
        if ans is None or ans.type != "predicate":
            kind = "no answer" if ans is None else f"answer type '{ans.type}'"
            return InspectionResult(None, self.on_failure, "api", latency, self.threshold,
                                    error=f"Unusable response: {kind}", input_tokens=tokens, crop_box=box)
        p = float(ans.probability)
        return InspectionResult(p, self._decide(p), "api", latency, self.threshold, input_tokens=tokens, crop_box=box)

"""Print versions of every dependency and run a tiny offscreen render smoke test."""
import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print(f"python      {sys.version.split()[0]}  ({platform.platform()}, {platform.machine()})")
ok = True
for mod, label in [("mujoco", "mujoco"), ("numpy", "numpy"), ("cv2", "opencv"), ("gymnasium", "gymnasium"),
                   ("stable_baselines3", "sb3"), ("torch", "torch"), ("openai", "openai")]:
    try:
        m = importlib.import_module(mod)
        print(f"{label:<11} {getattr(m, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"{label:<11} MISSING ({e})")

try:
    import torch
    print(f"torch MPS   available={torch.backends.mps.is_available()}")
except Exception:
    pass

try:
    from fieldpilot.sim import FieldSim
    sim = FieldSim()
    img = sim.render("robot_cam", 64, 48)
    print(f"model       nbody={sim.model.nbody} ngeom={sim.model.ngeom} nu={sim.model.nu} "
          f"substeps={sim.n_substeps}")
    print(f"render      OK {img.shape} mean={img.mean():.1f}")
    sim.close()
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"sim/render  FAILED: {type(e).__name__}: {e}")

print("ENV OK" if ok else "ENV PROBLEMS — see above")
sys.exit(0 if ok else 1)

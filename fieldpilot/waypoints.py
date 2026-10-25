"""Inspection waypoints, derived from panel positions in the MuJoCo model."""
from dataclasses import dataclass

import numpy as np

START_POSE = (-5.0, 0.5, 0.0)   # x, y, yaw — robot spawn, facing +x along the patrol lane
END_POSE = (5.0, 0.5, 0.0)      # where the patrol finishes
NORMAL_STANDOFF = 2.0           # m in front of the panel base (standard inspection view)
CLOSE_STANDOFF = 1.2            # m in front of the panel base (close-up evidence view)
FACE_PANEL_YAW = np.pi / 2      # panels face -y, so the robot looks along +y


@dataclass
class Waypoint:
    name: str
    panel: str
    kind: str        # "normal" | "close"
    x: float
    y: float
    yaw: float

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)


def inspection_waypoints(panels):
    """panels: list of (name, x, y) from FieldSim.panels(). Returns {panel: {"normal": wp, "close": wp}}."""
    out = {}
    for name, px, py in panels:
        out[name] = {
            "normal": Waypoint(f"{name}_normal", name, "normal", px, py - NORMAL_STANDOFF, FACE_PANEL_YAW),
            "close": Waypoint(f"{name}_close", name, "close", px, py - CLOSE_STANDOFF, FACE_PANEL_YAW),
        }
    return out

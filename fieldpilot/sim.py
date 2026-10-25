"""FieldSim: a thin, readable wrapper around the FieldPilot MuJoCo model."""
from pathlib import Path

import mujoco
import numpy as np

XML_PATH = Path(__file__).parent / "assets" / "field.xml"


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_to_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class FieldSim:
    """Load the field, drive the robot with wheel speeds, read its pose, render cameras."""

    def __init__(self, xml_path=XML_PATH, control_hz: int = 20):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.control_dt = 1.0 / control_hz
        self.n_substeps = max(1, round(self.control_dt / m.opt.timestep))

        root = m.joint("robot_root").id
        self._qpos = m.jnt_qposadr[root]
        self._qvel = m.jnt_dofadr[root]
        self._act_left = m.actuator("left_motor").id
        self._act_right = m.actuator("right_motor").id

        # Kinematic constants read from the model (single source of truth).
        self.wheel_radius = float(m.geom("left_wheel_geom").size[0])
        self.half_track = abs(float(m.body("left_wheel").pos[1]))
        self.max_wheel_speed = float(m.actuator_ctrlrange[self._act_left][1])
        self.spawn_z = float(m.body("robot").pos[2])

        robot_id = m.body("robot").id
        self._robot_geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == robot_id}
        self._floor = m.geom("floor").id
        self._renderers = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, x: float = -5.0, y: float = 0.5, yaw: float = 0.0):
        mujoco.mj_resetData(self.model, self.data)
        q = self._qpos
        self.data.qpos[q:q + 3] = [x, y, self.spawn_z]
        self.data.qpos[q + 3:q + 7] = yaw_to_quat(yaw)
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def pose(self):
        """Return (x, y, yaw) of the robot base."""
        q = self.data.qpos[self._qpos:self._qpos + 7]
        return float(q[0]), float(q[1]), quat_to_yaw(q[3:7])

    def base_qpos(self) -> np.ndarray:
        """Robot base position + quaternion (7,), e.g. to mirror it onto a ghost in another sim."""
        return self.data.qpos[self._qpos:self._qpos + 7].copy()

    def set_ghost(self, qpos7):
        """Place the visual-only ghost robot at a base pose (position + quaternion)."""
        mid = self.model.body("ghost").mocapid[0]
        self.data.mocap_pos[mid] = qpos7[:3]
        self.data.mocap_quat[mid] = qpos7[3:7]

    def velocity(self):
        """Return (forward speed [m/s], yaw rate [rad/s])."""
        _, _, yaw = self.pose()
        lin = self.data.qvel[self._qvel:self._qvel + 3]          # world frame
        ang = self.data.qvel[self._qvel + 3:self._qvel + 6]      # body frame
        return float(lin[0] * np.cos(yaw) + lin[1] * np.sin(yaw)), float(ang[2])

    def is_upright(self) -> bool:
        """True if the base z-axis still points mostly up."""
        R = self.data.xmat[self.model.body("robot").id].reshape(3, 3)
        return bool(R[2, 2] > 0.8)

    def in_collision(self) -> bool:
        """True if any robot geom touches anything other than the floor."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            r1, r2 = c.geom1 in self._robot_geoms, c.geom2 in self._robot_geoms
            if r1 != r2:
                other = c.geom2 if r1 else c.geom1
                if other != self._floor:
                    return True
        return False

    # ---------------------------------------------------------------- control
    def step(self, left: float, right: float):
        """Apply wheel angular-velocity targets [rad/s] for one control period."""
        lim = self.max_wheel_speed
        self.data.ctrl[self._act_left] = np.clip(left, -lim, lim)
        self.data.ctrl[self._act_right] = np.clip(right, -lim, lim)
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def twist_to_wheels(self, v: float, w: float):
        """Convert body twist (v [m/s], w [rad/s]) to wheel speeds, preserving curvature on saturation."""
        left = (v - w * self.half_track) / self.wheel_radius
        right = (v + w * self.half_track) / self.wheel_radius
        scale = max(1.0, abs(left) / self.max_wheel_speed, abs(right) / self.max_wheel_speed)
        return left / scale, right / scale

    # ------------------------------------------------------------- rendering
    def render(self, camera: str = "robot_cam", width: int = 640, height: int = 480) -> np.ndarray:
        """Render an RGB uint8 image (H, W, 3) from a named camera."""
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        r = self._renderers[key]
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    # ------------------------------------------------------------------ world
    def panels(self):
        """List of (name, x, y) for every panel base body, ordered by x."""
        out = []
        for b in range(self.model.nbody):
            name = self.model.body(b).name
            if name.startswith("panel_") and not name.endswith("_surface"):
                x, y, _ = self.model.body_pos[b]
                out.append((name, float(x), float(y)))
        return sorted(out, key=lambda p: p[1])

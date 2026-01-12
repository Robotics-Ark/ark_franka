#!/usr/bin/env python3
"""Robust pick-and-place test for Newton + Franka using ARK channels.

This version auto-detects the active namespace by subscribing to both
namespaced (e.g., "ark/panda/...") and legacy ("panda/...") channels and
locks onto whichever publishes joint states first.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

import sys

# Ensure we can import from the local utils folder
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ark.client.comm_infrastructure.comm_endpoint import CommEndpoint
from ark.client.comm_infrastructure.base_node import main
from ark.tools.log import log
from arktypes import joint_group_command_t, joint_state_t, pose_t, rigid_body_state_t, task_space_command_t
from arktypes.utils import pack, unpack


# =============================================================================
# PICK-AND-PLACE CONFIGURATION
# =============================================================================
# All position/offset parameters in one place for easy tuning.
# Distances in meters, angles in radians.

CONFIG = {
    # -------------------------------------------------------------------------
    # ROBOT / IK PARAMETERS
    # -------------------------------------------------------------------------
    # TCP offset: distance from link7 (wrist) to fingertips in link-local frame
    # This should match ee_tcp_offset in franka_newton.yaml
    "tcp_offset": [0.0, 0.0, 0.164],

    # Minimum TCP height the robot can reliably reach (empirically determined)
    # At x=0.45m, the Franka struggles below ~0.06m due to joint limits
    "min_reachable_tcp_z": 0.055,

    # Home position for the end-effector [x, y, z]
    "home_position": [0.4, 0.0, 0.4],

    # -------------------------------------------------------------------------
    # CUBE / PAYLOAD PARAMETERS (must match global_config_newton.yaml)
    # -------------------------------------------------------------------------
    # Cube center position [x, y, z] - this is the CENTER of the cube
    # 4cm cube on elevated surface: bottom at z=0.10, center at z=0.12, top at z=0.14
    "cube_position": [0.45, 0.0, 0.12],

    # Cube dimensions [x, y, z]
    "cube_size": [0.04, 0.04, 0.04],

    # -------------------------------------------------------------------------
    # GRASP STRATEGY PARAMETERS
    # -------------------------------------------------------------------------
    # Height above cube top for approach/retreat moves
    "approach_clearance": 0.15,

    # Grasp height relative to cube top (negative = below top, into cube)
    # 0.0 = grasp at cube top, -0.02 = grasp at cube center for 4cm cube
    # MUST BE NEGATIVE to grasp inside the cube!
    "grasp_depth_from_top": 0.02,  # 1cm below top (near upper edge for stable grasp)

    # Place position [x, y] - Z will match grasp height
    "place_xy": [0.30, 0.30],

    # -------------------------------------------------------------------------
    # GRIPPER PARAMETERS (finger joint positions, each finger)
    # Total gripper opening = 2 * finger_position
    # For a 4cm cube, the fingers physically stop at ~0.02m each (4cm total).
    # Setting target BELOW the physical stop creates squeeze pressure via PD controller.
    # Balance: enough pressure to grip, not so much it destabilizes simulation.
    # -------------------------------------------------------------------------
    "gripper_open": 0.04,       # Fully open: 8cm total (for approach)
    "gripper_closed": 0.008,    # Tighter grip: target 1.6cm total (firm squeeze on 4cm cube)
    "gripper_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],  # Franka gripper joints

    # -------------------------------------------------------------------------
    # CONTROL TOLERANCES
    # -------------------------------------------------------------------------
    "ee_tolerance": 0.015,           # Position error tolerance (m)
    "ee_resend_period": 0.01,        # Command resend interval (s)
    "gripper_tolerance": 0.002,      # Gripper position tolerance (m)
    "gripper_resend_period": 0.05,   # Gripper command resend (s)
    "gripper_settle_time": 0.75,     # Wait after gripper reaches target (s)

    # -------------------------------------------------------------------------
    # ORIENTATION
    # -------------------------------------------------------------------------
    # Grasp orientation quaternion [x, y, z, w]
    # This points the gripper straight down (180 deg rotation around Y)
    "grasp_orientation": [0.0, 1.0, 0.0, 0.0],
}

# Derived values (computed from CONFIG)
def _compute_derived():
    """Compute derived values from CONFIG."""
    cube_pos = np.array(CONFIG["cube_position"])
    cube_size = np.array(CONFIG["cube_size"])

    CONFIG["cube_top_z"] = cube_pos[2] + 0.5 * cube_size[2]
    CONFIG["cube_bottom_z"] = cube_pos[2] - 0.5 * cube_size[2]

    # Grasp Z = cube_top + grasp_depth (grasp_depth is typically negative)
    CONFIG["grasp_z"] = CONFIG["cube_top_z"] + CONFIG["grasp_depth_from_top"]

    # Approach Z = cube_top + clearance
    CONFIG["approach_z"] = CONFIG["cube_top_z"] + CONFIG["approach_clearance"]

    # Full pick/place positions
    CONFIG["pick_position"] = [cube_pos[0], cube_pos[1], CONFIG["grasp_z"]]
    CONFIG["pick_above"] = [cube_pos[0], cube_pos[1], CONFIG["approach_z"]]
    CONFIG["place_position"] = [CONFIG["place_xy"][0], CONFIG["place_xy"][1], CONFIG["grasp_z"]]
    CONFIG["place_above"] = [CONFIG["place_xy"][0], CONFIG["place_xy"][1], CONFIG["approach_z"]]

_compute_derived()


def _print_config():
    """Print configuration summary for diagnostics."""
    log.info("=" * 60)
    log.info("PICK-AND-PLACE CONFIGURATION")
    log.info("=" * 60)
    log.info(f"  Cube center:     [{CONFIG['cube_position'][0]:.3f}, {CONFIG['cube_position'][1]:.3f}, {CONFIG['cube_position'][2]:.3f}]")
    log.info(f"  Cube top Z:      {CONFIG['cube_top_z']:.3f}m")
    log.info(f"  Grasp Z:         {CONFIG['grasp_z']:.3f}m (depth from top: {CONFIG['grasp_depth_from_top']:.3f}m)")
    log.info(f"  Approach Z:      {CONFIG['approach_z']:.3f}m")
    log.info(f"  Min reachable:   {CONFIG['min_reachable_tcp_z']:.3f}m")
    log.info(f"  TCP offset:      {CONFIG['tcp_offset']}")
    log.info(f"  Home position:   {CONFIG['home_position']}")
    log.info("=" * 60)


# =============================================================================


class PickPlaceNode(CommEndpoint):
    """ARK node that executes pick-and-place via Cartesian commands."""

    @staticmethod
    def _quat_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
        q = np.array(q_xyzw, dtype=float).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm <= 0.0:
            return np.eye(3)
        x, y, z, w = (q / norm).tolist()

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    @staticmethod
    def _rotmat_to_quat(R: np.ndarray) -> list[float]:
        """Convert a 3x3 rotation matrix to quaternion (xyzw)."""
        m = np.array(R, dtype=float).reshape(3, 3)
        tr = float(m[0, 0] + m[1, 1] + m[2, 2])

        if tr > 0.0:
            S = float(np.sqrt(tr + 1.0) * 2.0)
            w = 0.25 * S
            x = (m[2, 1] - m[1, 2]) / S
            y = (m[0, 2] - m[2, 0]) / S
            z = (m[1, 0] - m[0, 1]) / S
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0)
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0)
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0)
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S

        q = np.array([x, y, z, w], dtype=float)
        norm = float(np.linalg.norm(q))
        if norm > 0.0:
            q /= norm
        return q.tolist()

    @classmethod
    def _vertical_quat_preserve_yaw(cls, current_xyzw: np.ndarray) -> list[float]:
        """Build a 'vertical gripper' quaternion (xyzw) preserving current yaw.

        Assumes the EE local +Z axis is the gripper approach direction, and enforces
        it to point along world -Z while keeping the projection of the local +X axis
        on the XY plane (i.e., yaw) consistent with the current pose.
        """
        R = cls._quat_to_rotmat(current_xyzw)
        x_world = R[:, 0]

        z_des = np.array([0.0, 0.0, -1.0], dtype=float)
        x_proj = np.array([float(x_world[0]), float(x_world[1]), 0.0], dtype=float)
        x_norm = float(np.linalg.norm(x_proj))
        if x_norm < 1e-6:
            x_proj = np.array([1.0, 0.0, 0.0], dtype=float)
            x_norm = 1.0
        x_des = x_proj / x_norm

        y_des = np.cross(z_des, x_des)
        y_norm = float(np.linalg.norm(y_des))
        if y_norm < 1e-6:
            # Degenerate; pick an arbitrary yaw axis.
            x_des = np.array([0.0, 1.0, 0.0], dtype=float)
            y_des = np.cross(z_des, x_des)
            y_norm = float(np.linalg.norm(y_des))
        y_des /= y_norm

        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack([x_des, y_des, z_des])
        return cls._rotmat_to_quat(R_des)

    @staticmethod
    def _quat_angle_error(current: np.ndarray, target: np.ndarray) -> float:
        """Return the smallest angle (rad) between two quaternions (xyzw)."""
        qc = np.array(current, dtype=float).reshape(4)
        qt = np.array(target, dtype=float).reshape(4)

        qc_norm = float(np.linalg.norm(qc))
        qt_norm = float(np.linalg.norm(qt))
        if qc_norm > 0.0:
            qc /= qc_norm
        if qt_norm > 0.0:
            qt /= qt_norm

        dot = abs(float(np.dot(qc, qt)))
        dot = float(np.clip(dot, -1.0, 1.0))
        return 2.0 * float(np.arccos(dot))

    def __init__(self, global_config, robot_name: str, namespace: str):
        super().__init__("pick_place_controller", global_config)

        self.robot_name = robot_name
        self.namespace = namespace.strip("/")
        self._active_prefix: Optional[str] = None
        self._payload_prefix: Optional[str] = None

        log.info("Initializing pick-and-place controller (v2)...")

        self._joint_state_subs = []
        self._ee_state_subs = []
        self._arm_pubs = []
        self._cart_pubs = []
        self._payload_state_subs = []

        prefixes = [f"{self.namespace}/{self.robot_name}", self.robot_name]
        prefixes = list(dict.fromkeys(prefixes))  # de-dupe

        for prefix in prefixes:
            joint_state_ch = f"{prefix}/joint_states/sim"
            ee_state_ch = f"{prefix}/ee_state/sim"
            joint_cmd_ch = f"{prefix}/joint_group_command/sim"
            cart_cmd_ch = f"{prefix}/cartesian_command/sim"

            self._arm_pubs.append(self.create_publisher(joint_cmd_ch, joint_group_command_t))
            self._cart_pubs.append(self.create_publisher(cart_cmd_ch, task_space_command_t))

            self._joint_state_subs.append(
                self.create_subscriber(
                    joint_state_ch,
                    joint_state_t,
                    lambda t, ch, msg, p=prefix: self._joint_state_callback(p, t, ch, msg),
                )
            )
            self._ee_state_subs.append(
                self.create_subscriber(
                    ee_state_ch,
                    pose_t,
                    lambda t, ch, msg, p=prefix: self._ee_state_callback(p, t, ch, msg),
                )
            )

        self.current_joint_positions = None
        self.current_ee_pose = None
        self.current_payload_state = None
        self.payload_initial_center = None
        self.sequence_done = False
        self.sequence_success = False
        self.last_gripper = 0.04

        self.ee_max_step = 0.03
        # NOTE: FrankaNewtonDriver applies per-command joint step limiting, so Cartesian
        # commands must be streamed at a reasonably high rate to achieve smooth motion.
        # All tolerances and parameters from CONFIG at top of file
        self.ee_tolerance = CONFIG["ee_tolerance"]
        self.ee_resend_period = CONFIG["ee_resend_period"]

        self.gripper_tolerance = CONFIG["gripper_tolerance"]
        self.gripper_resend_period = CONFIG["gripper_resend_period"]
        self.gripper_settle_seconds = CONFIG["gripper_settle_time"]

        # Grasp orientation from CONFIG
        self.grasp_orientation = CONFIG["grasp_orientation"]

        # Payload (cube) configuration from CONFIG
        self.payload_center = np.array(CONFIG["cube_position"], dtype=float)
        self.payload_size = np.array(CONFIG["cube_size"], dtype=float)
        self.payload_top_z = CONFIG["cube_top_z"]

        # Subscribe to payload ground truth (diagnostics + adaptive pick pose)
        payload_name = "payload"
        payload_prefixes = [f"{self.namespace}/{payload_name}", payload_name]
        payload_prefixes = list(dict.fromkeys(payload_prefixes))
        for prefix in payload_prefixes:
            payload_ch = f"{prefix}/ground_truth/sim"
            self._payload_state_subs.append(
                self.create_subscriber(
                    payload_ch,
                    rigid_body_state_t,
                    lambda t, ch, msg, p=prefix: self._payload_state_callback(p, t, ch, msg),
                )
            )

        self.create_stepper(1.0, self.start_sequence, oneshot=True)
        log.ok("Controller initialized. Waiting for joint states...")

    def _set_active_prefix(self, prefix: str) -> None:
        if self._active_prefix is None:
            self._active_prefix = prefix
            log.ok(f"Detected active channel prefix: '{prefix}'")

    def _joint_state_callback(self, prefix: str, _t, _ch, msg):
        self._set_active_prefix(prefix)
        self.current_joint_positions = dict(zip(msg.name, msg.position))

    def _ee_state_callback(self, prefix: str, _t, _ch, msg):
        if self._active_prefix is None:
            self._set_active_prefix(prefix)
        position, orientation = unpack.pose(msg)
        self.current_ee_pose = {
            "position": position,
            "orientation": orientation,
        }

    def _payload_state_callback(self, prefix: str, _t, _ch, msg):
        if self._payload_prefix is None:
            self._payload_prefix = prefix
            log.ok(f"Detected payload ground truth prefix: '{prefix}'")
        self.current_payload_state = msg

    def _get_payload_position(self) -> Optional[np.ndarray]:
        if self.current_payload_state is None:
            return None
        try:
            return np.array(self.current_payload_state.position, dtype=float)
        except Exception:  # noqa: BLE001
            return None

    def _log_payload_state(self, tag: str) -> None:
        pos = self._get_payload_position()
        if pos is None:
            log.warning(f"[{tag}] Payload ground truth unavailable")
            return
        dz = None
        if self.payload_initial_center is not None:
            dz = float(pos[2] - self.payload_initial_center[2])
        if dz is None:
            log.info(f"[{tag}] Payload pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        else:
            log.info(
                f"[{tag}] Payload pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] (dz={dz:+.3f}m)"
            )

    def _get_gripper_positions(self) -> Optional[np.ndarray]:
        if self.current_joint_positions is None:
            return None
        try:
            gripper_joints = CONFIG.get("gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"])
            return np.array(
                [
                    float(self.current_joint_positions.get(joint_name, 0.0))
                    for joint_name in gripper_joints
                ],
                dtype=float,
            )
        except Exception:  # noqa: BLE001
            return None

    def wait_for_gripper(
        self,
        target: list[float],
        tolerance: float | None = None,
        resend: list[float] | None = None,
        name: str = "gripper",
        settle_seconds: float | None = None,
    ) -> None:
        """Block until the gripper reaches target OR settles (e.g., contacts an object).

        This is intentionally non-failing: it keeps resending commands and proceeds once
        the gripper stops moving for `settle_seconds`, even if the exact target isn't
        reached due to object contact.
        """

        if tolerance is None:
            tolerance = self.gripper_tolerance
        if settle_seconds is None:
            settle_seconds = self.gripper_settle_seconds

        target_np = np.array(target[:2], dtype=float)
        last = None
        last_change_t = time.time()
        next_resend = 0.0
        last_log = 0.0

        while not self._done:
            now = time.time()
            if resend is not None and now >= next_resend:
                self.send_gripper_command(resend)
                next_resend = now + self.gripper_resend_period

            current = self._get_gripper_positions()
            if current is None:
                time.sleep(0.01)
                continue

            max_err = float(np.max(np.abs(current - target_np)))
            if max_err <= tolerance:
                return

            if last is not None:
                moved = float(np.max(np.abs(current - last)))
                if moved > 1e-5:
                    last_change_t = now
            last = current

            if now - last_change_t >= settle_seconds:
                log.info(
                    "Gripper settled for '%s' (max_err=%.4f m). Proceeding.",
                    name,
                    max_err,
                )
                return

            if now - last_log >= 1.0:
                log.info("Gripper error: %.4f m (target=%s)", max_err, target[:2])
                last_log = now

            time.sleep(0.01)

    def _active_publishers(self, pub_list):
        if self._active_prefix is None:
            return pub_list
        if self._active_prefix.startswith(self.namespace):
            return [pub_list[0]]
        return [pub_list[-1]]

    def send_gripper_command(self, positions: list[float]):
        msg = joint_group_command_t()
        msg.name = "gripper"
        msg.n = 2
        msg.cmd = positions
        for pub in self._active_publishers(self._arm_pubs):
            pub.publish(msg)
        if positions:
            self.last_gripper = float(positions[0])

    def send_cartesian_command(
        self,
        position: list[float],
        quaternion: list[float],
        gripper: float | None = None,
    ) -> None:
        if gripper is None:
            gripper = self.last_gripper
        msg = pack.task_space_command(
            "all",
            np.array(position, dtype=float),
            np.array(quaternion, dtype=float),
            float(gripper),
        )
        for pub in self._active_publishers(self._cart_pubs):
            pub.publish(msg)

    def move_to_ee_position(self, pos: list[float], name: str, use_grasp_orientation: bool = False) -> None:
        log.info(f"Moving EE to {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        fixed_quat: list[float]
        if self.current_ee_pose and self.current_ee_pose.get("orientation") is not None:
            fixed_quat = self.current_ee_pose["orientation"].tolist()
        else:
            fixed_quat = [0.0, 0.0, 0.0, 1.0]
        target = np.array(pos, dtype=float)
        last_log = 0.0

        while not self._done:
            if self.current_ee_pose is None or self.current_ee_pose.get("position") is None:
                time.sleep(self.ee_resend_period)
                continue

            current = np.array(self.current_ee_pose["position"], dtype=float)
            delta = target - current
            dist = float(np.linalg.norm(delta))

            # Keep gripper vertical without locking yaw: recompute the vertical target
            # from the *current* yaw each resend cycle.
            if use_grasp_orientation and self.current_ee_pose.get("orientation") is not None:
                quat = self._vertical_quat_preserve_yaw(self.current_ee_pose["orientation"])
            elif use_grasp_orientation:
                quat = self.grasp_orientation
            else:
                quat = fixed_quat

            if dist <= self.ee_tolerance:
                self.send_cartesian_command(pos, quat)
                return

            # Always command the final target pose (not incremental waypoints) so the
            # IK solver produces a strong, consistent joint-space target and the
            # driver-side joint-step limiter can do the smoothing.
            self.send_cartesian_command(pos, quat)

            now = time.time()
            if now - last_log >= 1.0:
                rot_err_deg = None
                if self.current_ee_pose.get("orientation") is not None:
                    try:
                        rot_err = self._quat_angle_error(
                            np.array(self.current_ee_pose["orientation"], dtype=float),
                            np.array(quat, dtype=float),
                        )
                        rot_err_deg = float(np.degrees(rot_err))
                    except Exception:  # noqa: BLE001
                        rot_err_deg = None

                extra = ""
                if rot_err_deg is not None and np.isfinite(rot_err_deg):
                    extra = f", rot_err={rot_err_deg:.1f} deg"
                log.info(
                    "EE progress to '%s': current=[%.3f, %.3f, %.3f] err=%.4f m%s",
                    name,
                    float(current[0]),
                    float(current[1]),
                    float(current[2]),
                    dist,
                    extra,
                )
                last_log = now

            time.sleep(self.ee_resend_period)

    def start_sequence(self):
        # Print configuration summary at sequence start
        _print_config()

        log.ok("=" * 60)
        log.ok("Starting Pick-and-Place Sequence (EE Control)")
        log.ok("=" * 60)

        log.info("Waiting for robot state...")
        last_log = 0.0
        while self.current_joint_positions is None and not self._done:
            now = time.time()
            if now - last_log >= 1.0:
                log.info("Waiting for joint states...")
                last_log = now
            if self._done:
                break
            time.sleep(0.1)

        if self.current_joint_positions is None:
            log.error("Failed to receive joint states. Is simulator running?")
            self.sequence_success = False
            self.sequence_done = True
            return

        # Optional: wait briefly for payload ground truth, then use it to seed pick pose
        log.info("Waiting for payload ground truth (up to 5s)...")
        payload_wait_start = time.time()
        while self.current_payload_state is None and time.time() - payload_wait_start < 5.0:
            if self._done:
                break
            time.sleep(0.1)

        payload_pos = self._get_payload_position()
        if payload_pos is not None:
            # Update CONFIG with actual cube position from simulation
            CONFIG["cube_position"] = payload_pos.tolist()
            _compute_derived()  # Recompute all derived positions
            self.payload_center = payload_pos
            self.payload_top_z = CONFIG["cube_top_z"]
            self.payload_initial_center = payload_pos.copy()
            log.ok(
                f"Using payload ground truth: center=[{payload_pos[0]:.3f}, {payload_pos[1]:.3f}, {payload_pos[2]:.3f}]"
            )
        else:
            # Fall back to config-derived nominal cube position
            self.payload_initial_center = self.payload_center.copy()
            log.warning(
                "Payload ground truth not received; using configured payload_center="
                f"[{self.payload_center[0]:.3f}, {self.payload_center[1]:.3f}, {self.payload_center[2]:.3f}]"
            )

        log.ok("Robot state received. Starting motion...")

        # Gripper positions from CONFIG (each value is per-finger, total = 2x)
        # With hydroelastic contacts, fingers will stop when they contact the cube
        GRIPPER_OPEN = [CONFIG["gripper_open"]] * 2
        GRIPPER_CLOSED = [CONFIG["gripper_closed"]] * 2

        # All positions from CONFIG (pre-computed in _compute_derived)
        home_pos = CONFIG["home_position"]
        pick_pos = CONFIG["pick_position"]
        pick_above = CONFIG["pick_above"]
        place_pos = CONFIG["place_position"]
        place_above = CONFIG["place_above"]

        # Log computed positions
        log.info(f"Cube: center z={CONFIG['cube_position'][2]:.3f}m, top z={CONFIG['cube_top_z']:.3f}m")
        log.info(f"Pick: grasp_z={CONFIG['grasp_z']:.3f}m, approach_z={CONFIG['approach_z']:.3f}m")
        log.info(f"Place: place_z={CONFIG['grasp_z']:.3f}m")
        log.info(f"Min reachable TCP z: {CONFIG['min_reachable_tcp_z']:.3f}m")

        # Warn if grasp height is below minimum reachable
        if CONFIG["grasp_z"] < CONFIG["min_reachable_tcp_z"]:
            log.warning(
                f"Grasp z={CONFIG['grasp_z']:.3f}m is BELOW min reachable z={CONFIG['min_reachable_tcp_z']:.3f}m! "
                f"Adjust cube_position or grasp_depth_from_top in CONFIG."
            )

        # Sequence format: (action_type, data, name, use_grasp_orientation)
        # use_grasp_orientation is optional, defaults to False
        # NOTE: Keep gripper vertical by commanding a fixed EE orientation on all moves.
        # Progression is state-based: each step blocks until the robot/gripper reaches target.
        # With hydroelastic contacts, the gripper will physically stop when contacting cube.
        sequence = [
            ("ee", home_pos, "Home", True),
            ("gripper", GRIPPER_OPEN, "Open gripper"),
            ("ee", pick_above, "Approach above cube", True),
            ("ee", pick_pos, "Lower to cube", True),
            ("gripper", GRIPPER_CLOSED, "Grasp cube"),  # Single step - hydroelastic stops at contact
            ("ee", pick_above, "Lift cube", True),
            ("ee", place_above, "Transport", True),
            ("ee", place_pos, "Lower to place", True),
            ("gripper", GRIPPER_OPEN, "Release cube"),
            ("ee", place_above, "Retreat", True),
            ("ee", home_pos, "Home", True),
        ]

        for idx, step in enumerate(sequence, 1):
            action_type = step[0]
            data = step[1]
            name = step[2]
            use_grasp = step[3] if len(step) > 3 else False
            log.info(f"\n[{idx}/{len(sequence)}] >> {name}")
            if action_type == "ee":
                self.move_to_ee_position(data, name, use_grasp_orientation=use_grasp)
            elif action_type == "gripper":
                self.send_gripper_command(data)
                self.wait_for_gripper(data, resend=data, name=name)

            # Key diagnostics: track whether the cube actually lifts and moves.
            if name in {
                "Grasp cube",
                "Lift cube",
                "Release cube",
            }:
                self._log_payload_state(name)

            if name == "Lift cube" and self.payload_initial_center is not None:
                pos = self._get_payload_position()
                if pos is not None:
                    dz = float(pos[2] - self.payload_initial_center[2])
                    if dz < 0.03:
                        log.warning(
                            f"[Lift check] Payload did not lift as expected (dz={dz:+.3f}m). "
                            "If MuJoCo solver is active, check robot_shape.ke/kd + collision_pipeline settings."
                        )
                    else:
                        log.ok(f"[Lift check] Payload lifted (dz={dz:+.3f}m)")

        log.ok("=" * 60)
        log.ok("Pick-and-Place Sequence Complete!")
        log.ok("=" * 60)
        self.sequence_success = True
        self.sequence_done = True

    def spin(self):
        log.info("Node spinning... (Ctrl+C to exit early)")
        while not self._done and not self.sequence_done:
            try:
                self._lcm.handle_timeout(100)
            except OSError as exc:
                log.warning(f"LCM threw OSError {exc}")
                self._done = True

        if self.sequence_done:
            if self.sequence_success:
                log.info("Sequence complete. Keeping node alive for observation...")
            else:
                log.warning("Sequence failed. Keeping node alive for observation...")
            log.info("Press Ctrl+C to exit...")
            try:
                while not self._done:
                    self._lcm.handle_timeout(100)
            except KeyboardInterrupt:
                log.info("Exiting...")
                self._done = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Newton pick-and-place test (v2)")
    parser.add_argument("--robot", default="panda", help="Robot name (default: panda)")
    parser.add_argument(
        "--namespace",
        default="ark",
        help="Namespace prefix (default: ark). Use empty string for none.",
    )
    parser.add_argument("--registry-host", default="127.0.0.1")
    parser.add_argument("--registry-port", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    namespace = args.namespace.strip("/")

    config = {
        "network": {
            "registry_host": args.registry_host,
            "registry_port": args.registry_port,
        }
    }

    main(PickPlaceNode, config, args.robot, namespace)

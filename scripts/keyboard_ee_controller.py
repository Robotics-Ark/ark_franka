#!/usr/bin/env python3
"""Keyboard controller for Franka end-effector position using Newton.

This controller sends task-space commands over ARK channels and avoids any
PyBullet-based IK. It auto-detects the active namespace (namespaced or legacy)
by subscribing to both sets of channels.

Usage:
    # Terminal 1: Start simulator
    python scripts/launch_ark_newton_simulator.py --no-logger

    # Terminal 2: Run keyboard controller
    python scripts/keyboard_ee_controller.py --namespace ark --robot panda

Controls:
    W/S     - Move EE forward/backward (X axis)
    A/D     - Move EE left/right (Y axis)
    Q/E     - Move EE up/down (Z axis)
    Z/X     - Rotate gripper yaw (about Z axis)
    C/V     - Rotate gripper roll (about X axis)
    B/N     - Rotate gripper pitch (about Y axis)
    G       - Toggle gripper open/close
    R       - Reset to home position
    ESC     - Quit
"""

from __future__ import annotations

import argparse
import math
import sys
import termios
import tty
import select
import time
from typing import Optional

import numpy as np

from ark.client.comm_infrastructure.comm_endpoint import CommEndpoint
from ark.client.comm_infrastructure.base_node import main
from ark.tools.log import log
from arktypes import joint_group_command_t, joint_state_t, pose_t, task_space_command_t
from arktypes.utils import pack, unpack


class KeyboardEEController(CommEndpoint):
    """Keyboard-controlled end-effector position controller."""

    # Franka Panda home configuration
    HOME_CONFIG = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    # End-effector position step size per key press (meters)
    STEP_SIZE = 0.015

    # End-effector rotation step size (radians per key press)
    ROT_STEP_RAD = 0.08

    # Command rate and step parameters
    COMMAND_RATE_HZ = 50.0
    MAX_EE_STEP = 0.05
    EE_TARGET_TOL = 0.005

    def __init__(
        self,
        global_config,
        robot_name: str = "panda",
        namespace: str = "ark",
    ):
        super().__init__("keyboard_ee_controller", global_config)

        log.info("Initializing keyboard end-effector controller...")

        self.robot_name = robot_name
        self.namespace = namespace.strip("/")
        self._active_prefix: Optional[str] = None

        # Publishers and subscribers (namespaced + legacy)
        self._arm_pubs_by_prefix: dict[str, object] = {}
        self._cart_pubs_by_prefix: dict[str, object] = {}
        self._joint_state_subs = []
        self._ee_state_subs = []

        prefixes = [f"{self.namespace}/{self.robot_name}", self.robot_name]
        prefixes = [p.strip("/") for p in prefixes if p]
        prefixes = list(dict.fromkeys(prefixes))

        for prefix in prefixes:
            joint_cmd_ch = f"{prefix}/joint_group_command/sim"
            joint_state_ch = f"{prefix}/joint_states/sim"
            cart_cmd_ch = f"{prefix}/cartesian_command/sim"
            ee_state_ch = f"{prefix}/ee_state/sim"

            self._arm_pubs_by_prefix[prefix] = self.create_publisher(
                joint_cmd_ch,
                joint_group_command_t,
            )
            self._cart_pubs_by_prefix[prefix] = self.create_publisher(
                cart_cmd_ch,
                task_space_command_t,
            )
            self._joint_state_subs.append(
                self.create_subscriber(
                    joint_state_ch,
                    joint_state_t,
                    lambda t, ch, msg, p=prefix: self.joint_state_callback(p, t, ch, msg),
                )
            )
            self._ee_state_subs.append(
                self.create_subscriber(
                    ee_state_ch,
                    pose_t,
                    lambda t, ch, msg, p=prefix: self.ee_state_callback(p, t, ch, msg),
                )
            )

        # Current state
        self.current_joint_positions: Optional[dict] = None
        self.current_ee_pose: Optional[dict] = None
        self.target_ee_pos = np.array([0.4, 0.0, 0.4], dtype=float)
        self.target_ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # xyzw
        self.last_gripper = 0.04
        self.gripper_open = True
        self._last_command_time = 0.0

        # Terminal settings for keyboard input
        self.old_settings = termios.tcgetattr(sys.stdin)

        log.ok("Controller initialized. Waiting for robot state...")
        log.info(
            "Controls: W/S=X, A/D=Y, Q/E=Z, Z/X=Yaw, C/V=Roll, B/N=Pitch, G=Gripper, R=Reset, ESC=Quit"
        )

    def _set_active_prefix(self, prefix: str) -> None:
        if self._active_prefix is None:
            self._active_prefix = prefix
            log.ok(f"Detected active channel prefix: '{prefix}'")

    def _active_arm_publishers(self):
        if self._active_prefix is None:
            return list(self._arm_pubs_by_prefix.values())
        active_pub = self._arm_pubs_by_prefix.get(self._active_prefix)
        if active_pub is None:
            return list(self._arm_pubs_by_prefix.values())
        return [active_pub]

    def _active_cart_publishers(self):
        if self._active_prefix is None:
            return list(self._cart_pubs_by_prefix.values())
        active_pub = self._cart_pubs_by_prefix.get(self._active_prefix)
        if active_pub is None:
            return list(self._cart_pubs_by_prefix.values())
        return [active_pub]

    def joint_state_callback(self, prefix: str, _t, _channel, msg):
        """Update current joint positions from robot state."""
        self._set_active_prefix(prefix)
        self.current_joint_positions = dict(zip(msg.name, msg.position))

    def ee_state_callback(self, prefix: str, _t, _channel, msg):
        """Update current end-effector pose from simulation state."""
        self._set_active_prefix(prefix)
        position, orientation = unpack.pose(msg)
        self.current_ee_pose = {
            "position": np.array(position, dtype=float),
            "orientation": np.array(orientation, dtype=float),
        }

    def send_joint_command(self, positions: list[float]):
        """Send joint position command to the arm group."""
        msg = joint_group_command_t()
        msg.name = "arm"
        msg.n = len(positions)
        msg.cmd = positions
        for pub in self._active_arm_publishers():
            pub.publish(msg)

    def send_gripper_command(self, positions: list[float]) -> None:
        msg = joint_group_command_t()
        msg.name = "gripper"
        msg.n = 2
        msg.cmd = positions
        for pub in self._active_arm_publishers():
            pub.publish(msg)
        if positions:
            self.last_gripper = float(positions[0])
            self.gripper_open = self.last_gripper > 0.02

    def send_cartesian_command(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper: float | None = None,
    ) -> None:
        if gripper is None:
            gripper = self.last_gripper
        msg = pack.task_space_command(
            "all",
            position.astype(float),
            quaternion.astype(float),
            float(gripper),
        )
        for pub in self._active_cart_publishers():
            pub.publish(msg)

    def _quat_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis.astype(float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        axis = axis / axis_norm
        half = angle * 0.5
        sin_half = math.sin(half)
        return np.array(
            [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, math.cos(half)],
            dtype=float,
        )

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )

    def _normalize_quat(self, q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm <= 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / norm

    def _quat_to_euler(self, q: np.ndarray) -> tuple[float, float, float]:
        """Convert quaternion (xyzw) to Euler angles (roll, pitch, yaw) in degrees."""
        x, y, z, w = q
        # Roll (X)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch (Y)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        # Yaw (Z)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def get_current_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.current_ee_pose is None:
            return self.target_ee_pos.copy(), self.target_ee_quat.copy()
        return self.current_ee_pose["position"].copy(), self.current_ee_pose["orientation"].copy()

    def _step_toward_target(self) -> None:
        """Advance toward the target EE position with bounded steps."""
        if self.current_ee_pose is None:
            return
        current_pos, _current_quat = self.get_current_ee_pose()
        delta = self.target_ee_pos - current_pos
        distance = float(np.linalg.norm(delta))

        if distance < self.EE_TARGET_TOL:
            # Already at position target - still send command for rotation updates
            self.send_cartesian_command(self.target_ee_pos, self.target_ee_quat)
            return

        step = min(self.MAX_EE_STEP, distance)
        next_target = current_pos + (delta / distance) * step
        self.send_cartesian_command(next_target, self.target_ee_quat)

    def get_key(self) -> Optional[str]:
        """Non-blocking keyboard input."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def spin(self):
        """Main control loop with keyboard input."""
        try:
            # Set terminal to raw mode for keyboard input
            tty.setraw(sys.stdin.fileno())

            log.info("\n" + "=" * 50)
            log.info("Keyboard EE Controller Active")
            log.info("=" * 50)
            print("\r\nControls:")
            print("\r\n  W/S - Move forward/backward (X)")
            print("\r\n  A/D - Move left/right (Y)")
            print("\r\n  Q/E - Move up/down (Z)")
            print("\r\n  Z/X - Rotate yaw (about Z)")
            print("\r\n  C/V - Rotate roll (about X)")
            print("\r\n  B/N - Rotate pitch (about Y)")
            print("\r\n  G   - Toggle gripper open/close")
            print("\r\n  R   - Reset to home")
            print("\r\n  ESC - Quit")
            print("\r\n" + "=" * 50)

            # Wait for initial EE state to avoid forcing a default orientation
            print("\r\nWaiting for robot state...", end="", flush=True)
            while self.current_ee_pose is None and not self._done:
                self._lcm.handle_timeout(100)

            current_pos, current_quat = self.get_current_ee_pose()
            self.target_ee_pos = current_pos
            self.target_ee_quat = current_quat
            roll, pitch, yaw = self._quat_to_euler(current_quat)
            print(
                f"\r\nInitial: pos=[{self.target_ee_pos[0]:.3f}, "
                f"{self.target_ee_pos[1]:.3f}, {self.target_ee_pos[2]:.3f}] "
                f"rot=[R:{roll:+.1f}, P:{pitch:+.1f}, Y:{yaw:+.1f}]"
            )

            last_print_time = 0

            while not self._done:
                # Handle LCM messages (short timeout for responsiveness)
                self._lcm.handle_timeout(5)

                # Check for keyboard input
                key = self.get_key()

                if key:
                    moved = False

                    if key == "\x1b":  # ESC
                        print("\r\nExiting...")
                        break
                    elif key.lower() == "w":
                        self.target_ee_pos[0] += self.STEP_SIZE
                        moved = True
                    elif key.lower() == "s":
                        self.target_ee_pos[0] -= self.STEP_SIZE
                        moved = True
                    elif key.lower() == "a":
                        self.target_ee_pos[1] += self.STEP_SIZE
                        moved = True
                    elif key.lower() == "d":
                        self.target_ee_pos[1] -= self.STEP_SIZE
                        moved = True
                    elif key.lower() == "q":
                        self.target_ee_pos[2] += self.STEP_SIZE
                        moved = True
                    elif key.lower() == "e":
                        self.target_ee_pos[2] -= self.STEP_SIZE
                        moved = True
                    elif key.lower() == "z":
                        delta = self._quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), self.ROT_STEP_RAD)
                        self.target_ee_quat = self._normalize_quat(
                            self._quat_multiply(delta, self.target_ee_quat)
                        )
                        moved = True
                    elif key.lower() == "x":
                        delta = self._quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -self.ROT_STEP_RAD)
                        self.target_ee_quat = self._normalize_quat(
                            self._quat_multiply(delta, self.target_ee_quat)
                        )
                        moved = True
                    elif key.lower() == "c":
                        delta = self._quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), self.ROT_STEP_RAD)
                        self.target_ee_quat = self._normalize_quat(
                            self._quat_multiply(delta, self.target_ee_quat)
                        )
                        moved = True
                    elif key.lower() == "v":
                        delta = self._quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -self.ROT_STEP_RAD)
                        self.target_ee_quat = self._normalize_quat(
                            self._quat_multiply(delta, self.target_ee_quat)
                        )
                        moved = True
                    elif key.lower() == "b":
                        delta = self._quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), self.ROT_STEP_RAD)
                        self.target_ee_quat = self._normalize_quat(
                            self._quat_multiply(delta, self.target_ee_quat)
                        )
                        moved = True
                    elif key.lower() == "n":
                        delta = self._quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -self.ROT_STEP_RAD)
                        self.target_ee_quat = self._normalize_quat(
                            self._quat_multiply(delta, self.target_ee_quat)
                        )
                        moved = True
                    elif key.lower() == "g":
                        if self.gripper_open:
                            self.send_gripper_command([0.005, 0.005])
                            print("\r\nGripper closed")
                        else:
                            self.send_gripper_command([0.04, 0.04])
                            print("\r\nGripper opened")
                    elif key.lower() == "r":
                        # Reset to home
                        self.send_joint_command(self.HOME_CONFIG)
                        print("\r\nReset to home configuration")
                        time.sleep(0.5)
                        current_pos, current_quat = self.get_current_ee_pose()
                        self.target_ee_pos = current_pos
                        self.target_ee_quat = current_quat
                        moved = False

                    if moved:
                        # Clamp to workspace bounds
                        self.target_ee_pos[0] = np.clip(self.target_ee_pos[0], 0.2, 0.8)
                        self.target_ee_pos[1] = np.clip(self.target_ee_pos[1], -0.5, 0.5)
                        self.target_ee_pos[2] = np.clip(self.target_ee_pos[2], 0.1, 0.8)

                        # Send command immediately on keypress for responsiveness
                        self.send_cartesian_command(self.target_ee_pos, self.target_ee_quat)
                        self._last_command_time = time.time()

                        roll, pitch, yaw = self._quat_to_euler(self.target_ee_quat)
                        print(
                            f"\r\nTarget: pos=[{self.target_ee_pos[0]:.3f}, "
                            f"{self.target_ee_pos[1]:.3f}, {self.target_ee_pos[2]:.3f}] "
                            f"rot=[R:{roll:+.1f}, P:{pitch:+.1f}, Y:{yaw:+.1f}]  ",
                            end="",
                            flush=True,
                        )

                now = time.time()
                if now - self._last_command_time >= 1.0 / self.COMMAND_RATE_HZ:
                    self._step_toward_target()
                    self._last_command_time = now

                # Periodic status update
                if now - last_print_time > 2.0:
                    current_pos, current_quat = self.get_current_ee_pose()
                    roll, pitch, yaw = self._quat_to_euler(current_quat)
                    print(
                        f"\r\nCurrent: pos=[{current_pos[0]:.3f}, "
                        f"{current_pos[1]:.3f}, {current_pos[2]:.3f}] "
                        f"rot=[R:{roll:+.1f}, P:{pitch:+.1f}, Y:{yaw:+.1f}]  ",
                        end="",
                        flush=True,
                    )
                    last_print_time = now

        except KeyboardInterrupt:
            print("\r\nInterrupted")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            print("\r\nController stopped.")


# Minimal config
CONFIG = {
    "network": {
        "registry_host": "127.0.0.1",
        "registry_port": 1234,
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyboard EE controller for ARK")
    parser.add_argument("--robot", default="panda", help="Robot name (default: panda)")
    parser.add_argument("--namespace", default="ark", help="Namespace prefix (default: ark)")
    args = parser.parse_args()

    main(KeyboardEEController, CONFIG, args.robot, args.namespace)

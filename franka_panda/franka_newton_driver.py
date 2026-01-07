"""Franka-specific driver for Newton physics backend."""

from typing import Any, Sequence

import numpy as np
import warp as wp
import newton.ik as ik

from ark.system.newton.newton_robot_driver import NewtonRobotDriver
from ark.tools.log import log


class FrankaNewtonDriver(NewtonRobotDriver):
    """Newton driver with Franka-specific features.

    Extends the generic NewtonRobotDriver to add Franka Panda specific
    functionality such as end-effector pose tracking and gripper control.
    """

    def _resolve_end_effector_idx(self, override: int | None = None) -> int:
        if override is not None:
            return int(override)
        for key in ("end_effector_idx", "ee_index", "ee_link_index"):
            if key in self.config:
                try:
                    return int(self.config[key])
                except (TypeError, ValueError):
                    break
        return 6

    def _resolve_group_joints(self, groups: dict[str, Any], name: str) -> list[str]:
        group = groups.get(name)
        if isinstance(group, dict):
            joints = group.get("joints", [])
            if isinstance(joints, dict):
                return list(joints.keys())
            if isinstance(joints, (list, tuple)):
                return list(joints)
            return []
        if isinstance(group, (list, tuple)):
            return list(group)
        return []

    def _infer_joint_groups(self) -> dict[str, dict[str, Any]]:
        if not self._joint_names:
            return {}

        def is_actuated(joint_name: str) -> bool:
            idx = self._joint_index_map.get(joint_name)
            if idx is None or self._joint_q_start is None:
                return True
            if idx >= len(self._joint_q_start) - 1:
                return False
            return int(self._joint_q_start[idx + 1]) > int(self._joint_q_start[idx])

        def is_gripper_joint(joint_name: str) -> bool:
            lowered = joint_name.lower()
            return any(token in lowered for token in ("finger", "gripper", "grip", "jaw"))

        actuation_joints = [name for name in self._joint_names if is_actuated(name)]
        gripper_joints = [name for name in actuation_joints if is_gripper_joint(name)]
        arm_joints = [name for name in actuation_joints if name not in gripper_joints]

        groups: dict[str, dict[str, Any]] = {
            "all": {"control_mode": "position", "joints": actuation_joints},
        }
        if arm_joints:
            groups["arm"] = {"control_mode": "position", "joints": arm_joints}
        if gripper_joints:
            groups["gripper"] = {"control_mode": "position", "joints": gripper_joints}
        return groups

    def _resolve_joint_groups(self) -> dict[str, dict[str, Any]]:
        groups = self.config.get("joint_groups")
        if isinstance(groups, dict) and groups:
            return groups
        if not hasattr(self, "_auto_joint_groups"):
            self._auto_joint_groups = self._infer_joint_groups()
        return self._auto_joint_groups

    def get_ee_pose(self) -> dict[str, Any]:
        """Get end-effector pose using Newton ArticulationView.

        Uses ArticulationView.get_link_transforms() which is more robust than
        directly accessing state.body_q (which can fail with CUDA-GL interop issues).

        Returns:
            Dictionary with 'position' (list of 3 floats in meters) and
            'orientation' (quaternion as list of 4 floats [x, y, z, w]).
        """
        default_pose = {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        }

        if self._articulation_view is None:
            return default_pose

        state = self._state_accessor()
        if state is None:
            return default_pose

        ee_link_index = self._resolve_end_effector_idx()

        try:
            # Use ArticulationView to get link transforms - more robust than body_q
            link_transforms = self._articulation_view.get_link_transforms(state)
            if link_transforms is None:
                return default_pose

            # link_transforms shape: (num_articulations, num_links, 7)
            # where 7 = [px, py, pz, qx, qy, qz, qw]
            transforms_np = link_transforms.numpy()

            if transforms_np.shape[0] == 0 or ee_link_index >= transforms_np.shape[1]:
                return default_pose

            # Get EE transform for first articulation (index 0)
            ee_transform = transforms_np[0, ee_link_index]

            return {
                "position": ee_transform[:3].tolist(),      # x, y, z position
                "orientation": ee_transform[3:].tolist()    # qx, qy, qz, qw quaternion
            }

        except Exception as e:
            log.warning(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"Failed to get EE pose: {e}"
            )
            return default_pose

    def _ensure_ik_solver(self, end_effector_idx: int) -> bool:
        if self._model is None:
            log.warning("FrankaNewtonDriver: IK unavailable - model not initialized yet")
            return False

        if getattr(self, "_ik_solver", None) is not None and getattr(self, "_ik_end_effector_idx", None) == end_effector_idx:
            return True

        device = self._model.device
        self._ik_end_effector_idx = end_effector_idx

        self._ik_target_pos = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        self._ik_target_rot = wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=device)

        self._ik_pos_obj = ik.IKPositionObjective(
            link_index=end_effector_idx,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=self._ik_target_pos,
        )
        self._ik_rot_obj = ik.IKRotationObjective(
            link_index=end_effector_idx,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self._ik_target_rot,
        )
        self._ik_limit_obj = ik.IKJointLimitObjective(
            joint_limit_lower=self._model.joint_limit_lower,
            joint_limit_upper=self._model.joint_limit_upper,
            weight=10.0,
        )

        self._ik_joint_q_in = wp.zeros((1, self._model.joint_coord_count), dtype=wp.float32, device=device)
        self._ik_joint_q_out = wp.zeros((1, self._model.joint_coord_count), dtype=wp.float32, device=device)

        self._ik_solver = ik.IKSolver(
            model=self._model,
            n_problems=1,
            objectives=[self._ik_pos_obj, self._ik_rot_obj, self._ik_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )
        return True

    def pass_joint_group_control_cmd(
        self,
        control_mode: str,
        cmd: dict[str, float | Sequence[float]],
        **kwargs: Any,
    ) -> None:
        """Send joint group command with Franka gripper mirroring.

        The Franka gripper has two finger joints (panda_finger_joint1 and
        panda_finger_joint2) that should move symmetrically. This method
        duplicates the command for finger joint 1 to finger joint 2.

        Args:
            control_mode: Control mode string ('position', 'velocity', 'torque')
            cmd: Dictionary mapping joint names to command values
            **kwargs: Additional arguments passed to parent method
        """
        # Duplicate finger joint command for symmetric gripper motion
        if "panda_finger_joint1" in cmd:
            panda_finger_joint1 = cmd.get("panda_finger_joint1")
            cmd["panda_finger_joint2"] = panda_finger_joint1

        # Call parent implementation
        super().pass_joint_group_control_cmd(control_mode, cmd, **kwargs)

    def pass_cartesian_control_cmd(self, control_mode: str, position, quaternion, **kwargs) -> None:
        """Send a Cartesian control command by computing inverse kinematics."""
        if not hasattr(self, "_cart_cmd_log_count"):
            self._cart_cmd_log_count = 0
        self._cart_cmd_log_count += 1
        if self._cart_cmd_log_count <= 3 or self._cart_cmd_log_count % 100 == 0:
            log.debug(
                f"FrankaNewtonDriver: cartesian cmd mode={control_mode}, "
                f"pos={position}, quat={quaternion}, kwargs={kwargs}"
            )
        if control_mode.lower() != "position":
            log.warning("FrankaNewtonDriver: Cartesian control only supports position mode")
            return

        if not (len(position) == 3 and len(quaternion) == 4):
            log.warning("FrankaNewtonDriver: position must be 3 elements and quaternion must be 4 elements")
            return

        end_effector_idx = self._resolve_end_effector_idx(kwargs.get("end_effector_idx"))
        if not self._ensure_ik_solver(end_effector_idx):
            return

        self._ik_pos_obj.set_target_position(0, wp.vec3(*position))
        self._ik_rot_obj.set_target_rotation(0, wp.vec4(*quaternion))

        state = self._state_accessor()
        current_joint_q = None
        if state is not None and state.joint_q is not None:
            current_joint_q = state.joint_q.numpy()[: self._model.joint_coord_count]
            self._ik_joint_q_in.assign(np.array([current_joint_q], dtype=np.float32))

        self._ik_solver.step(self._ik_joint_q_in, self._ik_joint_q_out, iterations=24)

        joint_positions = self._ik_joint_q_out.numpy()[0]
        control_hz = float(self.config.get("frequency", 240.0))
        control_dt = 1.0 / control_hz if control_hz > 0 else 1.0 / 240.0
        max_vel = self.config.get("cartesian_max_joint_velocity", None)
        max_step = self.config.get("cartesian_max_joint_step", None)
        smoothing = self.config.get("cartesian_smoothing", 1.0)

        try:
            max_vel = float(max_vel) if max_vel is not None else None
        except (TypeError, ValueError):
            max_vel = None
        try:
            max_step = float(max_step) if max_step is not None else None
        except (TypeError, ValueError):
            max_step = None
        try:
            smoothing = float(smoothing)
        except (TypeError, ValueError):
            smoothing = 1.0

        if smoothing <= 0.0 or smoothing > 1.0:
            smoothing = 1.0

        if max_vel is not None and max_vel > 0.0:
            vel_step = max_vel * control_dt
            if max_step is None or max_step <= 0.0:
                max_step = vel_step
            else:
                max_step = min(max_step, vel_step)
        joint_groups = self._resolve_joint_groups()
        arm_joints = self._resolve_group_joints(joint_groups, "arm")
        if not arm_joints:
            arm_joints = self._resolve_group_joints(joint_groups, "all")
        if not arm_joints:
            log.warning("FrankaNewtonDriver: No arm joint group configured; cannot apply IK solution")
            return

        cmd = {}
        for i, joint_name in enumerate(arm_joints):
            model_idx = self._joint_index_map.get(joint_name)
            if model_idx is None or self._joint_q_start is None:
                if i >= len(joint_positions):
                    break
                coord_idx = i
            else:
                if model_idx >= len(self._joint_q_start) - 1:
                    continue
                start = int(self._joint_q_start[model_idx])
                end = int(self._joint_q_start[model_idx + 1])
                if end - start <= 0:
                    continue
                coord_idx = start

            if coord_idx >= len(joint_positions):
                continue

            target = float(joint_positions[coord_idx])
            if current_joint_q is not None and coord_idx < len(current_joint_q):
                current = float(current_joint_q[coord_idx])
                delta = target - current
                if smoothing < 1.0:
                    delta *= smoothing
                if max_step is not None and max_step > 0.0:
                    delta = float(np.clip(delta, -max_step, max_step))
                target = current + delta
            cmd[joint_name] = float(target)

        gripper = kwargs.get("gripper", None)
        if gripper is not None:
            gripper_joints = self._resolve_group_joints(joint_groups, "gripper")
            for joint_name in gripper_joints:
                cmd[joint_name] = float(gripper)

        self.pass_joint_group_control_cmd("position", cmd)

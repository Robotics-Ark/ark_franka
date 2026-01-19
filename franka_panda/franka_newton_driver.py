"""Franka-specific driver for Newton physics backend."""

import time
from typing import Any, Callable, Sequence

import newton
import numpy as np
import warp as wp
import newton.ik as ik

from ark.system.newton.newton_robot_driver import NewtonRobotDriver
from ark.system.newton.scene_adapters.base_adapter import _COLLIDE_FLAG
from ark.tools.log import log


class FrankaNewtonDriver(NewtonRobotDriver):
    """Newton driver with Franka-specific features including gripper.

    Extends the generic NewtonRobotDriver to add Franka Panda specific
    functionality such as end-effector pose tracking, gripper control,
    and hydroelastic finger contacts.
    """

    def __init__(
        self,
        component_name: str,
        component_config: dict[str, Any],
        builder: newton.ModelBuilder,
    ) -> None:
        # Initialize Franka-specific gripper state BEFORE super().__init__
        # (which calls _load_into_builder that uses these)
        self._finger_shape_indices: list[int] = []
        self._finger_body_indices: list[int] = []
        self._gripper_is_releasing: bool = False

        # Collision filtering state (managed internally, not via backend callback)
        self._collision_disabled: bool = False
        self._collision_reenable_time: float = 0.0  # time.time() when to re-enable
        self._collision_reenable_duration: float = float(
            component_config.get("collision_reenable_duration", 0.5)
        )

        # Call parent init (triggers _load_into_builder)
        super().__init__(component_name, component_config, builder)

        # Gripper release threshold (configurable via gripper_release_threshold in config)
        self._gripper_release_threshold: float = float(
            self.config.get("gripper_release_threshold", 0.03)
        )

    def _load_into_builder(self) -> None:
        """Load Franka URDF and configure hydroelastic finger contacts."""
        pre_body_count = self.builder.body_count
        super()._load_into_builder()

        # Configure hydroelastic specifically for Franka gripper fingers
        self._configure_hydroelastic_fingers(pre_body_count)

    def _configure_hydroelastic_fingers(self, pre_body_count: int) -> None:
        """Enable hydroelastic contacts only on Franka finger shapes.

        Franka Panda uses "finger" in body names (panda_leftfinger, panda_rightfinger).
        This follows Newton's panda_hydro example which enables hydroelastic only on
        fingers for better contact behavior during grasping.
        """
        # Franka-specific: only use "finger" pattern
        finger_patterns = ["finger"]
        finger_body_indices: set[int] = set()

        # Verbose logging (default False for production)
        verbose = bool(self.config.get("verbose", False))

        if verbose:
            log.info(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"Bodies loaded (indices {pre_body_count}+):"
            )

        for idx, name in enumerate(self.builder.body_key):
            if idx < pre_body_count:
                continue  # Skip bodies loaded before this robot
            name_lower = name.lower()
            is_finger = any(pattern in name_lower for pattern in finger_patterns)
            if is_finger:
                finger_body_indices.add(idx)
            if verbose:
                log.info(f"  Body {idx}: '{name}' -> finger={is_finger}")

        if not finger_body_indices:
            log.info(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"No finger bodies found for hydroelastic configuration"
            )
            return

        log.info(
            f"FrankaNewtonDriver '{self.component_name}': "
            f"Finger body indices: {sorted(finger_body_indices)}"
        )

        # Get shape flags and body associations
        if not hasattr(self.builder, "shape_flags") or not hasattr(
            self.builder, "shape_body"
        ):
            log.warning(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"Builder missing shape_flags/shape_body - cannot configure hydroelastic"
            )
            return

        # Enable HYDROELASTIC flag on finger shapes, disable on others
        try:
            hydroelastic_flag = newton.ShapeFlags.HYDROELASTIC
            log.info(
                f"FrankaNewtonDriver: HYDROELASTIC flag value = {hydroelastic_flag}"
            )
        except AttributeError:
            log.warning(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"newton.ShapeFlags.HYDROELASTIC not available in this Newton version"
            )
            return

        finger_shape_count = 0
        non_finger_shape_count = 0
        finger_shape_indices: list[int] = []

        if verbose:
            log.info(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"Configuring shapes (total {len(self.builder.shape_body)} shapes):"
            )

        for shape_idx, body_idx in enumerate(self.builder.shape_body):
            if body_idx < pre_body_count:
                continue  # Skip shapes from other robots/objects

            body_name = (
                self.builder.body_key[body_idx]
                if body_idx < len(self.builder.body_key)
                else f"body_{body_idx}"
            )
            flags_before = self.builder.shape_flags[shape_idx]

            if body_idx in finger_body_indices:
                # Enable hydroelastic on finger shapes
                self.builder.shape_flags[shape_idx] |= hydroelastic_flag
                finger_shape_count += 1
                finger_shape_indices.append(shape_idx)
                action = "ENABLE"
            else:
                # Disable hydroelastic on arm link shapes (save computation)
                self.builder.shape_flags[shape_idx] &= ~hydroelastic_flag
                non_finger_shape_count += 1
                action = "disable"

            if verbose:
                flags_after = self.builder.shape_flags[shape_idx]
                log.info(
                    f"  Shape {shape_idx}: body={body_idx} ({body_name}) "
                    f"flags {flags_before} -> {flags_after} [{action} hydroelastic]"
                )

        # Store finger indices for collision filtering during release
        self._finger_shape_indices = finger_shape_indices
        self._finger_body_indices = list(finger_body_indices)

        log.ok(
            f"FrankaNewtonDriver '{self.component_name}': "
            f"Hydroelastic enabled on {finger_shape_count} finger shapes, "
            f"disabled on {non_finger_shape_count} arm shapes"
        )

    def _on_gripper_release_internal(self, is_releasing: bool) -> None:
        """Handle gripper release by toggling finger collision flags.

        When the gripper opens to release an object, we temporarily disable
        collisions on finger shapes to prevent the object from "sticking".
        Collisions are automatically re-enabled after a configurable delay.

        This uses self._model.shape_flags to toggle the COLLIDE flag on finger shapes.
        The collision flag value is imported from base_adapter to ensure consistency
        across Newton versions.

        Args:
            is_releasing: True to disable collisions (gripper opening),
                         False to re-enable (gripper closing or timeout)
        """
        if not self._finger_body_indices or self._model is None:
            return

        # Find shapes belonging to finger bodies
        shape_body = self._model.shape_body.numpy()
        finger_shapes = [
            idx
            for idx, body in enumerate(shape_body)
            if body in self._finger_body_indices
        ]

        if not finger_shapes:
            return

        shape_flags = self._model.shape_flags.numpy().copy()

        if is_releasing:
            # Disable collisions on finger shapes
            # for idx in finger_shapes: # TODO
            #     shape_flags[idx] &= ~_COLLIDE_FLAG
            self._collision_disabled = True
            self._collision_reenable_time = (
                time.time() + self._collision_reenable_duration
            )
            log.info(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"Disabled finger collisions on {len(finger_shapes)} shapes "
                f"(re-enable in {self._collision_reenable_duration}s)"
            )
        else:
            # Re-enable collisions on finger shapes
            # for idx in finger_shapes: # TODO
            #     shape_flags[idx] |= _COLLIDE_FLAG
            self._collision_disabled = False
            log.info(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"Re-enabled finger collisions on {len(finger_shapes)} shapes"
            )

        self._model.shape_flags.assign(shape_flags)

    def get_finger_body_indices(self) -> list[int]:
        """Return list of body indices for gripper fingers."""
        return list(self._finger_body_indices)

    def _check_gripper_release(self, cmd: dict[str, float | Sequence[float]]) -> None:
        """Check if gripper is being released and trigger collision filtering.

        When the gripper opens past the release threshold, finger collisions are
        temporarily disabled to allow clean object release. This is handled
        internally without needing a backend callback.
        """
        # Find gripper joint values in command (Franka uses "finger" in joint names)
        gripper_value = None
        for joint_name, value in cmd.items():
            if "finger" in joint_name.lower():
                gripper_value = (
                    float(value) if not isinstance(value, Sequence) else float(value[0])
                )
                break

        if gripper_value is None:
            return

        # Detect release (opening beyond threshold)
        is_releasing = gripper_value >= self._gripper_release_threshold
        if is_releasing != self._gripper_is_releasing:
            self._gripper_is_releasing = is_releasing
            log.info(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"Gripper {'RELEASING' if is_releasing else 'GRASPING'} "
                f"(value={gripper_value:.4f}, threshold={self._gripper_release_threshold})"
            )
            self._on_gripper_release_internal(is_releasing)

    def _resolve_end_effector_idx(self, override: int | None = None) -> int:
        """Resolve the EE link index used for IK + state.

        This tries to be robust against config/Robot defaults that often point at
        `panda_link6` (pre-wrist), which would make the wrist joint appear "stuck"
        and prevent stable orientation control.
        """

        def _idx_from_config() -> int | None:
            for key in ("end_effector_idx", "ee_index", "ee_link_index"):
                if key in self.config:
                    try:
                        return int(self.config[key])
                    except (TypeError, ValueError):
                        return None
            return None

        def _auto_idx() -> int | None:
            view = getattr(self, "_articulation_view", None)
            names = getattr(view, "body_names", None)
            if not names:
                return None

            # Prefer the actual gripper/hand body, then link7, then the last non-finger body.
            lowered = [str(n).lower() for n in names]

            for i, n in enumerate(lowered):
                if "hand" in n and "finger" not in n:
                    return i
            for i, n in enumerate(lowered):
                if "link7" in n:
                    return i
            for i in range(len(lowered) - 1, -1, -1):
                if "finger" not in lowered[i]:
                    return i
            return None

        def _looks_like_wrong_ee(idx: int) -> bool:
            view = getattr(self, "_articulation_view", None)
            names = getattr(view, "body_names", None)
            if not names:
                return False
            if idx < 0 or idx >= len(names):
                return True
            name = str(names[idx]).lower()
            # Common failure: pointing to link6 (pre-wrist) or a finger link.
            if "finger" in name:
                return True
            if "link6" in name and ("link7" not in name and "hand" not in name):
                return True
            return False

        requested = int(override) if override is not None else (_idx_from_config() or 6)

        auto = _auto_idx()
        if auto is not None and _looks_like_wrong_ee(requested):
            view = getattr(self, "_articulation_view", None)
            names = getattr(view, "body_names", None) or []
            requested_name = (
                str(names[requested])
                if 0 <= requested < len(names)
                else "<out-of-range>"
            )
            auto_name = str(names[auto]) if 0 <= auto < len(names) else "<unknown>"
            log.warning(
                "FrankaNewtonDriver: EE link index %s (%s) looks incorrect; using auto-detected %s (%s). "
                "Set 'end_effector_idx' in the robot config to silence this.",
                requested,
                requested_name,
                auto,
                auto_name,
            )
            return int(auto)

        return int(requested)

    def _resolve_tcp_offset(self) -> tuple[float, float, float]:
        """Get the TCP offset from config (ee_tcp_offset).

        Returns tuple (x, y, z) offset from the EE link frame to the tool center point.
        Default is (0, 0, 0.1034) for Franka Panda standard hand.
        """
        offset = self.config.get("ee_tcp_offset")
        if offset is None:
            return (0.0, 0.0, 0.1034)  # Default Franka TCP offset
        if isinstance(offset, (list, tuple)) and len(offset) >= 3:
            try:
                return (float(offset[0]), float(offset[1]), float(offset[2]))
            except (TypeError, ValueError):
                pass
        return (0.0, 0.0, 0.1034)

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
            return any(
                token in lowered for token in ("finger", "gripper", "grip", "jaw")
            )

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

    def _rotate_vector_by_quat(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q (xyzw format)."""
        qx, qy, qz, qw = q
        # Quaternion rotation: v' = q * v * q^-1
        # Using the formula: v' = v + 2*qw*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
        q_xyz = np.array([qx, qy, qz])
        t = 2.0 * np.cross(q_xyz, v)
        return v + qw * t + np.cross(q_xyz, t)

    def get_ee_pose(self) -> dict[str, Any]:
        """Get end-effector pose using Newton ArticulationView.

        Uses ArticulationView.get_link_transforms() which is more robust than
        directly accessing state.body_q (which can fail with CUDA-GL interop issues).

        The returned position includes the TCP offset (from config ee_tcp_offset)
        to give the actual tool center point position, not just the link frame.

        Returns:
            Dictionary with 'position' (list of 3 floats in meters) and
            'orientation' (quaternion as list of 4 floats [x, y, z, w]).
        """
        if not hasattr(self, "_get_ee_log_count"):
            self._get_ee_log_count = 0
        self._get_ee_log_count += 1

        default_pose = {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }

        if self._articulation_view is None:
            return default_pose

        state = self._state_accessor()
        if state is None:
            return default_pose

        ee_link_index = self._resolve_end_effector_idx()
        tcp_offset = self._resolve_tcp_offset()

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
            link_pos = ee_transform[:3]
            link_quat = ee_transform[3:]  # xyzw

            # Apply TCP offset: transform offset from link frame to world frame
            offset_world = self._rotate_vector_by_quat(np.array(tcp_offset), link_quat)
            tcp_pos = link_pos + offset_world

            # Log all link positions on first call for debugging
            if self._get_ee_log_count == 1:
                log.info(
                    f"FrankaNewtonDriver: Link transforms shape: {transforms_np.shape}"
                )
                for i in range(transforms_np.shape[1]):
                    pos = transforms_np[0, i, :3]
                    log.info(
                        f"  Link {i}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                    )
                log.info(
                    f"  Using ee_link_index={ee_link_index}, tcp_offset={tcp_offset}"
                )
                log.info(
                    f"  Link pos: [{link_pos[0]:.3f}, {link_pos[1]:.3f}, {link_pos[2]:.3f}]"
                )
                log.info(
                    f"  TCP pos:  [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]"
                )

            return {
                "position": tcp_pos.tolist(),  # TCP position in world frame
                "orientation": link_quat.tolist(),  # qx, qy, qz, qw quaternion
            }

        except Exception as e:
            log.warning(
                f"FrankaNewtonDriver '{self.component_name}': "
                f"Failed to get EE pose: {e}"
            )
            return default_pose

    def _ensure_ik_solver(self, end_effector_idx: int) -> bool:
        if self._model is None:
            log.warning(
                "FrankaNewtonDriver: IK unavailable - model not initialized yet"
            )
            return False

        if (
            getattr(self, "_ik_solver", None) is not None
            and getattr(self, "_ik_end_effector_idx", None) == end_effector_idx
        ):
            return True

        device = self._model.device
        self._ik_end_effector_idx = end_effector_idx

        # Get TCP offset from config
        tcp_offset = self._resolve_tcp_offset()
        self._tcp_offset = tcp_offset
        log.info(f"FrankaNewtonDriver: Using TCP offset: {tcp_offset}")

        self._ik_target_pos = wp.array(
            [wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device
        )
        self._ik_target_rot = wp.array(
            [wp.vec4(0.0, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=device
        )

        # NOTE: Do NOT use link_offset here - Newton's IK applies it in world frame,
        # but we need it in link-local frame (rotates with gripper orientation).
        # Instead, we'll compute the target link position ourselves in pass_cartesian_control_cmd()
        self._ik_pos_obj = ik.IKPositionObjective(
            link_index=end_effector_idx,
            link_offset=wp.vec3(0.0, 0.0, 0.0),  # No offset - we handle it manually
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

        self._ik_joint_q_in = wp.zeros(
            (1, self._model.joint_coord_count), dtype=wp.float32, device=device
        )
        self._ik_joint_q_out = wp.zeros(
            (1, self._model.joint_coord_count), dtype=wp.float32, device=device
        )

        self._ik_solver = ik.IKSolver(
            model=self._model,
            n_problems=1,
            objectives=[self._ik_pos_obj, self._ik_rot_obj, self._ik_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )
        log.info(
            f"FrankaNewtonDriver: Created IK solver for EE link_index={end_effector_idx}, "
            f"model.body_count={self._model.body_count}, model.joint_coord_count={self._model.joint_coord_count}"
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

        Also handles collision re-enable timeout: if finger collisions were
        disabled during release and the timeout has passed, re-enable them.

        Args:
            control_mode: Control mode string ('position', 'velocity', 'torque')
            cmd: Dictionary mapping joint names to command values
            **kwargs: Additional arguments passed to parent method
        """
        # Check for collision re-enable timeout (time-based instead of step-based)
        if self._collision_disabled and time.time() >= self._collision_reenable_time:
            self._on_gripper_release_internal(is_releasing=False)

        # Duplicate finger joint command for symmetric gripper motion
        if "panda_finger_joint1" in cmd:
            panda_finger_joint1 = cmd.get("panda_finger_joint1")
            cmd["panda_finger_joint2"] = panda_finger_joint1

        # Check for gripper release to trigger collision filtering
        self._check_gripper_release(cmd)

        # Call parent implementation
        super().pass_joint_group_control_cmd(control_mode, cmd, **kwargs)

    def pass_cartesian_control_cmd(
        self, control_mode: str, position, quaternion, **kwargs
    ) -> None:
        """Send a Cartesian control command by computing inverse kinematics."""
        if not hasattr(self, "_cart_cmd_log_count"):
            self._cart_cmd_log_count = 0
        self._cart_cmd_log_count += 1

        # Workspace validation - warn if target is likely unreachable
        # With TCP offset ~0.164m, minimum reachable TCP height is ~0.08-0.10m
        MIN_TCP_HEIGHT = 0.08  # Conservative minimum for forward reach
        if len(position) >= 3 and position[2] < MIN_TCP_HEIGHT:
            if self._cart_cmd_log_count <= 10 or self._cart_cmd_log_count % 100 == 0:
                tcp_offset = self._resolve_tcp_offset()
                required_link_z = position[2] - tcp_offset[2]
                log.warning(
                    f"FrankaNewtonDriver: Target TCP z={position[2]:.3f}m is below minimum reachable "
                    f"height (~{MIN_TCP_HEIGHT}m). With TCP offset z={tcp_offset[2]:.3f}m, this requires "
                    f"link7 at z={required_link_z:.3f}m which may be underground/unreachable."
                )

        if self._cart_cmd_log_count <= 5 or self._cart_cmd_log_count % 50 == 0:
            log.info(
                f"FrankaNewtonDriver: cartesian cmd #{self._cart_cmd_log_count} mode={control_mode}, "
                f"pos=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
            )
        if control_mode.lower() != "position":
            log.warning(
                "FrankaNewtonDriver: Cartesian control only supports position mode"
            )
            return

        if not (len(position) == 3 and len(quaternion) == 4):
            log.warning(
                "FrankaNewtonDriver: position must be 3 elements and quaternion must be 4 elements"
            )
            return

        end_effector_idx = self._resolve_end_effector_idx(
            kwargs.get("end_effector_idx")
        )
        if not self._ensure_ik_solver(end_effector_idx):
            return

        # Compute target LINK position from target TCP position
        # TCP_world = Link_world + rotate(tcp_offset_local, link_quat)
        # So: Link_world = TCP_world - rotate(tcp_offset_local, target_quat)
        tcp_offset = np.array(self._resolve_tcp_offset())
        target_quat = np.array(quaternion)
        # Rotate TCP offset by target orientation to get world-frame offset
        offset_world = self._rotate_vector_by_quat(tcp_offset, target_quat)
        # Target link position = target TCP position - rotated offset
        target_link_pos = np.array(position) - offset_world

        if self._cart_cmd_log_count <= 5 or self._cart_cmd_log_count % 100 == 0:
            log.info(
                f"FrankaNewtonDriver: TCP target=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] "
                f"-> Link target=[{target_link_pos[0]:.3f}, {target_link_pos[1]:.3f}, {target_link_pos[2]:.3f}]"
            )

        self._ik_pos_obj.set_target_position(0, wp.vec3(*target_link_pos))
        self._ik_rot_obj.set_target_rotation(0, wp.vec4(*quaternion))

        state = self._state_accessor()
        current_joint_q = None
        if state is not None and state.joint_q is not None:
            current_joint_q = state.joint_q.numpy()[: self._model.joint_coord_count]
            self._ik_joint_q_in.assign(np.array([current_joint_q], dtype=np.float32))

        ik_iterations = int(self.config.get("ik_iterations", 24))
        self._ik_solver.step(
            self._ik_joint_q_in, self._ik_joint_q_out, iterations=ik_iterations
        )

        joint_positions = self._ik_joint_q_out.numpy()[0]

        # Debug: log IK input/output difference and EE position
        if self._cart_cmd_log_count <= 5 or self._cart_cmd_log_count % 100 == 0:
            ik_in = self._ik_joint_q_in.numpy()[0][:7]
            ik_out = joint_positions[:7]
            diff = np.abs(ik_out - ik_in)
            log.info(f"FrankaNewtonDriver: IK input joints: {ik_in}")
            log.info(f"FrankaNewtonDriver: IK output joints: {ik_out}")
            log.info(f"FrankaNewtonDriver: IK delta max: {np.max(diff):.6f}")
            # Log current EE position (with TCP offset) vs target
            current_ee = self.get_ee_pose()
            current_pos = current_ee.get("position", [0, 0, 0])
            current_quat = current_ee.get("orientation", [0, 0, 0, 1])
            target_pos = list(position)
            error = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            log.info(
                f"FrankaNewtonDriver: TCP target=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
            )
            log.info(
                f"FrankaNewtonDriver: TCP actual=[{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}], error={error:.4f}m"
            )
            # Also log raw link position (before TCP offset) to debug offset application
            state = self._state_accessor()
            if state is not None and self._articulation_view is not None:
                try:
                    link_transforms = self._articulation_view.get_link_transforms(state)
                    if link_transforms is not None:
                        transforms_np = link_transforms.numpy()
                        ee_idx = self._resolve_end_effector_idx()
                        if ee_idx < transforms_np.shape[1]:
                            link_pos = transforms_np[0, ee_idx, :3]
                            log.info(
                                f"FrankaNewtonDriver: Link{ee_idx} raw pos=[{link_pos[0]:.3f}, {link_pos[1]:.3f}, {link_pos[2]:.3f}]"
                            )
                            log.info(
                                f"FrankaNewtonDriver: TCP offset={self._tcp_offset}, quat={current_quat}"
                            )
                except Exception as e:
                    log.warning(
                        f"FrankaNewtonDriver: Could not get link transform: {e}"
                    )
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
            log.warning(
                "FrankaNewtonDriver: No arm joint group configured; cannot apply IK solution"
            )
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

        if self._cart_cmd_log_count <= 5 or self._cart_cmd_log_count % 50 == 0:
            log.info(
                f"FrankaNewtonDriver: IK result -> {len(cmd)} joint targets: {list(cmd.values())[:7]}"
            )
        self.pass_joint_group_control_cmd("position", cmd)

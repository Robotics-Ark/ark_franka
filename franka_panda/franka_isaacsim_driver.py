from typing import Any

import numpy as np
from ark.system.isaac.isaac_robot_driver import IsaacSimRobotDriver
from ark.tools.log import log
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.robot.manipulators.grippers import ParallelGripper


class FrankaIsaacDriver(IsaacSimRobotDriver):
    """Isaac Sim driver for the Franka Panda manipulator."""

    def __init__(
        self,
        component_name: str,
        component_config: dict[str, Any],
        sim_app: Any,
        world: Any,
    ) -> None:
        """
        Initialize the Franka Panda driver in Isaac Sim.

        Sets up the robot articulation, adds the gripper, initializes the manipulator,
        and creates a kinematics solver for Cartesian control.

        Args:
            component_name (str): Name of the robot component.
            component_config (dict[str, Any]): Configuration dictionary for the robot.
            sim_app (Any): Isaac Sim application instance.
            world (Any): Isaac Sim world object.

        """

        super().__init__(component_name, component_config, sim_app, world)

        gripper = ParallelGripper(
            end_effector_prim_path=f"/panda/panda_hand",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.02, 0.02]),
            action_deltas=np.array([0.01, 0.01]),
        )

        self.franka = world.scene.add(
            SingleManipulator(
                prim_path=self.prim_path,
                name="franka",
                end_effector_prim_path="/panda/panda_hand",
                gripper=gripper,
            )
        )
        self.franka.initialize()

        self.franka.gripper.set_default_state(
            self.franka.gripper.joint_opened_positions
        )

        self._controller = KinematicsSolver(self.franka)

    def pass_cartesian_control_cmd(
        self, control_mode: str, position, quaternion, end_effector_idx, gripper
    ) -> None:
        """
        Apply a Cartesian-space control command to the end-effector.

        This method uses the IK solver to compute joint positions from a target
        end-effector position and orientation. Only position control mode is supported.

        Args:
            control_mode (str): Control mode type; only "position" is supported.
            position (array-like): Target end-effector position in world coordinates.
            quaternion (array-like): Target end-effector orientation as [x, y, z, w].
            end_effector_idx: Index of the end-effector (currently ignored).
            gripper: Gripper command (currently ignored).


        """

        if control_mode != "position":
            log.warn(f"Cartesian control_mode '{control_mode}' not supported in Isaac.")
            return
        actions, succ = self._controller.compute_inverse_kinematics(
            target_position=np.asarray(position),
            target_orientation=np.asarray(quaternion),
        )
        if succ:
            self.franka.apply_action(actions)
        else:
            log.warn("IK did not converge to a solution.  No action is being taken.")

    def get_ee_pose(self) -> dict[str, np.ndarray]:
        """Return the end-effector (EE) pose in world coordinates.

        Returns:
            Dictionary containing:
                - "position": EE world-space position.
                - "orientation": EE world-space quaternion orientation.
        """
        position, orientation = self.franka.gripper.get_world_pose()
        return {"position": position, "orientation": orientation}

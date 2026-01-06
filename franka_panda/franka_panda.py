import importlib
from enum import Enum
from typing import Any, Dict

import numpy as np
from ark.client.comm_infrastructure.base_node import main
from ark.system.component.robot import Robot
from ark.system.driver.robot_driver import RobotDriver
from arktypes import (
    joint_group_command_t,
    joint_state_t,
    pose_t,
    task_space_command_t,
)
from arktypes.utils import unpack, pack


class Drivers(Enum):
    PYBULLET_DRIVER = "franka_panda.franka_pybullet_driver.FrankaPyBulletDriver"
    ISAAC_DRIVER = "franka_panda.franka_isaacsim_driver.FrankaIsaacDriver"
    DRIVER = "franka_panda.FrankaResearch3Driver"

    def load(self):
        module_path, class_name = self.value.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


class FrankaPanda(Robot):
    def __init__(
        self,
        name: str,
        global_config: Dict[str, Any] = None,
        driver: RobotDriver = None,
    ) -> None:

        super().__init__(
            name=name,
            global_config=global_config,
            driver=driver,
        )

        #######################################
        ##     custom communication setup    ##
        #######################################
        self._joint_cmd_msg, self._cartesian_cmd_msg = None, None

        observation_channels = global_config["observation_channels"]
        action_channels = global_config["action_channels"]
        namespace = global_config["namespace"]

        self.joint_group_command_ch = None
        self.cartesian_position_control_ch = None
        # Action channels
        if action_channels is not None and len(action_channels) > 0:
            for key in action_channels.keys():
                if "joint_group_command" in key:
                    self.joint_group_command_ch = key
                if "cartesian_command" in key:
                    self.cartesian_position_control_ch = key
        else:
            self.joint_group_command_ch = (
                f"{namespace}/" + self.name + "/joint_group_command"
            )
            self.cartesian_position_control_ch = (
                f"{namespace}/" + self.name + "/cartesian_command"
            )

            if self.sim:
                self.joint_group_command_ch = self.joint_group_command_ch + "/sim"
                self.cartesian_position_control_ch = (
                    self.cartesian_position_control_ch + "/sim"
                )

        if self.joint_group_command_ch:
            self.create_subscriber(
                self.joint_group_command_ch,
                joint_group_command_t,
                self._joint_group_command_callback,
            )
        if self.cartesian_position_control_ch:
            self.create_subscriber(
                self.cartesian_position_control_ch,
                task_space_command_t,
                self._cartesian_position_command_callback,
            )

        # Observation channels
        if observation_channels is not None and len(observation_channels) > 0:
            for key in observation_channels.keys():
                if "joint_states" in key:
                    self.joint_states_pub = key
                if "ee_state" in key:
                    self.ee_state_pub = key
        else:
            if self.sim == True:
                self.joint_states_pub = (
                    f"{namespace}/" + self.name + "/joint_states/sim"
                )
                self.ee_state_pub = f"{namespace}/" + self.name + "/ee_state/sim"

            else:
                self.joint_states_pub = self.name + "/joint_states"
                self.ee_state_pub = self.name + "/ee_state"

        self.component_channels_init(
            {
                self.joint_states_pub: joint_state_t,
                self.ee_state_pub: pose_t,
            }
        )

        self.joint_group_command = None
        self.cartesian_position_control_command = None

    def control_robot(self):
        """
        will be called at the fequency dictated in the config
        handles the control of the robot and the
        """
        if self.joint_group_command:
            cmd_dict = {}
            group_name = self.joint_group_command["name"]
            for joint, goal in zip(
                list(
                    self.joint_groups[self.joint_group_command["name"]][
                        "actuated_joints"
                    ]
                ),
                self.joint_group_command["cmd"],
            ):
                cmd_dict[joint] = goal
            self._joint_cmd_msg = None
            control_mode = self.joint_groups[group_name]["control_mode"]
            self.control_joint_group(control_mode, cmd_dict)

        if self.cartesian_position_control_command:
            group_name = self.cartesian_position_control_command["name"]
            control_mode = self.joint_groups[group_name]["control_mode"]
            end_effector_idx = self.robot_config.get("end_effector_idx", 6)
            self.control_cartesian(
                control_mode,
                cmd=self.cartesian_position_control_command,
                end_effector_idx=end_effector_idx,
            )

        self.joint_group_command = None
        self.cartesian_position_control_command = None

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the robot.
        This method is called by the base class to get the state of the robot.
        """
        ee_pose = self._driver.get_ee_pose()
        joint_position = self.get_joint_positions()
        return {
            "joint_positions": joint_position,
            "end_effector_pose": ee_pose,
        }

    def pack_data(self, state: Dict[str, Any]) -> Dict[str, Any]:

        joint_state = state["joint_positions"]
        ee_pose = state["end_effector_pose"]

        joint_msg = joint_state_t()
        joint_msg.n = len(joint_state)
        joint_msg.name = list(joint_state.keys())
        joint_msg.position = list(joint_state.values())
        joint_msg.velocity = [0.0] * joint_msg.n
        joint_msg.effort = [0.0] * joint_msg.n

        ee_msg = pack.pose(
            np.array(ee_pose["position"]), np.array(ee_pose["orientation"])
        )

        return {self.joint_states_pub: joint_msg, self.ee_state_pub: ee_msg}

    ####################################################
    ##      Franka Subscriber Callbacks               ##
    ####################################################
    def _joint_group_command_callback(self, t, channel_name, msg):
        cmd, name = unpack.joint_group_command(msg)
        self.joint_group_command = {
            "cmd": cmd,
            "name": name,
        }

    def _cartesian_position_command_callback(self, t, channel_name, msg):
        name, position, quaternion, gripper = unpack.task_space_command(msg)
        self.cartesian_position_control_command = {
            "name": name,
            "position": position,
            "quaternion": quaternion,
            "gripper": gripper,
        }

    ####################################################
    ##       Franka Custom Control Methods            ##
    ##    note: control_joint_group is default        ##
    ####################################################

    def control_cartesian(self, control_mode, cmd, end_effector_idx) -> None:
        self._driver.pass_cartesian_control_cmd(
            control_mode,
            position=cmd["position"],
            quaternion=cmd["quaternion"],
            end_effector_idx=end_effector_idx,
            gripper=cmd.get("gripper", None),
        )

    #####################################################


CONFIG_PATH = "panda.yaml"
if __name__ == "__main__":
    name = "Franka"
    driver = FrankaResearch3Driver(name, CONFIG_PATH)
    main(FrankaPanda, name, CONFIG_PATH, driver)

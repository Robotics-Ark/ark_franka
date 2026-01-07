
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import pprint
# from viper_300s_driver import Viper300sDriver

from ark.client.comm_infrastructure.base_node import main
from ark.system.component.robot import Robot, robot_control
from ark.system.driver.robot_driver import RobotDriver
from franka_pybullet_driver import FrankaPyBulletDriver
from ark.tools.log import log
import arktypes
from arktypes import flag_t, joint_group_command_t, joint_state_t, pose_t, task_space_command_t
from arktypes.utils import unpack, pack
import numpy as np

@dataclass
class Drivers(Enum):
    PYBULLET_DRIVER = FrankaPyBulletDriver

    try:
        from franka_newton_driver import FrankaNewtonDriver
        NEWTON_DRIVER = FrankaNewtonDriver
    except ImportError:
        log.warn("FrankaNewtonDriver not available - Newton backend may not be installed")

    try:
        from franka_driver import FrankaResearch3Driver
        DRIVER = FrankaResearch3Driver
    except ImportError:
        log.warn("FrankaResearch3Driver is failing, OS might be incompatible with the Real Franka Panda Robot")
    

class FrankaPanda(Robot):
    def __init__(self,
                 name: str,   
                 global_config: Dict[str, Any] = None,
                 driver: RobotDriver = None,
                 ) -> None:
        
        super().__init__(name = name,
                         global_config = global_config,
                         driver = driver,
                         )
        log.info(f"[DEBUG TYPES] arktypes path: {arktypes.__file__}")
        log.info(
            f"[DEBUG TYPES] task_space_command_t fingerprint: "
            f"{task_space_command_t._get_packed_fingerprint().hex()}"
        )

        # WORKAROUND: Remove duplicate joints caused by base Robot class bug
        # The base class in robot.py appends joints twice (lines 186 and 200)
        # This deduplicates the list while preserving order
        self._all_actuated_joints = list(dict.fromkeys(self._all_actuated_joints))

        #######################################
        ##     custom communication setup    ##
        #######################################
        self._joint_cmd_msg, self._cartesian_cmd_msg =  None, None
        
        self.joint_group_command_ch = self.name + "/joint_group_command"
        self.cartesian_position_control_ch = self.name + "/cartesian_command"

        if self.sim:
            self.joint_group_command_ch = self.joint_group_command_ch + "/sim"
            self.cartesian_position_control_ch = self.cartesian_position_control_ch + "/sim"

        self.create_subscriber(self.joint_group_command_ch, joint_group_command_t, self._joint_group_command_callback)
        self.create_subscriber(self.cartesian_position_control_ch, task_space_command_t, self._cartesian_position_command_callback)

        if self.sim == True: 
            self.joint_states_pub = self.name + "/joint_states/sim"
            self.ee_state_pub = self.name + "/ee_state/sim"
            self.component_channels_init({
                self.joint_states_pub: joint_state_t,
                self.ee_state_pub: pose_t,
            })
        else:
            self.joint_states_pub = self.name + "/joint_states"
            self.ee_state_pub = self.name + "/ee_state"
            self.component_channels_init({
                self.joint_states_pub: joint_state_t,
                self.ee_state_pub: pose_t,
            })

        self.joint_group_command = None
        self.cartesian_position_control_command = None

    def control_robot(self):
        '''
        will be called at the fequency dictated in the config
        handles the control of the robot and the
        '''
        # DEBUG: Count calls and log
        if not hasattr(self, '_control_count'):
            self._control_count = 0
        self._control_count += 1

        if self._control_count <= 3 or self._control_count % 1000 == 0:
            log.info(f"[DEBUG FRANKA #{self._control_count}] control_robot() called, has_cmd={self.joint_group_command is not None}")

        if self.joint_group_command:
            if self._control_count <= 3:
                log.ok(f"[DEBUG FRANKA #{self._control_count}] Processing joint_group_command: {self.joint_group_command}")

            cmd_dict = {}
            group_name = self.joint_group_command['name']
            for joint, goal in zip(list(self.joint_groups[self.joint_group_command['name']]["actuated_joints"]), self.joint_group_command['cmd']):
                cmd_dict[joint] = goal
            self._joint_cmd_msg = None
            control_mode = self.joint_groups[group_name]["control_mode"]

            if self._control_count <= 3:
                log.ok(f"[DEBUG FRANKA #{self._control_count}] Calling control_joint_group(mode={control_mode}, {len(cmd_dict)} joints)")

            self.control_joint_group(control_mode, cmd_dict)

        if self.cartesian_position_control_command:
            group_name = self.cartesian_position_control_command['name']
            control_mode = self.joint_groups[group_name]["control_mode"]
            end_effector_idx = self.robot_config.get(
                "end_effector_idx",
                self.robot_config.get("ee_index", self.robot_config.get("ee_link_index", 6)),
            )
            if not hasattr(self, "_cartesian_control_count"):
                self._cartesian_control_count = 0
            self._cartesian_control_count += 1
            if self._cartesian_control_count <= 3 or self._cartesian_control_count % 500 == 0:
                log.info(
                    f"[DEBUG FRANKA CART #{self._cartesian_control_count}] "
                    f"Processing cartesian_command: group={group_name}, mode={control_mode}"
                )
            self.control_cartesian(control_mode, cmd=self.cartesian_position_control_command, end_effector_idx=end_effector_idx)

        # For simulated robots, keep commands persistent (high-frequency control loop)
        # For real robots, clear after each execution (command-rate control)
        if not self.sim:
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

        ee_msg = pack.pose(np.array(ee_pose["position"]), np.array(ee_pose["orientation"]))

        return {
            self.joint_states_pub: joint_msg,
            self.ee_state_pub: ee_msg
        }
    
    ####################################################
    ##      Franka Subscriber Callbacks               ##
    ####################################################
    def _joint_group_command_callback(self, t, channel_name, msg):
        # DEBUG: Count callbacks
        if not hasattr(self, '_callback_count'):
            self._callback_count = 0
        self._callback_count += 1
        try:
            cmd, name = unpack.joint_group_command(msg)
            self.joint_group_command = {
                "cmd": cmd,
                "name": name,
            }

            # DEBUG: Log first few callbacks
            if self._callback_count <= 3 or self._callback_count % 200 == 0:
                log.ok(f"[DEBUG CALLBACK #{self._callback_count}] Received command: group={name}, cmd={cmd}")
        except Exception as exc:
            log.error(f"Failed to unpack joint_group_command: {exc}")

    def _cartesian_position_command_callback(self, t, channel_name, msg): 
        try:
            name, position, quaternion, gripper = unpack.task_space_command(msg)
            self.cartesian_position_control_command = {
                "name": name,
                "position": position,
                "quaternion": quaternion,
                "gripper": gripper
            }
            if not hasattr(self, "_cartesian_callback_count"):
                self._cartesian_callback_count = 0
            self._cartesian_callback_count += 1
            if self._cartesian_callback_count <= 3 or self._cartesian_callback_count % 200 == 0:
                log.ok(
                    f"[DEBUG CART CALLBACK #{self._cartesian_callback_count}] "
                    f"Received cartesian command: name={name}, pos={position}, quat={quaternion}, grip={gripper}"
                )
        except Exception as exc:
            log.error(f"Failed to unpack cartesian_command: {exc}")

    ####################################################
    ##       Franka Custom Control Methods            ##
    ##    note: control_joint_group is default        ##
    ####################################################

    def control_cartesian(self, control_mode,cmd, end_effector_idx) -> None:
        self._driver.pass_cartesian_control_cmd(control_mode,
                                        position=cmd['position'],
                                       quaternion=cmd['quaternion'],
                                       end_effector_idx=end_effector_idx,
                                       gripper=cmd.get('gripper', None))

    #####################################################

CONFIG_PATH = "panda.yaml"
if __name__ == "__main__":
    name = "Franka"
    driver = FrankaResearch3Driver(name, CONFIG_PATH)
    main(FrankaPanda, name, CONFIG_PATH, driver)

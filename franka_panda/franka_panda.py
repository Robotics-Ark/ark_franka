
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import pprint
# from viper_300s_driver import Viper300sDriver

from ark.client.comm_infrastructure.base_node import main
from ark.system.component.robot import Robot, robot_control
from ark.system.driver.robot_driver import RobotDriver
from ark.system.pybullet.pybullet_robot_driver import BulletRobotDriver
from ark.tools.log import log
from arktypes import flag_t, joint_group_command_t, joint_state_t

# from franka_driver import FrankaResearch3Driver


@dataclass
class Drivers(Enum): 
    PYBULLET_DRIVER = BulletRobotDriver
    # DRIVER = FrankaResearch3Driver
    

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

        #######################################
        ##     custom communication setup    ##
        #######################################
        self._joint_cmd_msg, self._cartesian_cmd_msg =  None, None
        
        self.joint_group_command = self.name + "/joint_group_command"
        self.cartesian_position_control_name = self.name + "/cartesian_command"

        if self.sim:
            self.subscriber_name = self.joint_group_command + "/sim"
            self.cartesian_position_control_name = self.cartesian_position_control_name + "/sim"

        self.create_subscriber(self.joint_group_command, joint_group_command_t, self._joint_group_command_callback)
        self.create_subscriber(self.cartesian_position_control_name, joint_group_command_t, self._cartesian_position_command_callback)

        if self.sim == True: 
            self.publisher_name = self.name + "/joint_states/sim"
            self.component_channels_init([
                (self.publisher_name, joint_state_t)
            ])
        else:
            self.publisher_name = self.name + "/joint_states"
            self.component_channels_init([
                (self.publisher_name, joint_state_t)
            ])

    def get_robot_data(self):
        '''
        will be called at the fequency dictated in the config
        handles the control of the robot and the 
        '''
        if self._joint_cmd_msg:
            msg = self._joint_cmd_msg
            group_name = msg.name
            cmd = msg.cmd
            cmd_dict = {}
            for joint, goal in zip(list(self.joint_groups[group_name]["actuated_joints"]), cmd):
                cmd_dict[joint] = goal
            self._joint_cmd_msg = None
            self.control_joint_group(group_name, cmd_dict)
        elif self._cartesian_cmd_msg:
            msg = self._cartesian_cmd_msg
            group_name = msg.name
            cmd = msg.cmd
            cmd_dict = {}
            for joint, goal in zip(list(self.joint_groups[group_name]["actuated_joints"]), cmd):
                cmd_dict[joint] = goal
            self._cartesian_cmd_msg = None
            self.control_cartesian(group_name, cmd_dict)
            
        # print(self.get_joint_positions())
        return self.get_joint_positions()

    def pack_data(self, pos_dict):
        msg = joint_state_t()
        msg.n = len(pos_dict)
        msg.name = list(pos_dict.keys())
        msg.position = list(pos_dict.values())
        msg.velocity = [0.0] * msg.n
        msg.effort = [0.0] * msg.n

        return {
            self.publisher_name: msg
        }
    
    ####################################################
    ##      Franka Subscriber Callbacks               ##
    ####################################################
    def _joint_group_command_callback(self, t, channel_name, msg):
        self._joint_cmd_msg = msg

    def _cartesian_position_command_callback(self, t, channel_name, msg): 
        self._cartesian_cmd_msg = msg

    ####################################################
    ##       Franka Custom Control Methods            ##
    ##    note: control_joint_group is default        ##
    ####################################################

    @robot_control
    def control_cartesian(self, control_mode: str, joints: List[str], cmd: Dict[str, float], **kwargs) -> None: 
        if self.sim == True: 
            log.error("Cartesian Positon control is not availble for Franka in Pybullet yet") #TODO add Cartesian Position Control
        elif self.sim == False: 
            self._driver.pass_cartesian_control_cmd(control_mode, joints, cmd, **kwargs)

    #####################################################

CONFIG_PATH = "panda.yaml"
if __name__ == "__main__":
    name = "Franka"
    driver = FrankaResearch3Driver(name, CONFIG_PATH)
    main(FrankaPanda, name, CONFIG_PATH, driver)
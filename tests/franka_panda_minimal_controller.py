
from ark.client.comm_infrastructure.base_node import BaseNode,  main
import numpy as np
from arktypes import joint_group_command_t, joint_single_command_t, header_t, stamp_t, joint_state_t

__doc__ = (
    """Controls a singular joint in the ViperX"""
)

SIM = True


class FrankaControllerNode(BaseNode):

    def __init__(self):
        '''
        Initialize the FrankaJointController.
        This class is responsible for controlling the Franka robot's joints.
        '''
        super().__init__("FrankaJointController")
        self.index = 0

        if SIM == True:
            self.pub = self.create_publisher("Franka/joint_group_command/sim", joint_group_command_t)
            self.pub2 = self.create_publisher("Franka/joint_single_command/sim", joint_single_command_t)

        self.create_stepper(1, self.step)
        
    def step(self):
        '''
        Publish joint commands to the Franka robot.
        The command is a list of joint angles for the robot's joints.
        '''
        msg = joint_group_command_t()
        msg.name = "all"
        msg.n = 9
        user_value = float(input("Enter a value: "))
        msg.cmd = [-0.3, 0.1, 0.3, -1.4 + user_value, 0.1, 1.8, 0.7, user_value, 0]
        self.pub.publish(msg)
        print(msg.cmd)

if __name__ == "__main__":
    main(FrankaControllerNode)

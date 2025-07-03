
from ark.client.comm_infrastructure.instance_node import InstanceNode
import numpy as np
from arktypes import joint_group_command_t, task_space_command_t
from arktypes.utils import unpack, pack
__doc__ = (
    """Controls a singular joint in the ViperX"""
)

SIM = True


class FrankaControllerNode(InstanceNode):

    def __init__(self):
        '''
        Initialize the FrankaJointController.
        This class is responsible for controlling the Franka robot's joints.
        '''
        super().__init__("FrankaJointController")

        if SIM == True:
            self.joint_group_command = self.create_publisher("Franka/joint_group_command/sim", joint_group_command_t)
            self.task_space_command = self.create_publisher("Franka/cartesian_command/sim", task_space_command_t)

from ark.client.comm_infrastructure.base_node import main
from ark.system.simulation.simulator_node import SimulatorNode
from ark.tools.log import log

CONFIG_PATH = "config/global_config.yaml"

class MySimulatorNode(SimulatorNode):

    def initialize_scene(self):
        pass
    
    def step(self):
        pass

if __name__ == "__main__":
    main(MySimulatorNode, CONFIG_PATH)
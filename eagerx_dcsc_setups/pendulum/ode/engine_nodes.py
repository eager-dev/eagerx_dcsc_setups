import eagerx
from eagerx.utils.utils import Msg
from typing import Optional, List
from std_msgs.msg import Float32, Float32MultiArray, UInt64
import numpy as np


class CustomOdeInput(eagerx.EngineNode):
    @staticmethod
    @eagerx.register.spec("CustomOdeInput", eagerx.EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        default_action: List,
        process: Optional[int] = eagerx.process.ENGINE,
        delay_state: bool = True,
        color: Optional[str] = "green",
    ):
        """OdeInput spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(CustomOdeInput)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]
        spec.config.states = ["delay"] if delay_state else []

        # Set custom node params
        spec.config.default_action = default_action

        # Set space converter for delay state
        spec.states.delay.space_converter = eagerx.SpaceConverter.make("Space_Float32", low=0, high=0, dtype="float32")

    def initialize(self, default_action):
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        assert (
            self.process == eagerx.process.ENGINE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Engine process"
        self.obj_name = self.config["name"]
        self.default_action = np.array(default_action)

    @eagerx.register.states(delay=Float32)
    def reset(self, delay=None):
        self.simulator[self.obj_name]["input"] = np.squeeze(np.array(self.default_action))
        if delay is not None:
            self.set_delay(delay.data, "inputs", "action")

    @eagerx.register.inputs(tick=UInt64, action=Float32MultiArray)
    @eagerx.register.outputs(action_applied=Float32MultiArray)
    def callback(
        self,
        t_n: float,
        tick: Optional[Msg] = None,
        action: Optional[Float32MultiArray] = None,
    ):
        assert isinstance(self.simulator[self.obj_name], dict), (
            'Simulator object "%s" is not compatible with this simulation node.' % self.simulator[self.obj_name]
        )

        # Set action in simulator for next step.
        self.simulator[self.obj_name]["input"] = np.squeeze(action.msgs[-1].data)

        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])

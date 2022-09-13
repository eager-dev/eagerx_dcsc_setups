import eagerx
from eagerx import Space, register
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg

from typing import Optional, List, Any
import numpy as np


class CustomOdeInput(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        default_action: List,
        delay_state: bool = True,
        color: Optional[str] = "green",
    ):
        """OdeInput spec"""
        spec = cls.get_specification()

        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(CustomOdeInput)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = eagerx.process.ENGINE
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]
        spec.config.states = ["delay"] if delay_state else []

        # Set custom node params
        spec.config.default_action = default_action
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        # We will probably use self.simulator in callback & reset.
        assert (
            self.process == eagerx.process.ENGINE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Engine process"
        self.default_action = np.array(spec.config.default_action)
        self.simulator = simulator

    @register.states(delay=Space(low=0, high=0, shape=(), dtype="float32"))
    def reset(self, delay: np.ndarray = None):
        self.simulator["input"] = self.default_action
        if delay is not None:
            self.set_delay(float(delay), "inputs", "action")

    @register.inputs(tick=Space(dtype="int64"), action=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(
        self,
        t_n: float,
        tick: Optional[Msg] = None,
        action: Optional[Msg] = None,
    ):
        # Set action in simulator for next step.
        self.simulator["input"] = action.msgs[-1]

        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])

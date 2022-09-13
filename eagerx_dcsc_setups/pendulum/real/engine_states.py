import rospy
from typing import Any
from random import random
import eagerx
from eagerx.core.specs import EngineStateSpec
from dcsc_setups.srv import MopsWrite, MopsWriteRequest


class RandomActionAndSleep(eagerx.EngineState):
    @classmethod
    def make(cls, sleep_time: float = 1.0, repeat: int = 1) -> EngineStateSpec:
        spec = cls.get_specification()
        spec.config.sleep_time = sleep_time
        spec.config.repeat = repeat
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        self.sleep_time = spec.config.sleep_time
        self.repeat = spec.config.repeat
        self.service = rospy.ServiceProxy("/mops/write", MopsWrite)
        self.service.wait_for_service()

    def reset(self, state):
        for _i in range(self.repeat):
            action = -3.0 + random() * 6.0
            req = MopsWriteRequest()
            req.actuators.digital_outputs = 1
            req.actuators.voltage0 = action
            req.actuators.voltage1 = 0.0
            req.actuators.timeout = 0.5
            self.service(req)
            rospy.sleep(self.sleep_time)


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls):
        spec = cls.get_specification()
        spec.initialize(DummyState)
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        pass

    def reset(self, state):
        pass

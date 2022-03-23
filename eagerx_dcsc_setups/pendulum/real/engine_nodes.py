from typing import Optional
import numpy as np
import rospy

# IMPORT ROS
from std_msgs.msg import UInt64, Float32MultiArray

# IMPORT EAGERX
import eagerx.core.register as register
from eagerx.utils.utils import Msg
from eagerx import EngineNode
from eagerx import process

from dcsc_setups.srv import MopsWrite, MopsWriteRequest, MopsReadRequest, MopsRead


class PendulumOutput(EngineNode):
    @staticmethod
    @register.spec("PendulumOutput", EngineNode)
    def spec(spec, name: str, rate: float, process: Optional[int] = process.NEW_PROCESS, color: Optional[str] = "cyan"):
        """PendulumOutput spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(PendulumOutput)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["pendulum_output"]

    def initialize(self):
        self.service = rospy.ServiceProxy("/mops/read", MopsRead)
        self.service.wait_for_service()

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64)
    @register.outputs(pendulum_output=Float32MultiArray)
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        response = self.service.call(MopsReadRequest())
        data = [np.pi - response.sensors.position0, response.sensors.speed]
        return dict(pendulum_output=Float32MultiArray(data=data))


class PendulumInput(EngineNode):
    @staticmethod
    @register.spec("PendulumInput", EngineNode)
    def spec(spec, name: str, rate: float, process: Optional[int] = process.NEW_PROCESS, color: Optional[str] = "green"):
        """PendulumInput spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(PendulumInput)

        # Modify default node params
        params = dict(
            name=name, rate=rate, process=process, color=color, inputs=["tick", "pendulum_input"], outputs=["action_applied"]
        )
        spec.config.update(params)

        # Set component parameter
        spec.inputs.pendulum_input.window = 1

    def initialize(self):
        self.service = rospy.ServiceProxy("/mops/write", MopsWrite)
        self.service.wait_for_service()

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64, pendulum_input=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(self, t_n: float, tick: Optional[Msg] = None, pendulum_input: Optional[Msg] = None):
        if len(pendulum_input.msgs) > 0:
            input = np.squeeze(pendulum_input.msgs[-1].data)
            if input is not None:
                req = MopsWriteRequest()
                req.actuators.digital_outputs = 1
                req.actuators.voltage0 = input
                req.actuators.voltage1 = 0.0
                req.actuators.timeout = 0.5
                self.service(req)
            # Send action that has been applied.
        else:
            input = 0
        return dict(action_applied=Float32MultiArray(data=[input]))

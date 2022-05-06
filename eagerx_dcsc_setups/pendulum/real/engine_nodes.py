from typing import Optional
import numpy as np
import rospy
from time import time, sleep

# IMPORT ROS
from std_msgs.msg import UInt64, Float32MultiArray, Float32

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
        spec.config.outputs = ["x"]

    def initialize(self):
        self.service = rospy.ServiceProxy("/mops/read", MopsRead)
        self.service.wait_for_service()

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64)
    @register.outputs(x=Float32MultiArray)
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        response = self.service.call(MopsReadRequest())
        data = [response.sensors.position0 + np.pi, response.sensors.speed]
        return dict(x=Float32MultiArray(data=data))


class PendulumInput(EngineNode):
    @staticmethod
    @register.spec("PendulumInput", EngineNode)
    def spec(spec, name: str, rate: float, process: Optional[int] = process.NEW_PROCESS, color: Optional[str] = "green"):
        """PendulumInput spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(PendulumInput)

        # Modify default node params
        params = dict(
            name=name, rate=rate, process=process, color=color, inputs=["tick", "u", "x"], outputs=["action_applied"]
        )
        spec.config.update(params)

        # Set component parameter
        spec.inputs.u.window = 1

    def initialize(self):
        self.pub = rospy.Publisher("delay", Float32, queue_size=1)
        self.service = rospy.ServiceProxy("/mops/write", MopsWrite)
        self.service.wait_for_service()

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64, u=Float32MultiArray, x=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(self, t_n: float, tick: Optional[Msg] = None, u: Optional[Msg] = None, x: Optional[Msg] = None):
        # Check if delay is constant
        delay = u.info.t_in[-1].wc_stamp - x.info.t_in[-1].wc_stamp
        self.pub.publish(Float32(data=delay))
        if len(u.msgs) > 0:
            input = np.squeeze(u.msgs[-1].data)
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

class ConstantDelayAction(EngineNode):
    @staticmethod
    @register.spec("ConstantDelayAction", EngineNode)
    def spec(spec, name: str, rate: float, desired_delay: float, process: Optional[int] = process.NEW_PROCESS, color: Optional[str] = "green"):
        """DelayAction spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(ConstantDelayAction)

        # Modify default node params
        params = dict(
            name=name, rate=rate, process=process, color=color, inputs=["observation", "action"], outputs=["delayed_action"]
        )
        spec.config.update(params)

        # Custom params
        spec.config.desired_delay = desired_delay

    def initialize(self, desired_delay):
        self.desired_delay = desired_delay

    @register.states()
    def reset(self):
        pass

    @register.inputs(observation=Float32MultiArray, action=Float32MultiArray)
    @register.outputs(delayed_action=Float32MultiArray)
    def callback(self, t_n: float, observation: Optional[Msg] = None, action: Optional[Msg] = None):
        t_in = observation.info.t_in[-1].wc_stamp
        sleep_duration = t_in + self.desired_delay - time()
        # rospy.loginfo("Desired delay: " + str(self.desired_delay) + ", sleep_duration: " + str(sleep_duration))
        if sleep_duration > 0:
            sleep(sleep_duration)
        else:
            rospy.logwarn("[DelayAction] Delay is already larger than desired delay.")
        return dict(delayed_action=action.msgs[-1])
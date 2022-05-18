from typing import Optional
import numpy as np
import rospy
from time import time, sleep
from threading import Thread

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
    def spec(
        spec,
        name: str,
        rate: float,
        fixed_delay: float = 0.0,
        process: Optional[int] = process.NEW_PROCESS,
        color: Optional[str] = "green",
    ):
        """PendulumInput spec"""
        # Modify default node params
        spec.config.update(
            name=name, rate=rate, process=process, color=color, inputs=["tick", "u", "x"], outputs=["action_applied"]
        )
        spec.config.fixed_delay = fixed_delay

        # Set component parameter
        spec.inputs.u.window = 1

    def initialize(self, fixed_delay: float):
        assert fixed_delay >= 0, "Delay must be non-negative."
        self.fixed_delay = fixed_delay
        self.pub_act = rospy.Publisher("/mops/actuator_delay", Float32, queue_size=1)
        self.pub_comp = rospy.Publisher("/mops/computation_delay", Float32, queue_size=1)
        self.service = rospy.ServiceProxy("/mops/write", MopsWrite)
        self.service.wait_for_service()
        self.prev_seq = None

    @register.states()
    def reset(self):
        self.prev_seq = None

    @register.inputs(tick=UInt64, u=Float32MultiArray, x=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(self, t_n: float, tick: Optional[Msg] = None, u: Optional[Msg] = None, x: Optional[Msg] = None):
        if self.prev_seq is not None and self.prev_seq == u.info.t_in[-1].seq:
            # We do not want to apply the same action more than once
            return dict(action_applied=u.msgs[-1])
        self.prev_seq = u.info.t_in[-1].seq

        # Create request
        input = np.squeeze(u.msgs[-1].data)
        req = MopsWriteRequest()
        req.actuators.digital_outputs = 1
        req.actuators.voltage0 = input
        req.actuators.voltage1 = 0.0
        req.actuators.timeout = 0.5

        # Determine sleep duration to achieve fixed delay.
        obs_ts = x.info.t_in[-1].wc_stamp
        act_ts = u.info.t_in[-1].wc_stamp
        dt = act_ts - obs_ts
        sleep_duration = self.fixed_delay - dt

        # Call srvs asynchronously
        thread = Thread(target=self._async_srvs_call, args=(req, sleep_duration, obs_ts, dt))
        thread.start()
        return dict(action_applied=u.msgs[-1])

    def _async_srvs_call(self, req, sleep_duration, obs_ts, dt):
        """Asynchronously sleep to implement the fixed delay followed by the srvs call."""
        sleep(max(sleep_duration, 0))
        delay = time() - obs_ts
        self.service(req)

        # Publish actual delay
        self.pub_act.publish(Float32(data=delay))
        self.pub_comp.publish(Float32(data=dt))

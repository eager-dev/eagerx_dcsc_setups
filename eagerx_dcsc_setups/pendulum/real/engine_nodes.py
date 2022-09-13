from typing import Optional, Any
import numpy as np
from time import time, sleep
from threading import Thread

# ROS imports
import rospy
from std_msgs.msg import Float32
from dcsc_setups.srv import MopsWrite, MopsWriteRequest, MopsReadRequest, MopsRead

# IMPORT EAGERX
import eagerx
from eagerx import register, Space
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg


class PendulumOutput(eagerx.EngineNode):
    @classmethod
    def make(
        cls, name: str, rate: float, process: Optional[int] = eagerx.process.NEW_PROCESS, color: Optional[str] = "cyan"
    ) -> NodeSpec:
        """PendulumOutput spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["x"]
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        self.service = rospy.ServiceProxy("/mops/read", MopsRead)
        self.service.wait_for_service()

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(dtype="int64"))
    @register.outputs(x=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        response = self.service.call(MopsReadRequest())
        data = [response.sensors.position0 + np.pi, response.sensors.speed]
        return dict(x=np.array(data, dtype="float32"))


class PendulumInput(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        fixed_delay: float = 0.0,
        process: Optional[int] = eagerx.process.NEW_PROCESS,
        color: Optional[str] = "green",
    ):
        """PendulumInput spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(
            name=name, rate=rate, process=process, color=color, inputs=["tick", "u", "x"], outputs=["action_applied"]
        )
        spec.config.fixed_delay = fixed_delay

        # Set component parameter
        spec.inputs.u.window = 1
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        assert spec.config.fixed_delay >= 0, "Delay must be non-negative."
        self.fixed_delay = spec.config.fixed_delay
        self.pub_act = rospy.Publisher("/mops/actuator_delay", Float32, queue_size=1)
        self.pub_comp = rospy.Publisher("/mops/computation_delay", Float32, queue_size=1)
        self.service = rospy.ServiceProxy("/mops/write", MopsWrite)
        self.service.wait_for_service()
        self.prev_seq = None

    @register.states()
    def reset(self):
        self.prev_seq = None

    @register.inputs(tick=Space(dtype="int64"), u=Space(dtype="float32"), x=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
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

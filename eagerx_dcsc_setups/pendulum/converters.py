# ROS IMPORTS
from std_msgs.msg import Float32MultiArray

# RX IMPORTS
import eagerx.core.register as register
from eagerx import Processor, SpaceConverter
from eagerx.core.specs import ConverterSpec
import numpy as np
from gym.spaces import Box


class AngleDecomposition(Processor):
    MSG_TYPE = Float32MultiArray

    @staticmethod
    @register.spec("AngleDecomposition", Processor)
    def spec(spec, angle_idx: int = 0):
        spec.initialize(AngleDecomposition)
        spec.config.angle_idx = angle_idx

    def initialize(self, angle_idx=0):
        assert angle_idx == 0, "Converter currently only works when angle_idx = 0."
        self.angle_idx = angle_idx

    def convert(self, msg):
        if msg.data == []:
            return msg
        angle = msg.data[self.angle_idx]
        new_data = np.concatenate(([np.sin(angle), np.cos(angle)], msg.data[self.angle_idx + 1 :]), axis=0)
        return Float32MultiArray(data=new_data)


class Space_AngleDecomposition(SpaceConverter):
    MSG_TYPE_A = np.ndarray
    MSG_TYPE_B = Float32MultiArray

    @staticmethod
    @register.spec("Space_AngleDecomposition", SpaceConverter)
    def spec(spec: ConverterSpec, low=None, high=None, dtype="float32"):
        # Initialize spec with default arguments
        spec.initialize(Space_AngleDecomposition)
        params = dict(low=low, high=high, dtype=dtype)
        spec.config.update(params)

    def initialize(self, low=None, high=None, dtype="float32"):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.dtype = dtype

    def get_space(self):
        return Box(self.low, self.high, dtype=self.dtype)

    def A_to_B(self, msg):
        return Float32MultiArray(data=msg)

    def B_to_A(self, msg):
        angle = msg.data[0]
        return np.concatenate(([np.sin(angle), np.cos(angle)], msg.data[1:]), axis=0)

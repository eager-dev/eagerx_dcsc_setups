import numpy as np
from eagerx import Processor
from eagerx.core.specs import ProcessorSpec


class AngleDecomposition(Processor):
    @classmethod
    def make(cls, dtype: str = "float32", index: int = 0) -> ProcessorSpec:
        """Make AngleDecomposition processor spec.

        :param dtype: Data type of msg
        :param index: Index of entry that contains angle that should be decomposed.
        :return: Specification
        """
        spec = cls.get_specification()
        spec.config.dtype = dtype
        spec.config.index = index
        return spec

    def initialize(self, spec: ProcessorSpec):
        self.dtype = spec.config.dtype
        self.index = spec.config.index

    def convert(self, data: np.ndarray):
        angle = data[self.index]
        processed_data = np.concatenate((data[: self.index], [np.cos(angle), np.sin(angle)], data[self.index + 1 :]))
        return processed_data.astype(self.dtype)

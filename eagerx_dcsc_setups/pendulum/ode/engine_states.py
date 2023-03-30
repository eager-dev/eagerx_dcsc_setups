from typing import Any
import eagerx
from eagerx.core.specs import EngineStateSpec


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

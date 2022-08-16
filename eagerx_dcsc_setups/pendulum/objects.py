import numpy as np
from typing import List

# EAGERx IMPORTS
import eagerx
from eagerx import register, Space
from eagerx_reality.engine import RealEngine
from eagerx_ode.engine import OdeEngine
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph


class Pendulum(eagerx.Object):
    @classmethod
    @register.sensors(
        x=Space(low=[-1, -1, -999], high=[1, 1, 999], shape=(3,), dtype="float32"),
        image=Space(dtype="uint8"),
        action_applied=Space(low=[-2], high=[2], dtype="float32"),
    )
    @register.actuators(u=Space(low=[-2], high=[2], dtype="float32"))
    @register.engine_states(
        model_state=Space(low=[-np.pi, -9], high=[np.pi, 9], dtype="float32"), model_parameters=Space(dtype="float32")
    )
    def make(
        cls,
        name: str,
        actuators: List[str] = None,
        sensors: List[str] = None,
        states: List[str] = None,
        sensor_rate: float = 30,
        image_rate: float = 15,
        actuator_rate: float = 30,
        render_shape: List[int] = None,
        camera_index: int = 2,
        Dfun: str = None,
    ) -> ObjectSpec:
        """Agnostic definition of the Pendulum"""
        spec = cls.get_specification()

        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["x", "image"]
        spec.config.actuators = ["u"]
        spec.config.states = states if states else ["model_state"]

        # Add custom agnostic params
        spec.config.render_shape = render_shape if render_shape else [480, 480]
        spec.config.render_fn = "pendulum_render_fn"
        spec.config.camera_index = camera_index
        spec.config.Dfun = Dfun if Dfun else "eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_dfun"

        # Set observation properties: (rate, etc...)
        spec.sensors.action_applied.rate = sensor_rate
        spec.sensors.image.rate = image_rate
        from eagerx_dcsc_setups.pendulum.processors import AngleDecomposition

        spec.sensors.x.rate = sensor_rate
        spec.sensors.x.processor = AngleDecomposition.make(index=0, dtype="float32")

        # Set image space
        shape = (spec.config.render_shape[0], spec.config.render_shape[1], 3)
        spec.sensors.image.space = Space(low=0, high=255, shape=shape, dtype="uint8")

        # Set actuator properties: (rate, etc...)
        spec.actuators.u.rate = actuator_rate

        # Set model_parameters properties: (space_converters) # [J, m, l, b, K, R, c, d]
        fixed = [
            0.000159931461600856,
            0.75 * 0.0508581731919534,
            0.0415233722862552,
            0.75 * 1.43298488358436e-05,
            0.0333391179016334,
            7.73125142447252,
            0.75 * 0.000975041213361349,
            165.417960777425,
        ]
        # diff = [0.00, 0.00, 0.00, 0.25, 0.00, 0.25, 0.00]  # Percentual delta with respect to fixed value & scale damping with 0.5
        diff = [0.00, 0.25, 0.00, 0.25, 0.00, 0.00, 0.25, 0.00]  # Percentual delta with respect to fixed value
        low = [val - diff * val for val, diff in zip(fixed, diff)]
        high = [val + diff * val for val, diff in zip(fixed, diff)]
        spec.states.model_parameters.space = Space(low=low, high=high, dtype="float32")
        return spec

    @staticmethod
    @register.engine(OdeEngine)  # This decorator pre-initializes engine implementation with default object_params
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeEngine) of the object."""
        # Set object arguments (nothing to set here in this case)
        spec.engine.ode = "eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_ode"
        spec.engine.Dfun = spec.config.Dfun

        # Set default params of pendulum ode [J, m, l, b, K, R, c, d].
        spec.engine.ode_params = [
            0.000159931461600856,
            0.0508581731919534,
            0.0415233722862552,
            1.43298488358436e-05,
            0.0333391179016334,
            7.73125142447252,
            0.000975041213361349,
            165.417960777425,
        ]

        # Create engine_states (no agnostic states defined in this case)
        from eagerx_ode.engine_states import OdeEngineState, OdeParameters
        from eagerx_dcsc_setups.pendulum.ode.engine_nodes import CustomOdeInput
        from eagerx_ode.engine_nodes import OdeOutput, OdeRender

        # Create engine_states (no agnostic states defined in this case)
        spec.engine.states.model_state = OdeEngineState.make()
        spec.engine.states.model_parameters = OdeParameters.make(list(range(7)))

        # Create sensor engine nodes
        obs = OdeOutput.make("x", rate=spec.sensors.x.rate, process=2)
        render_fn = f"eagerx_dcsc_setups.pendulum.ode.pendulum_render/{spec.config.render_fn}"
        shape = spec.sensors.image.space.shape[:2]
        image = OdeRender.make(
            "image",
            shape=shape,
            render_fn=render_fn,
            rate=spec.sensors.image.rate,
            process=eagerx.process.ENVIRONMENT,
        )

        # Create actuator engine nodes
        action = CustomOdeInput.make(
            "pendulum_actuator",
            rate=spec.actuators.u.rate,
            default_action=[0],
        )

        # Connect all engine nodes
        graph.add([obs, image, action])
        graph.connect(source=obs.outputs.observation, sensor="x")
        graph.connect(source=obs.outputs.observation, target=image.inputs.observation)
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(source=action.outputs.action_applied, target=image.inputs.action_applied, skip=True)
        graph.connect(actuator="u", target=action.inputs.action)

    @staticmethod
    @register.engine(RealEngine)  # This decorator pre-initializes engine implementation with default object_params
    def real_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (RealEngine) of the object."""
        # Import any object specific entities for this engine
        from eagerx_reality.enginenodes import CameraRender
        from eagerx_dcsc_setups.pendulum.real.engine_nodes import PendulumInput, PendulumOutput
        from eagerx_dcsc_setups.pendulum.real.engine_states import RandomActionAndSleep, DummyState

        # Couple engine states
        spec.engine.states.model_state = RandomActionAndSleep.make(sleep_time=1.0, repeat=1)
        spec.engine.states.model_parameters = DummyState.make()

        # Create sensor engine nodes
        # Rate=None, because we will connect them to sensors (thus uses the rate set in the agnostic specification)
        obs = PendulumOutput.make("x", rate=spec.sensors.x.rate, process=0)
        image = CameraRender.make(
            "image",
            camera_idx=spec.config.camera_index,
            shape=spec.config.render_shape,
            rate=spec.sensors.image.rate,
            process=eagerx.process.NEW_PROCESS,
        )
        action = PendulumInput.make("u", rate=spec.actuators.u.rate, fixed_delay=0)

        # Connect all engine nodes
        graph.add([obs, image, action])
        graph.connect(source=obs.outputs.x, sensor="x")
        graph.connect(actuator="u", target=action.inputs.u)
        graph.connect(source=obs.outputs.x, target=action.inputs.x)
        graph.connect(source=image.outputs.image, sensor="image")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)

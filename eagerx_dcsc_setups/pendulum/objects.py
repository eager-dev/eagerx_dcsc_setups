import numpy as np
from typing import List

# EAGERx IMPORTS
import eagerx
from eagerx import register, Space
from eagerx_reality.engine import RealEngine
from eagerx_ode.engine import OdeEngine
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
from eagerx.engines.openai_gym.engine import GymEngine


class Pendulum(eagerx.Object):
    @classmethod
    @register.sensors(
        x=Space(low=[-1, -1, -999], high=[1, 1, 999], shape=(3,), dtype="float32"),
        image=Space(dtype="uint8"),
        action_applied=Space(low=[-2], high=[2], dtype="float32"),
    )
    @register.actuators(u=Space(low=[-2], high=[2], dtype="float32"))
    @register.engine_states(
        model_state=Space(low=[-np.pi, -9], high=[np.pi, 9], dtype="float32"),
        model_parameters=Space(dtype="float32"),
        mass=Space(low=0.04, high=0.04, shape=(), dtype="float32"),
        length=Space(low=0.12, high=0.12, shape=(), dtype="float32"),
        max_speed=Space(low=22, high=22, shape=(), dtype="float32"),
        dt=Space(low=0.05, high=0.05, shape=(), dtype="float32"),
    )
    def make(
        cls,
        name: str,
        actuators: List[str] = None,
        sensors: List[str] = None,
        states: List[str] = None,
        sensor_rate: float = 20,
        image_rate: float = 15,
        actuator_rate: float = 60,
        render_shape: List[int] = None,
        camera_index: int = 2,
        Dfun: str = None,
    ) -> ObjectSpec:
        """Agnostic definition of the Pendulum"""
        spec = cls.get_specification()

        spec.config.name = name
        spec.config.seed = 1
        spec.config.sensors = sensors if sensors else ["x", "image"]
        spec.config.actuators = actuators if actuators else ["u"]
        spec.config.states = states if states else ["model_state"]

        # Add custom agnostic params
        spec.config.render_shape = render_shape if render_shape else [480, 480]
        spec.config.render_fn = "pendulum_render_fn"
        spec.config.camera_index = camera_index
        spec.config.Dfun = Dfun if Dfun else "eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_dfun"

        # Set observation properties: (rate, etc...)
        spec.sensors.action_applied.rate = actuator_rate
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
            0.0508581731919534,
            0.0415233722862552,
            1.43298488358436e-05,
            0.0333391179016334,
            7.73125142447252,
            0.000975041213361349,
            165.417960777425,
        ]
        diff = [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # Percentual delta with respect to fixed value
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
        from eagerx_dcsc_setups.pendulum.ode.engine_states import DummyState

        # Create engine_states (no agnostic states defined in this case)
        spec.engine.states.model_state = OdeEngineState.make()
        spec.engine.states.model_parameters = OdeParameters.make(list(range(7)))
        spec.engine.states.mass = DummyState.make()
        spec.engine.states.length = DummyState.make()
        spec.engine.states.max_speed = DummyState.make()
        spec.engine.states.dt = DummyState.make()

        # Create sensor engine nodes
        obs = OdeOutput.make("x", rate=spec.sensors.x.rate, process=2)
        render_fn = f"eagerx_dcsc_setups.pendulum.ode.pendulum_render/{spec.config.render_fn}"
        shape = spec.sensors.image.space.shape[:2]
        image = OdeRender.make(
            "image",
            shape=shape,
            render_fn=render_fn,
            rate=spec.sensors.image.rate,
            process=eagerx.NEW_PROCESS,
        )

        # Create actuator engine nodes
        action = CustomOdeInput.make(
            "u",
            rate=spec.actuators.u.rate,
            default_action=[0],
            delay_state=True,
        )

        # Connect all engine nodes
        graph.add([obs, image, action])
        graph.connect(source=obs.outputs.observation, sensor="x")
        graph.connect(source=obs.outputs.observation, target=image.inputs.observation)
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(source=action.outputs.action_applied, target=image.inputs.action_applied, skip=True)
        graph.connect(actuator="u", target=action.inputs.action)
        graph.connect(source=action.outputs.action_applied, sensor="action_applied")

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
        # spec.engine.states.model_state = DummyState.make()
        spec.engine.states.model_parameters = DummyState.make()
        spec.engine.states.mass = DummyState.make()
        spec.engine.states.length = DummyState.make()
        spec.engine.states.max_speed = DummyState.make()
        spec.engine.states.dt = DummyState.make()

        # Create sensor engine nodes
        # Rate=None, because we will connect them to sensors (thus uses the rate set in the agnostic specification)
        obs = PendulumOutput.make("x", rate=spec.sensors.x.rate, process=eagerx.NEW_PROCESS)
        image = CameraRender.make(
            "image",
            camera_idx=spec.config.camera_index,
            shape=spec.config.render_shape,
            rate=spec.sensors.image.rate,
            process=eagerx.NEW_PROCESS,
        )
        action = PendulumInput.make("u", rate=spec.actuators.u.rate, fixed_delay=0, process=eagerx.NEW_PROCESS)

        # Connect all engine nodes
        graph.add([obs, image, action])
        graph.connect(source=obs.outputs.x, sensor="x")
        graph.connect(actuator="u", target=action.inputs.u)
        graph.connect(source=obs.outputs.x, target=action.inputs.x)
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(source=action.outputs.action_applied, sensor="action_applied")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)

    @register.engine(GymEngine)
    def gym_engine(spec: eagerx.specs.ObjectSpec, graph: eagerx.EngineGraph):
        """Engine-specific implementation (GymEngine) of the Pendulum object."""
        # Import the openai engine-specific nodes (ObservationSensor, ActionActuator, GymImage)
        from eagerx.engines.openai_gym.enginenodes import ObservationSensor, ActionActuator, GymImage

        # Import the tutorial engine-specific nodes (FloatOutput)
        from eagerx_dcsc_setups.pendulum.gym.engine_nodes import FloatOutput

        # Set engine-specific parameters
        spec.engine.env_id = "Pendulum-v1"
        # spec.engine.seed = spec.config.seed

        # Create engine states that implement the registered states
        # Note: The GymEngine implementation unfortunately does not support setting the OpenAI environment state,
        #       nor does it support changing the dynamic parameters.
        #       However, you could create an Engine specifically for the Pendulum-v1 environment.
        from eagerx_dcsc_setups.pendulum.gym.engine_states import DummyState, SetGymAttribute

        spec.engine.states.model_state = DummyState.make()  # Use dummy state, so it can still be selected.
        spec.engine.states.model_parameters = DummyState.make()  # Use dummy state (same reason as above).
        spec.engine.states.mass = SetGymAttribute.make(attribute="m")
        spec.engine.states.length = SetGymAttribute.make(attribute="l")
        spec.engine.states.max_speed = SetGymAttribute.make(attribute="max_speed")
        spec.engine.states.dt = SetGymAttribute.make(attribute="dt")

        # Create sensor engine nodes.
        image = GymImage.make("image", rate=spec.sensors.image.rate, shape=spec.config.render_shape)
        obs = ObservationSensor.make("obs", rate=spec.sensors.x.rate, process=2)

        from eagerx_dcsc_setups.pendulum.gym.processor import ObsWithDecomposedAngle

        obs.outputs.observation.processor = ObsWithDecomposedAngle.make(convert_to="theta_theta_dot")

        x = FloatOutput.make("x", rate=spec.sensors.x.rate, idx=[0, 1])
        # Create actuator engine node
        # action = ActionActuator.make("u", rate=spec.actuators.u.rate, process=2, zero_action=[0], delay_state=True)
        action = ActionActuator.make("u", rate=spec.actuators.u.rate, process=2, zero_action=[0])

        # Add all engine nodes to the engine-specific graph
        graph.add([obs, x, image, action])

        # x
        graph.connect(source=obs.outputs.observation, target=x.inputs.observation_array)
        graph.connect(source=x.outputs.observation, sensor="x")

        # image
        graph.connect(source=image.outputs.image, sensor="image")

        # u
        # Note: not to be confused with sensor "u", for which we do not provide an implementation here.
        # Note: We add a processor that negates the action, as the torque in OpenAI gym is defined counter-clockwise.
        from eagerx_dcsc_setups.pendulum.gym.processor import VoltageToMotorTorque

        action.inputs.action.processor = VoltageToMotorTorque.make(K=0.03333, R=7.731)
        action.outputs.action_applied.processor = VoltageToMotorTorque.make(K=7.731, R=0.03333)
        graph.connect(actuator="u", target=action.inputs.action)
        graph.connect(source=action.outputs.action_applied, sensor="action_applied")

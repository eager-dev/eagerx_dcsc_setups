# ROS IMPORTS
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

# EAGERx IMPORTS
from eagerx_reality.engine import RealEngine
from eagerx_ode.engine import OdeEngine
from eagerx import Object, EngineNode, SpaceConverter, EngineState, process
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Pendulum(Object):
    entity_id = "Pendulum"

    @staticmethod
    @register.sensors(x=Float32MultiArray, image=Image, action_applied=Float32MultiArray)
    @register.actuators(u=Float32MultiArray)
    @register.engine_states(model_state=Float32MultiArray, model_parameters=Float32MultiArray)
    @register.config(
        always_render=False,
        render_shape=[480, 480],
        camera_index=0,
        Dfun="eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_dfun",
        fixed_delay=0.0,
    )
    def agnostic(spec: ObjectSpec, sensor_rate, image_rate, actuator_rate):
        """Agnostic definition of the Pendulum"""
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.x.rate = sensor_rate
        spec.sensors.x.space_converter = SpaceConverter.make(
            "Space_AngleDecomposition", low=[-1, -1, -9], high=[1, 1, 9], dtype="float32"
        )

        spec.sensors.action_applied.rate = sensor_rate
        spec.sensors.action_applied.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-2], high=[2], dtype="float32"
        )

        spec.sensors.image.rate = image_rate
        spec.sensors.image.space_converter = SpaceConverter.make(
            "Space_Image", low=0, high=1, shape=spec.config.render_shape, dtype="float32"
        )

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.u.rate = actuator_rate
        spec.actuators.u.space_converter = spec.sensors.action_applied.space_converter

        # Set model_state properties: (space_converters)
        spec.states.model_state.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3.14159265359, -9], high=[3.14159265359, 9], dtype="float32"
        )

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
        # diff = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05]  # Percentual delta with respect to fixed value
        # diff = [0.00, 0.00, 0.00, 0.25, 0.00, 0.25, 0.00]  # Percentual delta with respect to fixed value & scale damping with 0.5
        diff = [0.00, 0.25, 0.00, 0.25, 0.00, 0.00, 0.25, 0.00]  # Percentual delta with respect to fixed value
        low = [val - diff * val for val, diff in zip(fixed, diff)]
        high = [val + diff * val for val, diff in zip(fixed, diff)]
        spec.states.model_parameters.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=low, high=high, dtype="float32"
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
        name: str,
        sensors=None,
        states=None,
        sensor_rate=30,
        actuator_rate=30,
        image_rate=15,
        always_render=False,
        render_shape=None,
        camera_index=2,
        Dfun="eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_dfun",
        fixed_delay=0.0,
    ):
        """Object spec of Pendulum"""
        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["x", "image"]
        spec.config.actuators = ["u"]
        spec.config.states = states if states else ["model_state"]

        # Add registered agnostic params
        spec.config.always_render = always_render
        spec.config.render_shape = render_shape if render_shape else [480, 480]
        spec.config.camera_index = camera_index
        spec.config.Dfun = Dfun
        spec.config.fixed_delay = fixed_delay

        # Add engine implementation
        Pendulum.agnostic(spec, sensor_rate, image_rate, actuator_rate)

    @staticmethod
    @register.engine(entity_id, OdeEngine)  # This decorator pre-initializes engine implementation with default object_params
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeEngine) of the object."""
        # Import any object specific entities for this engine
        import eagerx_dcsc_setups.pendulum.ode  # noqa # pylint: disable=unused-import

        # Set object arguments (nothing to set here in this case)
        spec.OdeEngine.ode = "eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_ode"
        spec.OdeEngine.Dfun = spec.config.Dfun

        # Set default params of pendulum ode [J, m, l, b, K, R, c, d].
        spec.OdeEngine.ode_params = [
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
        spec.OdeEngine.states.model_state = EngineState.make("OdeEngineState")
        spec.OdeEngine.states.model_parameters = EngineState.make("OdeParameters", list(range(7)))

        # Create sensor engine nodes
        obs = EngineNode.make("OdeOutput", "x", rate=spec.sensors.x.rate, process=2)
        image = EngineNode.make(
            "OdeRender",
            "image",
            shape=spec.config.render_shape,
            render_fn="eagerx_dcsc_setups.pendulum.ode.pendulum_render/pendulum_render_fn",
            rate=spec.sensors.image.rate,
            process=process.NEW_PROCESS,
        )

        # Create actuator engine nodes
        action = EngineNode.make(
            "CustomOdeInput", "pendulum_actuator", rate=spec.actuators.u.rate, process=2, default_action=[0]
        )

        # Connect all engine nodes
        graph.add([obs, image, action])
        graph.connect(source=obs.outputs.observation, sensor="x")
        graph.connect(source=obs.outputs.observation, target=image.inputs.observation)
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(source=action.outputs.action_applied, target=image.inputs.action_applied, skip=True)
        graph.connect(actuator="u", target=action.inputs.action)

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)

    @staticmethod
    @register.engine(entity_id, RealEngine)  # This decorator pre-initializes engine implementation with default object_params
    def real_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (RealEngine) of the object."""
        # Import any object specific entities for this engine
        import eagerx_dcsc_setups.pendulum.real  # noqa # pylint: disable=unused-import

        # Couple engine states
        spec.RealEngine.states.model_state = EngineState.make("RandomActionAndSleep", sleep_time=1.0, repeat=1)
        spec.RealEngine.states.model_parameters = EngineState.make("Dummy")

        # Create sensor engine nodes
        # Rate=None, because we will connect them to sensors (thus uses the rate set in the agnostic specification)
        obs = EngineNode.make("PendulumOutput", "x", rate=spec.sensors.x.rate, process=0)
        image = EngineNode.make(
            "CameraRender",
            "image",
            camera_idx=spec.config.camera_index,
            shape=spec.config.render_shape,
            rate=spec.sensors.image.rate,
            process=process.NEW_PROCESS,
        )
        action = EngineNode.make("PendulumInput", "u", rate=spec.actuators.u.rate, fixed_delay=spec.config.fixed_delay)

        # Connect all engine nodes
        graph.add([obs, image, action])
        graph.connect(source=obs.outputs.x, sensor="x")
        graph.connect(actuator="u", target=action.inputs.u)
        graph.connect(source=obs.outputs.x, target=action.inputs.x)
        graph.connect(source=image.outputs.image, sensor="image")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)

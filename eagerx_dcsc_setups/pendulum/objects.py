# ROS IMPORTS
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

# EAGERx IMPORTS
from eagerx_reality.bridge import RealBridge
from eagerx_ode.bridge import OdeBridge
from eagerx import Object, EngineNode, SpaceConverter, EngineState, process
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Pendulum(Object):
    entity_id = "Pendulum"

    @staticmethod
    @register.sensors(x=Float32MultiArray, action_applied=Float32MultiArray, image=Image)
    @register.actuators(u=Float32MultiArray)
    @register.engine_states(model_state=Float32MultiArray, model_parameters=Float32MultiArray)
    @register.config(
        always_render=False,
        render_shape=[480, 480],
        camera_index=0,
        Dfun="eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_dfun",
    )
    def agnostic(spec: ObjectSpec, rate):
        """Agnostic definition of the Pendulum"""
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.x.rate = rate
        spec.sensors.x.space_converter = SpaceConverter.make(
            "Space_AngleDecomposition", low=[-1, -1, -9], high=[1, 1, 9], dtype="float32"
        )

        spec.sensors.action_applied.rate = rate
        spec.sensors.action_applied.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3], high=[3], dtype="float32"
        )

        spec.sensors.image.rate = rate/2
        spec.sensors.image.space_converter = SpaceConverter.make(
            "Space_Image", low=0, high=1, shape=spec.config.render_shape, dtype="float32"
        )

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.u.rate = 10 * rate
        spec.actuators.u.window = 1
        spec.actuators.u.space_converter = SpaceConverter.make("Space_Float32MultiArray", low=[-3], high=[3], dtype="float32")

        # Set model_state properties: (space_converters)
        spec.states.model_state.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3.14159265359, -9], high=[3.14159265359, 9], dtype="float32"
        )

        # Set model_parameters properties: (space_converters) # [J, m, l, b, K, R, c, d]
        fixed = [0.000159931461600856, 0.0508581731919534, 0.0415233722862552, 1.43298488358436e-05, 0.0333391179016334,
                 7.73125142447252, 0.000975041213361349, 165.417960777425]
        diff = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05]  # Percentual delta with respect to fixed value
        low = [val - diff * val for val, diff in zip(fixed, diff)]
        high = [val + diff * val for val, diff in zip(fixed, diff)]
        # low = [1.7955e-04, 5.3580e-02, 4.1610e-02, 1.3490e-04, 4.7690e-02, 9.3385e+00, 1.4250e+00, 1.7480e-03]
        # high = [1.98450e-04, 5.92200e-02, 4.59900e-02, 1.49100e-04, 5.27100e-02, 1.03215e+01, 1.57500e+00, 1.93200e-03]
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
        rate=30,
        always_render=False,
        render_shape=None,
        camera_index=2,
        Dfun="eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_dfun",
    ):
        """Object spec of Pendulum"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        Pendulum.initialize_spec(spec)

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["x", "action_applied", "image"]
        spec.config.actuators = ["u"]
        spec.config.states = states if states else ["model_state"]

        # Add registered agnostic params
        spec.config.always_render = always_render
        spec.config.render_shape = render_shape if render_shape else [480, 480]
        spec.config.camera_index = camera_index
        spec.config.Dfun = Dfun

        # Add bridge implementation
        Pendulum.agnostic(spec, rate)

    @staticmethod
    @register.bridge(entity_id, OdeBridge)  # This decorator pre-initializes bridge implementation with default object_params
    def ode_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeBridge) of the object."""
        # Import any object specific entities for this bridge
        import eagerx_dcsc_setups.pendulum.ode  # noqa # pylint: disable=unused-import

        # Set object arguments (nothing to set here in this case)
        spec.OdeBridge.ode = "eagerx_dcsc_setups.pendulum.ode.pendulum_ode/pendulum_ode"
        spec.OdeBridge.Dfun = spec.config.Dfun

        # Set default params of pendulum ode [J, m, l, b, K, R, c, d].
        spec.OdeBridge.ode_params =  [0.000159931461600856, 0.0508581731919534, 0.0415233722862552,
                                      1.43298488358436e-05, 0.0333391179016334, 7.73125142447252, 0.000975041213361349,
                                      165.417960777425]

        # Create engine_states (no agnostic states defined in this case)
        spec.OdeBridge.states.model_state = EngineState.make("OdeEngineState")
        spec.OdeBridge.states.model_parameters = EngineState.make("OdeParameters", list(range(7)))

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
        action = EngineNode.make("OdeInput", "pendulum_actuator", rate=spec.actuators.u.rate, process=2, default_action=[0])

        # Connect all engine nodes
        graph.add([obs, image, action])
        graph.connect(source=obs.outputs.observation, sensor="x")
        graph.connect(source=obs.outputs.observation, target=image.inputs.observation)
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(source=action.outputs.action_applied, target=image.inputs.action_applied, skip=True)
        graph.connect(actuator="u", target=action.inputs.action)

        # Add action applied
        applied = EngineNode.make("ActionApplied", "applied", rate=spec.sensors.action_applied.rate, process=2)
        graph.add(applied)
        graph.connect(source=action.outputs.action_applied, target=applied.inputs.action_applied, skip=True)
        graph.connect(source=applied.outputs.action_applied, sensor="action_applied")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)

    @staticmethod
    @register.bridge(entity_id, RealBridge)  # This decorator pre-initializes bridge implementation with default object_params
    def real_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (RealBridge) of the object."""
        # Import any object specific entities for this bridge
        import eagerx_dcsc_setups.pendulum.real  # noqa # pylint: disable=unused-import

        # Couple engine states
        spec.RealBridge.states.model_state = EngineState.make("RandomActionAndSleep", sleep_time=1.0, repeat=1)

        # Create sensor engine nodes
        # Rate=None, because we will connect them to sensors (thus uses the rate set in the agnostic specification)
        obs = EngineNode.make("PendulumOutput", "x", rate=spec.sensors.x.rate)
        applied = EngineNode.make("ActionApplied", "applied", rate=spec.sensors.action_applied.rate, process=0)
        image = EngineNode.make(
            "CameraRender",
            "image",
            camera_idx=spec.config.camera_index,
            shape=spec.config.render_shape,
            rate=spec.sensors.image.rate,
            process=process.NEW_PROCESS,
        )
        action = EngineNode.make("PendulumInput", "u", rate=spec.sensors.x.rate)

        # Create actuator engine nodes
        # Rate=None, because we will connect it to an actuator (thus uses the rate set in the agnostic specification)
        action_delay = EngineNode.make(
            "ConstantDelayAction",
            "action_delay",
            rate=spec.actuators.u.rate,
            desired_delay=1/(2*spec.sensors.x.rate),
        )

        # Connect all engine nodes
        graph.add([obs, applied, image, action, action_delay])
        graph.connect(source=obs.outputs.x, sensor="x")
        graph.connect(actuator="u", target=action_delay.inputs.action)
        graph.connect(source=obs.outputs.x, target=action_delay.inputs.observation)
        graph.connect(source=action_delay.outputs.delayed_action, target=action.inputs.u)
        graph.connect(source=obs.outputs.x, target=action.inputs.x)
        graph.connect(source=action.outputs.action_applied, target=applied.inputs.action_applied, skip=True)
        graph.connect(source=applied.outputs.action_applied, sensor="action_applied")
        graph.connect(source=image.outputs.image, sensor="image")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)

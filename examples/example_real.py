# ROS packages required
import eagerx
from eagerx import Object, Bridge, Node, process

eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
from eagerx.wrappers import Flatten

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_ode  # Registers OdeBridge # noqa # pylint: disable=unused-import
import eagerx_reality  # Registers RealBridge # noqa # pylint: disable=unused-import
import eagerx_dcsc_setups.pendulum  # Registers Pendulum # noqa # pylint: disable=unused-import

# Other
import numpy as np
import rospy
import stable_baselines3 as sb
from functools import partial

def make_graph(
        sensor_rate: float,
        actuator_rate: float,
        image_rate: float,
        evaluation: bool,
):
    u_limit = 2.
    states = ["model_state"]

    # Create empty graph
    graph = Graph.create()

    # Create pendulum
    pendulum = Object.make(
        "Pendulum",
        "pendulum",
        render_shape=[480, 480],
        sensors=["x"],
        states=states,
        sensor_rate=sensor_rate,
        actuator_rate=actuator_rate,
        image_rate=image_rate,
    )
    graph.add(pendulum)

    # Create Butterworth filter
    bf = Node.make(
        "ButterworthFilter",
        name="butterworth_filter",
        rate=sensor_rate,
        N=2,
        Wn=13,
        process=process.NEW_PROCESS,
    )
    bf.inputs.signal.space_converter = pendulum.actuators.u.space_converter
    bf.outputs.filtered.space_converter = pendulum.actuators.u.space_converter
    graph.add(bf)

    if evaluation:
        reset = eagerx.ResetNode.make("ResetAngle", "reset_angle", sensor_rate, u_range=[-u_limit, +u_limit])
        graph.add(reset)

        graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
        graph.connect(action="voltage", target=reset.feedthroughs.u)
        graph.connect(source=reset.outputs.u, target=bf.inputs.signal)
        graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)
    else:
        graph.connect(action="voltage", target=bf.inputs.signal)
    graph.connect(source=bf.outputs.filtered, target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, observation="angle_data")
    graph.connect(source=bf.outputs.filtered, observation="action_applied", skip=True, initial_obs=[0], window=1)

    # Add rendering
    graph.add_component(pendulum.sensors.image)
    layover = Node.make("Overlay", "overlay", rate=image_rate)
    graph.add(layover)
    graph.connect(source=pendulum.sensors.x, target=layover.inputs.x)
    graph.connect(source=bf.outputs.filtered, target=layover.inputs.u)
    graph.connect(source=pendulum.sensors.image, target=layover.inputs.base_image)
    graph.render(source=layover.outputs.image, rate=image_rate, display=True)

    return graph, pendulum

if __name__ == "__main__":
    # Define rates
    sensor_rate = 30.
    actuator_rate = 30.
    image_rate = sensor_rate / 2
    bridge_rate = max([sensor_rate, actuator_rate, image_rate])
    algorithm = {"name": "SAC", "sim_params": {}, "real_params": {"ent_coef": "auto_0.1"}}

    seed = 1
    length_train_eps = 100
    length_eval_eps = 270
    sim_eps = 100

    train_graph, pendulum = make_graph(
        sensor_rate=sensor_rate, actuator_rate=actuator_rate, image_rate=image_rate, evaluation=False
    )
    eval_graph, _ = make_graph(
        sensor_rate=sensor_rate, actuator_rate=actuator_rate, image_rate=image_rate, evaluation=True
    )

    # Show in the gui
    train_graph.gui()
    pendulum.gui("OdeBridge")
    pendulum.gui("RealBridge")


    # Define bridges
    bridge_ode = Bridge.make("OdeBridge", rate=bridge_rate, sync=True, real_time_factor=0, process=process.NEW_PROCESS)
    bridge_real = Bridge.make("RealBridge", rate=bridge_rate, sync=True, process=process.NEW_PROCESS)

    # Define step function
    def step_fn(prev_obs, obs, action, steps, length_eps):
        state = obs["angle_data"][0]
        u = action["voltage"][0]

        # Calculate reward
        sin_th, cos_th, thdot = state
        th = np.arctan2(sin_th, cos_th)
        cost = th**2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u ** 2

        # Determine done flag
        done = steps > length_eps
        # Set info:
        info = {"TimeLimit.truncated": done}
        return obs, -cost, done, info

    def reset_fn(env):
        states = env.state_space.sample()
        offset = np.random.rand() - 0.5
        theta = np.pi - offset if offset > 0 else -np.pi - offset
        states["pendulum/model_state"] = np.array([theta, 0], dtype="float32")
        return states

    train_step_fn = partial(step_fn, length_eps=length_train_eps)
    eval_step_fn = partial(step_fn, length_eps=length_eval_eps)

    # Initialize Environment
    simulation_env = Flatten(EagerxEnv(name="ode", rate=sensor_rate, graph=train_graph, bridge=bridge_ode, step_fn=train_step_fn))
    real_env = Flatten(EagerxEnv(name="real", rate=sensor_rate, graph=eval_graph, bridge=bridge_real, step_fn=eval_step_fn, reset_fn=reset_fn))

    # Seed envs
    real_env.seed(seed)
    simulation_env.seed(seed)

    # Initialize learner (kudos to Antonin)
    sb_algorithm = getattr(sb, algorithm["name"])
    model = sb_algorithm("MlpPolicy", simulation_env, verbose=1, seed=seed, **algorithm["sim_params"])

    # First train in simulation
    simulation_env.render("human")
    model.learn(total_timesteps=sim_eps * length_train_eps)
    simulation_env.close()

    # Evaluate for 30 seconds in simulation
    rospy.loginfo("Start simulation evaluation!")
    obs = simulation_env.reset()
    for i in range(int(30 * sensor_rate)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = simulation_env.step(action)
        if done:
            obs = simulation_env.reset()

    model.save("simulation")
    simulation_env.shutdown()

    # Train on real system
    model = sb_algorithm.load("simulation", env=real_env, seed=seed, **algorithm["real_params"])
    real_env.render("human")

    # Evaluate on real system
    rospy.loginfo("Start zero-shot evaluation!")
    while True:
        done = False
        obs = real_env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = real_env.step(action)
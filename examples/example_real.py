# ROS packages required
from eagerx import Object, Bridge, Node, initialize, log, process

initialize("eagerx_core", anonymous=True, log_level=log.INFO)

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


if __name__ == "__main__":
    # Define rates
    sensor_rate = 30.
    actuator_rate = 90.
    image_rate = sensor_rate / 2
    bridge_rate = max([sensor_rate, actuator_rate, image_rate])

    delay = 0  # 1/90 worked.
    seed = 27
    length_eps = 200
    sim_eps = 100

    # Initialize empty graph
    graph = Graph.create()

    # Create pendulum
    pendulum = Object.make(
        "Pendulum",
        "pendulum",
        render_shape=[480, 480],
        sensors=["x"],
        states=["model_state", "model_parameters"],
        sensor_rate=sensor_rate,
        actuator_rate=actuator_rate,
        image_rate=image_rate,
        fixed_delay=delay,
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

    # Connect the nodes
    graph.connect(action="action", target=bf.inputs.signal)
    graph.connect(source=bf.outputs.filtered, target=pendulum.actuators.u, delay=delay)
    graph.connect(source=pendulum.sensors.x, observation="observation")
    # graph.connect(source=bf.outputs.filtered, observation="action_applied", skip=True, initial_obs=[0], window=2)

    # Add rendering
    graph.add_component(pendulum.sensors.image)
    graph.render(source=pendulum.sensors.image, rate=15, display=True)

    # Show in the gui
    # pendulum.gui("OdeBridge")
    # pendulum.gui("RealBridge")
    # graph.gui()

    # Define bridges
    bridge_ode = Bridge.make("OdeBridge", rate=bridge_rate, sync=True, real_time_factor=0, process=process.NEW_PROCESS)
    bridge_real = Bridge.make("RealBridge", rate=bridge_rate, sync=True, process=process.NEW_PROCESS)

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        state = obs["observation"][0]
        u = action["action"][0]

        # Calculate reward
        sin_th, cos_th, thdot = state
        th = np.arctan2(sin_th, cos_th)
        cost = th**2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u ** 2
        # Determine done flag
        done = steps > length_eps
        # Set info:
        info = dict()
        return obs, -cost, done, info

    def reset_fn(env):
        states = env.state_space.sample()
        # states["pendulum/model_parameters"] = env.state_space["pendulum/model_parameters"].low
        return states

    # Initialize Environment
    simulation_env = Flatten(EagerxEnv(name="ode", rate=sensor_rate, graph=graph, bridge=bridge_ode, step_fn=step_fn, reset_fn=reset_fn))
    graph.remove_component(pendulum.states.model_parameters)
    real_env = Flatten(EagerxEnv(name="real", rate=sensor_rate, graph=graph, bridge=bridge_real, step_fn=step_fn))

    # Seed envs
    real_env.seed(seed)
    simulation_env.seed(seed)

    # Initialize learner (kudos to Antonin)
    model = sb.SAC("MlpPolicy", simulation_env, verbose=1, seed=seed)

    # First train in simulation
    simulation_env.render("human")
    model.learn(total_timesteps=sim_eps * length_eps)
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
    model = sb.SAC.load("simulation", env=real_env, ent_coef="auto_0.1", seed=seed)
    real_env.render("human")

    # Evaluate on real system
    rospy.loginfo("Start zero-shot evaluation!")
    obs = real_env.reset()
    for i in range(int(60 * sensor_rate)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = real_env.step(action)
        real_env.render()
        if done:
            obs = real_env.reset()

    # Fine-tune policy
    rospy.loginfo("Start fine-tuning!")
    model.learn(total_timesteps=int(60 * sensor_rate))
    model.save("real")

    # Evaluate on real system
    rospy.loginfo("Start fine-tuned evaluation!")
    obs = real_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = real_env.step(action)
        real_env.render()
        if done:
            obs = real_env.reset()

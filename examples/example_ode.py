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
import eagerx_dcsc_setups.pendulum  # Registers Pendulum # noqa # pylint: disable=unused-import

# Other
import numpy as np
import stable_baselines3 as sb


if __name__ == "__main__":
    # Define rate (depends on rate of ode)
    rate = 30.0

    # Initialize empty graph
    graph = Graph.create()

    # Create pendulum
    pendulum = Object.make("Pendulum", "pendulum", render_shape=[480, 480], sensors=["pendulum_output", "action_applied"],
                       states=["model_state", "model_parameters"])
    graph.add(pendulum)

    # Create Butterworth filter
    bf = Node.make("ButterworthFilter", name="bf", rate=rate, N=2, Wn=13, process=process.NEW_PROCESS)
    graph.add(bf)

    # Connect the nodes
    graph.connect(action="action", target=bf.inputs.signal)
    graph.connect(source=bf.outputs.filtered, target=pendulum.actuators.pendulum_input)
    graph.connect(source=pendulum.sensors.pendulum_output, observation="observation", window=1)
    graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", window=1)

    # Add rendering
    graph.add_component(pendulum.sensors.image)
    graph.render(source=pendulum.sensors.image, rate=10, display=True)

    # Show in the gui
    # graph.gui()

    # Define bridges
    bridge = Bridge.make("OdeBridge", rate=rate, is_reactive=True, real_time_factor=0, process=process.NEW_PROCESS)

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        state = obs["observation"][0]
        # Calculate reward
        sin_th, cos_th, thdot = state
        th = np.arctan2(sin_th, cos_th)
        cost = th**2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2
        # Determine done flag
        done = steps > 500
        # Set info:
        info = dict()
        return obs, -cost, done, info

    # Initialize Environment
    env = Flatten(EagerxEnv(name="rx", rate=rate, graph=graph, bridge=bridge, step_fn=step_fn))

    # Initialize learner (kudos to Antonin)
    model = sb.SAC("MlpPolicy", env, verbose=1)

    # First train in simulation for 5 minutes
    env.render("human")
    model.learn(total_timesteps=int(300 * rate))

    # Evaluate for 30 seconds in simulation
    obs = env.reset()
    eps = 0
    action = env.action_space.sample()
    print(f"Episode {eps}")
    for i in range(int(30 * rate)):
        action, _states = model.predict(obs, deterministic=True)
        if i % 500 == 0:
            eps += 1
            obs = env.reset()
            print(f"Episode {eps}")

    model.save("simulation")

# ROS packages required
from eagerx import Object, Engine, Node, initialize, log, process

initialize("eagerx_core", anonymous=True, log_level=log.INFO)

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
from eagerx.wrappers import Flatten

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_ode  # Registers OdeEngine # noqa # pylint: disable=unused-import
import eagerx_dcsc_setups  # Registers Pendulum # noqa # pylint: disable=unused-import

# Other
import numpy as np
import stable_baselines3 as sb


if __name__ == "__main__":
    # Define rate (depends on rate of ode)
    rate = 30.0

    # Initialize empty graph
    graph = Graph.create()

    # Create pendulum
    pendulum = Object.make(
        "Pendulum",
        "pendulum",
        render_shape=[480, 480],
        sensors=["x"],
        states=["model_state", "model_parameters"],
    )
    # Visualize EngineGraph
    pendulum.gui(engine_id="OdeEngine")

    graph.add(pendulum)

    # Create Butterworth filter
    bf = Node.make(
        "ButterworthFilter",
        name="bf",
        rate=rate,
        N=2,
        Wn=13,
        process=process.NEW_PROCESS,
    )
    graph.add(bf)

    # Connect the nodes
    graph.connect(action="action", target=bf.inputs.signal)
    graph.connect(source=bf.outputs.filtered, target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, observation="observation", window=1)

    # Add rendering
    graph.add_component(pendulum.sensors.image)
    graph.render(source=pendulum.sensors.image, rate=10, display=True)

    # Visualize Graph
    graph.gui()

    # Define engines
    engine = Engine.make("OdeEngine", rate=rate, sync=True, real_time_factor=0, process=process.NEW_PROCESS)

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        state = obs["observation"][0]
        u = action["action"][0]

        # Calculate reward
        sin_th, cos_th, thdot = state
        th = np.arctan2(sin_th, cos_th)
        cost = th**2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u ** 2
        # Determine done flag
        done = steps > 500
        # Set info:
        info = {"TimeLimit.truncated": done}
        return obs, -cost, done, info

    # Initialize Environment
    env = Flatten(
        EagerxEnv(name="ode_env", rate=rate, graph=graph, engine=engine, step_fn=step_fn)
    )

    # Initialize learner (kudos to Antonin)
    model = sb.SAC("MlpPolicy", env, verbose=1)

    # First train in simulation for 5 minutes and save
    env.render("human")
    model.learn(total_timesteps=int(300 * rate))

import eagerx

eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
from eagerx.wrappers import Flatten

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx.converters  # Registers SpaceConverters # noqa # pylint: disable=unused-import
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

    # Show and modify in the gui
    graph.gui()

    # Define bridges
    bridge = eagerx.Bridge.make(
        "OdeBridge",
        rate=rate,
        is_reactive=True,
        real_time_factor=0,
        process=eagerx.process.NEW_PROCESS,
    )

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
    env = Flatten(EagerxEnv(name="gui_env", rate=rate, graph=graph, bridge=bridge, step_fn=step_fn))
    env.render("human")

    # Initialize learner (kudos to Antonin)
    model = sb.SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(300 * rate))

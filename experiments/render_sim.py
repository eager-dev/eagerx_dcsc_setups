# ROS packages required
import eagerx
from eagerx import Engine, process, Graph

eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.wrappers import Flatten

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_reality  # Registers RealEngine # noqa # pylint: disable=unused-import
import eagerx_dcsc_setups.pendulum  # Registers Pendulum # noqa # pylint: disable=unused-import

# Other
import stable_baselines3 as sb
from functools import partial
import experiments.util as util
import os
import numpy as np


if __name__ == "__main__":
    DR = False
    FA = 2
    eval_delay = 1 / 30
    real_delay = None

    # Define constants
    sensor_rate = 30.0
    actuator_rate = 90.0
    image_rate = sensor_rate / 2
    engine_rate = max([sensor_rate, actuator_rate, image_rate])
    seed = 27
    np.random.seed(seed)

    length_eval_eps = 90
    eval_eps = 3
    log_name = "2022-05-09-0659"
    model_name = "bl_ed_fa2_0"
    NAME = "sim"
    LOG_DIR = os.path.dirname(eagerx_dcsc_setups.__file__) + f"/../logs/{NAME}_{log_name}"

    step_fn = partial(util.step_fn, length_eps=length_eval_eps)
    eval_reset_fn = partial(util.eval_reset_fn, eval_delay=eval_delay)

    # Define engines
    engine_ode = Engine.make("OdeEngine", rate=engine_rate, sync=True, real_time_factor=0, process=process.NEW_PROCESS)

    # Create evaluation environment
    graph = Graph.create()
    graph = util.make_graph(
        DR=DR, FA=FA, evaluation=True, sensor_rate=sensor_rate, actuator_rate=actuator_rate, image_rate=image_rate, render=True
    )
    env = Flatten(
        EagerxEnv(name="render_env", rate=sensor_rate, graph=graph, engine=engine_ode, step_fn=step_fn, reset_fn=eval_reset_fn)
    )

    model = sb.SAC.load(f"{LOG_DIR}/models/{model_name}")

    util.eval_env(model, env, eval_eps, gif_file=f"{LOG_DIR}/{model_name}_sim.gif")

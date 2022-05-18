# ROS packages required
from eagerx import Bridge, process, Graph

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.wrappers import Flatten

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_reality  # Registers RealBridge # noqa # pylint: disable=unused-import
import eagerx_dcsc_setups.pendulum  # Registers Pendulum # noqa # pylint: disable=unused-import

# Other
import stable_baselines3 as sb
from functools import partial
import experiments.util as util
import yaml
import os


def evaluate_real(
    sensor_rate,
    bridge_rate,
    delay,
    seed,
    length_eval_eps,
    eval_eps,
    repetitions,
    envs,
    log_name,
    eval_envs=["real", "real_async"],
):
    NAME = "sim"
    LOG_DIR = os.path.dirname(eagerx_dcsc_setups.__file__) + f"/../logs/{NAME}_{log_name}"

    step_fn = partial(util.step_fn, length_eps=length_eval_eps)
    eval_reset_fn = partial(util.eval_reset_fn, eval_delay=None)

    # Define bridges
    bridge_real_sync = Bridge.make("RealBridge", rate=bridge_rate, sync=True, process=process.NEW_PROCESS)
    bridge_real_async = Bridge.make("RealBridge", rate=bridge_rate, sync=False, process=process.NEW_PROCESS)

    cumulative_rewards = {}
    # Create Environments
    for name, params in envs.items():
        # Initalize results dict
        cumulative_rewards[name] = {}
        for eval_env_name in eval_envs:
            cumulative_rewards[name][eval_env_name] = []

        # Create evaluation environment
        graph = Graph.create()
        graph.load(f"{LOG_DIR}/graphs/{name}_eval.yaml")

        eval_env = Flatten(
            EagerxEnv(
                name=name + "_eval_env",
                rate=sensor_rate,
                graph=graph,
                bridge=bridge_real_sync,
                step_fn=step_fn,
                reset_fn=eval_reset_fn,
            )
        )
        eval_env.seed(seed)
        envs[name]["real"] = eval_env

        eval_env_delay = Flatten(
            EagerxEnv(
                name=name + "_eval_env_delay",
                rate=sensor_rate,
                graph=graph,
                bridge=bridge_real_sync,
                step_fn=step_fn,
                reset_fn=eval_reset_fn,
            )
        )
        eval_env_delay.seed(seed)
        envs[name]["real_delay"] = eval_env_delay

        eval_env_async = Flatten(
            EagerxEnv(
                name=name + "_eval_env_async",
                rate=sensor_rate,
                graph=graph,
                bridge=bridge_real_async,
                step_fn=step_fn,
                reset_fn=eval_reset_fn,
            )
        )
        eval_env_async.seed(seed)
        envs[name]["real_async"] = eval_env_async

    # Train and Evaluate
    for repetition in range(repetitions):
        for name, params in envs.items():
            model_name = name + "_{}".format(repetition)

            model = sb.SAC.load(f"{LOG_DIR}/models/{model_name}")

            # Evaluate
            for eval_env_name in eval_envs:
                eval_env = envs[name][eval_env_name]
                real_delay = delay if eval_env_name == "real_delay" else None
                cumulative_reward = util.eval_env(model, eval_env, eval_eps, real_delay=real_delay)
                cumulative_rewards[name][eval_env_name].append(cumulative_reward)

                with open(f"{LOG_DIR}/real_cumulative_rewards_async.yaml", "w") as file:
                    yaml.dump(cumulative_rewards, file)

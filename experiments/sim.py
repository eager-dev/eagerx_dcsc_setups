# ROS packages required
from eagerx import Bridge, process

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.wrappers import Flatten

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_ode  # Registers OdeBridge # noqa # pylint: disable=unused-import
import eagerx_dcsc_setups.pendulum  # Registers Pendulum # noqa # pylint: disable=unused-import

# Other
import stable_baselines3 as sb
from functools import partial
import experiments.util as util
import yaml
from datetime import datetime
import os


def simulate(
    image_rate,
    sensor_rate,
    actuator_rate,
    bridge_rate,
    delay,
    seed,
    length_train_eps,
    length_eval_eps,
    train_eps,
    eval_eps,
    repetitions,
    envs,
):

    NAME = "sim"
    LOG_DIR = os.path.dirname(eagerx_dcsc_setups.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"
    os.mkdir(LOG_DIR)
    os.mkdir(LOG_DIR + "/graphs")
    os.mkdir(LOG_DIR + "/models")

    train_step_fn = partial(util.step_fn, length_eps=length_train_eps)
    eval_step_fn = partial(util.step_fn, length_eps=length_eval_eps)

    # Define bridges
    bridge_ode_sync = Bridge.make("OdeBridge", rate=bridge_rate, sync=True, real_time_factor=0, process=process.NEW_PROCESS)
    bridge_ode_async = Bridge.make("OdeBridge", rate=bridge_rate, sync=False, real_time_factor=0, process=process.NEW_PROCESS)

    cumulative_rewards = {}
    # Create Environments
    for name, params in envs.items():
        # Initalize results dict
        cumulative_rewards[name] = {}
        for eval_env_name in params["evaluate_on"]:
            cumulative_rewards[name][eval_env_name] = []

        # Choose bridge
        bridge_ode = bridge_ode_sync if params["sync"] else bridge_ode_async

        # Create train environment
        train_graph = util.make_graph(
            DR=params["dr"],
            FA=params["fa"],
            evaluation=False,
            sensor_rate=sensor_rate,
            actuator_rate=actuator_rate,
            image_rate=image_rate,
        )
        train_graph.save(f"{LOG_DIR}/graphs/{name}_train.yaml")
        train_delay = delay if params["ed"] else None
        train_reset_fn = partial(util.train_reset_fn, train_delay=train_delay)
        train_env = Flatten(
            EagerxEnv(
                name=name + "_train_env",
                rate=sensor_rate,
                graph=train_graph,
                bridge=bridge_ode,
                step_fn=train_step_fn,
                reset_fn=train_reset_fn,
            )
        )
        train_env.seed(seed)
        envs[name]["train_env"] = train_env

        # Create evaluation environment
        eval_graph = util.make_graph(
            DR=params["dr"],
            FA=params["fa"],
            evaluation=True,
            sensor_rate=sensor_rate,
            actuator_rate=actuator_rate,
            image_rate=image_rate,
        )
        eval_graph.save(f"{LOG_DIR}/graphs/{name}_eval.yaml")
        eval_delay = delay if params["ed"] else None
        eval_reset_fn = partial(util.eval_reset_fn, eval_delay=eval_delay)
        eval_env = Flatten(
            EagerxEnv(
                name=name + "_eval_env",
                rate=sensor_rate,
                graph=eval_graph,
                bridge=bridge_ode,
                step_fn=eval_step_fn,
                reset_fn=eval_reset_fn,
            )
        )
        eval_env.seed(seed)
        envs[name]["eval_env"] = eval_env

    # Train and Evaluate
    for repetition in range(repetitions):
        for name, params in envs.items():
            model_name = name + "_{}".format(repetition)

            # Initialize learner (kudos to Antonin)
            model = sb.SAC("MlpPolicy", params["train_env"], verbose=1, seed=seed, tensorboard_log=f"{LOG_DIR}/tensorboard")

            # Train
            model.learn(total_timesteps=train_eps * length_train_eps, tb_log_name=model_name)
            model.save(f"{LOG_DIR}/models/{model_name}")

            # Evaluate
            for eval_env_name in params["evaluate_on"]:
                eval_env = envs[eval_env_name]["eval_env"]
                cumulative_reward = util.eval_env(model, eval_env, eval_eps)
                cumulative_rewards[name][eval_env_name].append(cumulative_reward)

                with open(f"{LOG_DIR}/sim_cumulative_rewards.yaml", "w") as file:
                    yaml.dump(cumulative_rewards, file)

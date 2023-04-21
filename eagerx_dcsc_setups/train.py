# eagerx imports
import eagerx
from eagerx.wrappers import Flatten
from eagerx_dcsc_setups.pendulum.envs import PendulumEnv
from eagerx.engines.openai_gym.engine import GymEngine
from eagerx_ode.engine import OdeEngine

# Common imports
import glob
import os
import yaml
from pathlib import Path
from typing import Dict
import gym.wrappers as w


# Stable baselines imports
import stable_baselines3 as sb3
from stable_baselines3.common.utils import set_random_seed


def create_env(
    cfg: Dict,
    repetition: int,
    graph: eagerx.Graph,
    engine: eagerx.specs.EngineSpec,
    backend: eagerx.specs.BackendSpec,
    pendulum: eagerx.specs.ObjectSpec,
):
    t_max = cfg["train"]["t_max"]

    param_dict = {}
    for parameter in ["mass_low", "mass_high", "length_low", "length_high", "rate"]:
        if parameter in cfg["settings"][setting]:
            param_dict[parameter] = cfg["settings"][setting][parameter]

    if "delay_low" in cfg["settings"][setting] and "delay_high" in cfg["settings"][setting]:
        rate = cfg["settings"][setting]["rate"]
        param_dict["delay_low"] = cfg["settings"][setting]["delay_low"] / rate
        param_dict["delay_high"] = cfg["settings"][setting]["delay_high"] / rate

    seed = repetition * 5
    set_random_seed(seed)

    pendulum.config.seed = seed

    env = PendulumEnv(
        name=f"ArmEnv{seed}",
        graph=graph,
        engine=engine,
        backend=backend,
        t_max=t_max,
        seed=seed,
        **param_dict,
    )
    return w.rescale_action.RescaleAction(Flatten(env), min_action=-1.0, max_action=1.0)


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Get root path
    root = Path(__file__).parent.parent

    # Load config
    cfg_path = root / "cfg" / "train.yaml"
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)

    # Get parameters
    repetitions = cfg["train"]["repetitions"]
    t_max = cfg["train"]["t_max"]
    disp = cfg["train"]["disp"]
    learning_rate = cfg["train"]["learning_rate"]
    device = cfg["train"]["device"]

    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()

    for repetition in range(repetitions):
        for setting in cfg["settings"].keys():
            rate = cfg["settings"][setting]["rate"]
            actuator_rate = cfg["settings"][setting]["actuator_rate"]
            total_timesteps = cfg["settings"][setting]["total_timesteps"]
            engine = cfg["settings"][setting]["engine"]
            engine_rate = max(rate, actuator_rate)
            if engine == "ode":
                engine = OdeEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)
            elif engine == "gym":
                engine = GymEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)

            seed = repetition
            log_dir = root / "exps" / "train" / "runs" / f"{setting}_{repetition}"

            graph_file = root / "exps" / "train" / "graphs" / f"graph_{setting}.yaml"
            graph = eagerx.Graph.load(str(graph_file))

            pendulum = graph.get_spec("pendulum")
            if disp:
                graph.add_component(pendulum.sensors.image)
                graph.render(source=pendulum.sensors.image, rate=rate)

            # Check if log dir exists
            if os.path.exists(log_dir) and len(glob.glob(str(log_dir) + "/rl_model_*.zip")) > 0:
                # Get last model
                checkpoints = glob.glob(str(log_dir) + "/rl_model_*.zip")
                checkpoints.sort()
                LOAD_DIR = checkpoints[-1].split(".zip")[0]
                step = int(LOAD_DIR.split("_")[-2])
                if step >= total_timesteps:
                    print("Model already trained")
                    continue

            train_env = create_env(cfg, repetition, graph, engine, backend, pendulum=pendulum)
            if disp:
                train_env.render("human")
            train_env.reset()
            model = sb3.SAC(
                "MlpPolicy", train_env, verbose=1, seed=seed, learning_rate=7e-4, device=device, tensorboard_log=str(log_dir)
            )
            model.learn(
                total_timesteps=total_timesteps,
                tb_log_name="logs",
                reset_num_timesteps=False,
            )
            model.save(str(log_dir) + f"/rl_model_{total_timesteps}_steps")
            model.save_replay_buffer(str(log_dir) + f"/rl_model_{total_timesteps}_steps_replay_buffer")
            train_env.shutdown()

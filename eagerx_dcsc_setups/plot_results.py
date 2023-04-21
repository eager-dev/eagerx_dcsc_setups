# eagerx imports
import eagerx
from eagerx.wrappers import Flatten
from eagerx_dcsc_setups.pendulum.envs import PendulumEnv
from eagerx.engines.openai_gym.engine import GymEngine
from eagerx_reality.engine import RealEngine
from eagerx_ode.engine import OdeEngine
from eagerx_dcsc_setups.pendulum.nodes import ResetAngle

# Common imports
import os
import yaml
from pathlib import Path
from typing import Dict
import gym.wrappers as w
import pickle
from tqdm import tqdm
import numpy as np

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
    t_max = cfg["eval"]["t_max"]

    param_dict = {}
    for parameter in ["mass_low", "mass_high", "length_low", "length_high", "rate"]:
        if parameter in cfg["settings"][setting]:
            param_dict[parameter] = cfg["settings"][setting][parameter]

    if "delay" in cfg["settings"][setting]:
        rate = cfg["settings"][setting]["rate"]
        param_dict["delay_low"] = cfg["settings"][setting]["delay"] / rate
        param_dict["delay_high"] = cfg["settings"][setting]["delay"] / rate

    seed = 10**5 - repetition * 5
    set_random_seed(seed)

    pendulum.config.seed = seed

    env = PendulumEnv(
        name=f"ArmEnv{seed}",
        graph=graph,
        engine=engine,
        backend=backend,
        t_max=t_max,
        seed=seed,
        evaluate=True,
        **param_dict,
    )
    return w.rescale_action.RescaleAction(Flatten(env), min_action=-1.0, max_action=1.0)


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Get root path
    root = Path(__file__).parent.parent

    # Load config
    cfg_path = root / "cfg" / "eval.yaml"
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)

    # Get parameters
    repetitions = cfg["eval"]["repetitions"]
    t_max = cfg["eval"]["t_max"]
    disp = cfg["eval"]["disp"]
    episodes = cfg["eval"]["episodes"]
    device = cfg["eval"]["device"]
    sim = cfg["eval"]["sim"]

    for repetition in range(repetitions):
        for setting in cfg["settings"].keys():
            rate = cfg["settings"][setting]["rate"]
            actuator_rate = cfg["settings"][setting]["actuator_rate"]
            total_timesteps = cfg["settings"][setting]["total_timesteps"]
            engine = cfg["settings"][setting]["engine"]
            engine_rate = max(rate, actuator_rate)
            if sim:
                if engine == "ode":
                    engine = OdeEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)
                elif engine == "gym":
                    engine = GymEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)
                mode = "sim"

                from eagerx.backends.single_process import SingleProcess
                backend = SingleProcess.make()
            else:
                engine = RealEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)
                mode = "real"

                from eagerx.backends.ros1 import Ros1
                backend = Ros1.make()
            seed = repetition

            train_log_dir = root / "exps" / "train" / "runs" / f"{setting}_{repetition}"
            LOAD_DIR = str(train_log_dir) + f"/rl_model_{total_timesteps}_steps.zip"
            eval_log_dir = root / "exps" / "eval" / "runs" / f"{setting}_{repetition}"
            eval_file = eval_log_dir / "eval.yaml"

            # Check if evaluation already done
            if os.path.exists(eval_file):
                eval_results = yaml.safe_load(open(str(eval_file), "r"))
                if eval_results is not None and f"{mode}" in eval_results.keys():
                    print(f"{mode} evaluation already done for {setting} {repetition}")
                    continue
            else:
                # Create evaluation directory
                eval_log_dir.mkdir(parents=True, exist_ok=True)
                # Create evaluation file
                eval_file.touch()

            graph_file = root / "exps" / "train" / "graphs" / f"graph_{setting}.yaml"
            graph = eagerx.Graph.load(str(graph_file))

            pendulum = graph.get_spec("pendulum")
            if disp:
                pendulum.config.sensors.append("image")
                graph.render(source=pendulum.sensors.image, rate=rate)

            graph.disconnect(action="voltage", target=pendulum.actuators.u)

            reset = ResetAngle.make("reset_angle", rate, u_range=[-2, 2], process=eagerx.NEW_PROCESS)
            graph.add(reset)

            graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
            graph.connect(action="voltage", target=reset.feedthroughs.u)
            graph.connect(source=reset.outputs.u, target=pendulum.actuators.u)
            graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)

            # Check if log dir exists
            if os.path.exists(LOAD_DIR):
                eval_env = create_env(cfg, repetition, graph, engine, backend, pendulum)
                print("Loading model from: ", LOAD_DIR)
                model = sb3.SAC.load(LOAD_DIR, env=eval_env, device=device)
            else:
                print(f"Model not found at {LOAD_DIR}.")
                continue

            if disp:
                eval_env.render("human")
            print(f"Starting evaluation for {setting} {repetition}")
            eval_results = []
            obs_dict = {}
            action_dict = {}
            for i in tqdm(range(episodes)):
                obs_dict[i] = []
                action_dict[i] = []
                obs = eval_env.reset()
                obs_dict[i].append(obs)
                done = False
                episodic_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    action_dict[i].append(action)
                    obs, reward, done, info = eval_env.step(action)
                    obs_dict[i].append(obs)
                    episodic_reward += reward
                eval_results.append(episodic_reward)
            eval_results = np.array(eval_results)
            mean = np.mean(eval_results)
            std = np.std(eval_results)
            print(f"Mean: {mean}, Std: {std}")
            # Save results
            eval_dict = yaml.safe_load(open(str(eval_file), "r"))
            if eval_dict is None:
                eval_dict = {}
            eval_dict[mode] = {"mean": float(mean), "std": float(std), "results": eval_results.tolist()}
            with open(str(eval_file), "w") as f:
                yaml.dump(eval_dict, f)
            # Save observations and actions
            with open(str(eval_log_dir / f"{mode}_obs.pkl"), "wb") as f:
                pickle.dump(obs_dict, f)
            with open(str(eval_log_dir / f"{mode}_action.pkl"), "wb") as f:
                pickle.dump(action_dict, f)
            eval_env.shutdown()

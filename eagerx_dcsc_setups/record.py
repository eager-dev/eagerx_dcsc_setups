# eagerx imports
import eagerx
from eagerx.wrappers import Flatten
from eagerx_dcsc_setups.pendulum.envs import PendulumEnv
from eagerx.engines.openai_gym.engine import GymEngine
from eagerx_reality.engine import RealEngine
from eagerx_ode.engine import OdeEngine
from eagerx_dcsc_setups.pendulum.nodes import ResetAngle

# Common imports
import numpy as np
import os
import yaml
from pathlib import Path
from typing import Dict
import gym.wrappers as w
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

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
        name=f"ArmEnv_{setting}_{seed}",
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
    device = cfg["eval"]["device"]
    sim = cfg["eval"]["sim"]

    # Record parameters
    episodes = cfg["record"]["episodes"]
    video_width = cfg["record"]["video_width"]
    video_height = cfg["record"]["video_height"]
    overwrite = cfg["record"]["overwrite"]

    for repetition in range(repetitions):
        for setting in cfg["settings"].keys():
            rate = cfg["settings"][setting]["rate"]
            actuator_rate = cfg["settings"][setting]["actuator_rate"]
            total_timesteps = cfg["settings"][setting]["total_timesteps"]
            engine = cfg["settings"][setting]["engine"]
            engine_rate = max(rate, actuator_rate)
            if sim:
                if engine == "ode":
                    encoding = "bgr"
                    engine = OdeEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)
                elif engine == "gym":
                    encoding = "rgb"
                    engine = GymEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)
                else:
                    raise ValueError(f"Engine {engine} not supported.")
                mode = "sim"

                from eagerx.backends.single_process import SingleProcess
                backend = SingleProcess.make()
            else:
                encoding = "bgr"
                engine = RealEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)
                mode = "real"

                from eagerx.backends.ros1 import Ros1
                backend = Ros1.make()
            seed = repetition

            train_log_dir = root / "exps" / "train" / "runs" / f"{setting}_{repetition}"
            LOAD_DIR = str(train_log_dir) + f"/rl_model_{total_timesteps}_steps.zip"
            eval_log_dir = root / "exps" / "eval" / "runs" / f"{setting}_{repetition}"
            record_file = eval_log_dir / f"{mode}_recording.mp4"

            # Check if recording already exists
            if os.path.exists(record_file) and not overwrite:
                print(f"Recording already exists at for {mode}, {setting}, {repetition}.")
                continue

            graph_file = root / "exps" / "train" / "graphs" / f"graph_{setting}.yaml"
            graph = eagerx.Graph.load(str(graph_file))

            pendulum = graph.get_spec("pendulum")
            pendulum.config.sensors.append("image")
            pendulum.config.render_shape = [video_height, video_width]
            pendulum.sensors.image.space.update({"shape": [video_height, video_width, 3]})
            graph.render(source=pendulum.sensors.image, rate=rate, encoding=encoding)

            graph.disconnect(action="voltage", target=pendulum.actuators.u)

            gains = np.array([2.0, 0.2, 1.0]) * 30 / actuator_rate
            reset = ResetAngle.make("reset_angle", rate, u_range=[-2, 2], gains=gains, process=eagerx.NEW_PROCESS)
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

            print(f"Starting recording for {setting} {repetition}")
            video_buffer = []
            # TODO: Fix extra reset
            # obs = eval_env.reset()
            for i in tqdm(range(episodes)):
                obs = eval_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    video_buffer.append(eval_env.render(mode="rgb_array"))
            clip = ImageSequenceClip(video_buffer, fps=rate)
            clip.write_videofile(str(record_file), fps=rate)
            eval_env.shutdown()

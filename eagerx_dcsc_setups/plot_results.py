# eagerx imports
import eagerx

# Common imports
import os
import yaml
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np


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

    if sim:
        train_cfg_path = root / "cfg" / "train.yaml"
        with open(str(cfg_path), "r") as f:
            train_cfg = yaml.safe_load(f)
        cfg["settings"] = train_cfg["settings"]


    for repetition in range(repetitions):
        for setting in cfg["settings"].keys():
            engine_rate = max(rate, actuator_rate)

            train_log_dir = root / "exps" / "train" / "runs" / f"{setting}_{repetition}"
            eval_log_dir = root / "exps" / "eval" / "runs" / f"{setting}_{repetition}"
            eval_file = eval_log_dir / "eval.yaml"

            # Check if evaluation already done
            if os.path.exists(eval_file):
                eval_results = yaml.safe_load(open(str(eval_file), "r"))
                if eval_results is not None and "f"{mode}"" in eval_results.keys():
                    print(f"{mode} evaluation already done for {setting} {repetition}")
                    continue
            else:
                # Evaluation not done yet, skip
                print(f"Skipping {setting} {repetition}, evaluation not done yet.")


            # Check if log dir exists
            if os.path.exists(LOAD_DIR):
                print("Loading model from: ", LOAD_DIR)
            else:
                print(f"Model not found at {LOAD_DIR}.")
                continue
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

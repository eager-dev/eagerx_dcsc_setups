# ROS packages required
import eagerx
from experiments.sim import simulate
from experiments.real import evaluate_real

eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

# Other
import numpy as np


if __name__ == "__main__":
    mode = "real"
    load_log = "2022-05-08-1101"

    # Define constants
    sensor_rate = 30.
    actuator_rate = 90.
    image_rate = sensor_rate / 2
    engine_rate = max([sensor_rate, actuator_rate, image_rate])
    delay = 0.025
    seed = 27
    np.random.seed(seed)

    length_train_eps = 90
    length_eval_eps = 270
    train_eps = 100
    eval_eps = 10
    repetitions = 1

    envs = {
        "bl": {"dr": False, "emmd": False, "fa": 0, "evaluate_on": ["bl"]},
        "bl_fa1": {"dr": False, "emmd": False, "fa": 1, "evaluate_on": ["bl_fa1", "dr_fa1"]},
        "dr_fa1": {"dr": True, "emmd": False, "fa": 1, "evaluate_on": ["dr_fa1", "dr_emmd_fa1"]},
        "dr_emmd_fa1": {"dr": True, "emmd": True, "fa": 1, "evaluate_on": ["dr_emmd_fa1"]},
        "dr_emmd_fa2": {"dr": True, "emmd": True, "fa": 2, "evaluate_on": ["dr_emmd_fa2"]},
    }

    if mode == "sim":
        simulate(image_rate, sensor_rate=sensor_rate, actuator_rate=actuator_rate, engine_rate=engine_rate, delay=delay,
                 seed=seed, length_train_eps=length_train_eps, length_eval_eps=length_eval_eps, train_eps=train_eps,
                 eval_eps=eval_eps, repetitions=repetitions, envs=envs)
    elif mode == "real":
        evaluate_real(sensor_rate=sensor_rate, engine_rate=engine_rate, delay=delay, seed=seed, length_eval_eps=length_eval_eps,
                      eval_eps=eval_eps, repetitions=repetitions, envs=envs, log_name=load_log)

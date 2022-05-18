# ROS packages required
import eagerx
from experiments.sim import simulate
from experiments.real import evaluate_real

eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

# Other
import numpy as np


if __name__ == "__main__":
    mode = "real"
    load_log = "2022-05-09-0659"

    # Define constants
    sensor_rate = 30.0
    actuator_rate = 90.0
    image_rate = sensor_rate / 2
    bridge_rate = max([sensor_rate, actuator_rate, image_rate])
    delay = 0.0
    seed = 27
    np.random.seed(seed)

    length_train_eps = 90
    length_eval_eps = 270
    train_eps = 250
    eval_eps = 10
    repetitions = 1

    # envs = {
    #     "bl_fa1": {"dr": False, "ed": False, "fa": 1, "evaluate_on": ["bl_fa1", "dr_ed_fa1"]},
    #     "dr_ed_fa1": {"dr": True, "ed": True, "fa": 1, "evaluate_on": ["dr_ed_fa1"]},
    #     "dr_ed_fa2": {"dr": True, "ed": True, "fa": 2, "evaluate_on": ["dr_ed_fa2"]},
    # }

    envs = {
        "bl_fa1": {"dr": False, "ed": False, "fa": 1, "evaluate_on": ["bl_fa1"]},
    }

    if mode == "sim":
        simulate(
            image_rate,
            sensor_rate=sensor_rate,
            actuator_rate=actuator_rate,
            bridge_rate=bridge_rate,
            delay=delay,
            seed=seed,
            length_train_eps=length_train_eps,
            length_eval_eps=length_eval_eps,
            train_eps=train_eps,
            eval_eps=eval_eps,
            repetitions=repetitions,
            envs=envs,
        )
    elif mode == "real":
        evaluate_real(
            sensor_rate=sensor_rate,
            bridge_rate=bridge_rate,
            delay=delay,
            seed=seed,
            length_eval_eps=length_eval_eps,
            eval_eps=eval_eps,
            repetitions=repetitions,
            envs=envs,
            log_name=load_log,
        )

import eagerx
from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_ode.engine import OdeEngine
from eagerx.wrappers import Flatten
from eagerx.backends.single_process import SingleProcess
from eagerx_dcsc_setups.pendulum.envs import PendulumEnv

import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from gym import wrappers as w
import numpy as np


if __name__ == "__main__":
    rate = 30
    image_rate = 15
    seed = 1
    np.random.seed(seed)

    sensor_rate = rate
    actuator_rate = 3 * rate
    engine_rate = max(sensor_rate, actuator_rate)

    # Create pendulum object
    pendulum = Pendulum.make(
        "pendulum",
        actuators=["u"],
        sensors=["x", "image", "action_applied"],
        states=["model_state"],
        actuator_rate=actuator_rate,
        sensor_rate=sensor_rate,
    )

    # Create graph
    graph = eagerx.Graph.create()
    graph.add(pendulum)
    graph.connect(action="voltage", target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, observation="angle_data", window=2)
    graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", skip=True)
    graph.render(source=pendulum.sensors.image, rate=image_rate)

    ode_engine = OdeEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT)
    backend = SingleProcess.make()


    # Create envs
    train_env = PendulumEnv(
        name="TrainEnv",
        rate=rate,
        graph=graph,
        engine=ode_engine,
        backend=backend,
        delay_low=1/(rate),
        delay_high=1/(rate),
        evaluate=False,
    )
    train_env = w.rescale_action.RescaleAction(Flatten(train_env), min_action=-1.0, max_action=1.0)

    # Check env validity
    check_env(train_env)

    # Set seed
    train_env.seed(seed)

    model = sb3.SAC("MlpPolicy", train_env, verbose=1, seed=seed, learning_rate=7e-4)

    # First train in simulation
    train_env.render("human")
    model.learn(total_timesteps=10_000)
    # model.save("pendulum_delay")
    train_env.close()

    eval_env = PendulumEnv(
        name="EvalEnv",
        rate=rate,
        graph=graph,
        engine=ode_engine,
        backend=backend,
        evaluate=True,
        delay_low=1 / rate,
        delay_high=1 / rate,
    )
    eval_env = w.rescale_action.RescaleAction(Flatten(eval_env), min_action=-1.0, max_action=1.0)
    eval_env.render("human")

    # model = sb3.SAC.load("pendulum_delay", env=eval_env)


    # eval_env.reset()
    for _ in range(10):
        done = False
        obs = eval_env.reset()
        obs = eval_env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

    eval_env.shutdown()


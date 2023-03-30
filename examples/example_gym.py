import eagerx
from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_dcsc_setups.pendulum.envs import PendulumEnv
from eagerx.engines.openai_gym.engine import GymEngine
from eagerx_ode.engine import OdeEngine
from eagerx.wrappers import Flatten
from eagerx.backends.single_process import SingleProcess

import stable_baselines3 as sb3
import numpy as np
import gym.wrappers as w

if __name__ == "__main__":
    rate = 20
    image_rate = 20
    seed = 1
    np.random.seed(seed)

    sensor_rate = rate
    actuator_rate = 5*rate
    engine_rate = max(sensor_rate, actuator_rate)

    # Create pendulum object
    pendulum = Pendulum.make(
        "pendulum",
        actuators=["u"],
        sensors=["x", "image", "action_applied"],
        states=["model_state", "max_speed", "length", "mass", "dt"],
        actuator_rate=actuator_rate,
        sensor_rate=sensor_rate,
    )

    # Create graph
    graph = eagerx.Graph.create()
    graph.add(pendulum)
    graph.connect(action="voltage", target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, observation="angle_data")
    graph.render(source=pendulum.sensors.image, rate=image_rate)

    gym_engine = GymEngine.make(rate=sensor_rate, process=eagerx.ENVIRONMENT)
    ode_engine = OdeEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT, real_time_factor=1)

    backend = SingleProcess.make()

    # Create envs
    train_env = PendulumEnv(
        name="TrainEnv",
        rate=rate,
        graph=graph,
        engine=gym_engine,
        backend=backend,
        # delay_low=0,
        # delay_high=1/(rate),
        mass_low=0.03,
        mass_high=0.05,
        length_low=0.12,
        length_high=0.12,
        evaluate=False,
        dt=1.0 / engine_rate,
    )
    train_env = w.rescale_action.RescaleAction(Flatten(train_env), min_action=-1.0, max_action=1.0)

    # Set seed
    train_env.render("human")

    # Extra reset because of render bug
    train_env.reset()
    model = sb3.SAC("MlpPolicy", train_env, verbose=1, seed=seed, learning_rate=7e-4)
    model.learn(total_timesteps=15_000)
    train_env.close()

    eval_env = PendulumEnv(
        name="EvalEnv",
        rate=rate,
        graph=graph,
        engine=ode_engine,
        backend=backend,
        evaluate=True,
        delay_low=0,
        delay_high=1/rate,
    )
    eval_env = w.rescale_action.RescaleAction(Flatten(eval_env), min_action=-1.0, max_action=1.0)
    eval_env.render("human")

    # eval_env.reset()
    for _ in range(10):
        done = False
        obs = eval_env.reset()
        obs = eval_env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

    eval_env.shutdown()

import eagerx
from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_dcsc_setups.pendulum.envs import PendulumEnv
from eagerx.engines.openai_gym.engine import GymEngine
from eagerx_ode.engine import OdeEngine
from eagerx_reality.engine import RealEngine
from eagerx.wrappers import Flatten
from eagerx.backends.single_process import SingleProcess
from eagerx.backends.ros1 import Ros1
from eagerx_dcsc_setups.pendulum.nodes import ResetAngle

import stable_baselines3 as sb3
import numpy as np
import gym.wrappers as w

if __name__ == "__main__":
    rate = 20
    image_rate = 20
    seed = 1
    np.random.seed(seed)

    sensor_rate = rate
    actuator_rate = rate
    engine_rate = max(sensor_rate, actuator_rate)

    # Create pendulum object
    pendulum = Pendulum.make(
        "pendulum",
        actuators=["u"],
        sensors=["x", "image"],
        states=["model_state", "max_speed", "length", "mass"],
        actuator_rate=actuator_rate,
        sensor_rate=sensor_rate,
        camera_index=0,
    )

    # Create graph
    graph = eagerx.Graph.create()
    graph.add(pendulum)
    graph.connect(action="voltage", target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, observation="angle_data")
    graph.render(source=pendulum.sensors.image, rate=image_rate)

    gym_engine = GymEngine.make(rate=sensor_rate, process=eagerx.ENVIRONMENT)
    real_engine = RealEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT, sync=True)

    backend = Ros1.make()

    # Create envs
    train_env = PendulumEnv(
        name="TrainEnv",
        rate=rate,
        graph=graph,
        engine=gym_engine,
        backend=backend,
        mass_low=0.03,
        mass_high=0.05,
        length_low=0.1,
        length_high=0.14,
        evaluate=False,
    )
    train_env = w.rescale_action.RescaleAction(Flatten(train_env), min_action=-1.0, max_action=1.0)

    # Set seed
    train_env.render("human")

    # Extra reset because of render bug
    train_env.reset()
    model = sb3.SAC("MlpPolicy", train_env, verbose=1, seed=seed, learning_rate=7e-4)
    model.learn(total_timesteps=5_000)
    train_env.close()

    pendulum = graph.get_spec("pendulum")

    graph.disconnect(action="voltage", target=pendulum.actuators.u)

    reset = ResetAngle.make("reset_angle", rate, u_range=[-2, 2], process=eagerx.NEW_PROCESS)
    graph.add(reset)

    graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
    graph.connect(action="voltage", target=reset.feedthroughs.u)
    graph.connect(source=reset.outputs.u, target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)

    eval_env = PendulumEnv(
        name="EvalEnv",
        rate=rate,
        graph=graph,
        engine=real_engine,
        backend=backend,
        evaluate=True,
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

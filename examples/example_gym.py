import eagerx
from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_dcsc_setups.pendulum.envs import PendulumEnv
from eagerx.engines.openai_gym.engine import GymEngine
from eagerx_reality.engine import RealEngine
from eagerx.wrappers import Flatten
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
    actuator_rate = 20
    engine_rate = max(sensor_rate, actuator_rate)

    # Create pendulum object
    pendulum = Pendulum.make(
        "pendulum",
        actuators=["u"],
        sensors=["x", "image", "action_applied"],
        states=["model_state", "max_speed", "length", "mass"],
        actuator_rate=actuator_rate,
        sensor_rate=sensor_rate,
        camera_index=0,
    )
    pendulum.states.dt.space.low = 1 / engine_rate
    pendulum.states.dt.space.high = 1 / engine_rate
    mass = 0.07153543216026934
    length = 0.10380430347908418
    # pendulum.states.mass.space.low = 0.05697082840218357
    # pendulum.states.mass.space.high = 0.07014605309248077
    # pendulum.states.length.space.low = 0.08234486282363933
    # pendulum.states.length.space.high = 0.12032068386221356
    pendulum.states.mass.space.low = mass * 0.95
    pendulum.states.mass.space.high = mass * 1.05
    pendulum.states.length.space.low = length * 0.95
    pendulum.states.length.space.high = length * 1.05

    # Create graph
    graph = eagerx.Graph.create()
    graph.add(pendulum)
    graph.connect(action="voltage", target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", skip=True)
    graph.connect(source=pendulum.sensors.x, observation="angle_data", window=2)
    graph.render(source=pendulum.sensors.image, rate=image_rate)

    gym_engine = GymEngine.make(rate=sensor_rate, process=eagerx.ENVIRONMENT)
    real_engine = RealEngine.make(rate=engine_rate, process=eagerx.ENVIRONMENT, sync=True)

    from eagerx.backends.ros1 import Ros1

    backend = Ros1.make()

    # Create envs
    train_env = PendulumEnv(
        name="TrainEnv",
        rate=rate,
        graph=graph,
        engine=gym_engine,
        backend=backend,
        evaluate=False,
        delay_low=0.03,
        delay_high=0.04,
    )
    train_env = w.rescale_action.RescaleAction(Flatten(train_env), min_action=-1.0, max_action=1.0)

    # Set seed
    train_env.render("human")

    # Extra reset because of render bug
    train_env.reset()
    model = sb3.SAC("MlpPolicy", train_env, verbose=1, seed=seed, learning_rate=7e-4)
    model.set_parameters("pendulum_dr_ds")
    model.learn(total_timesteps=5_000)
    model.save("pendulum_dr_ds")
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
        delay_low=0.035,
        delay_high=0.035,
    )
    eval_env = w.rescale_action.RescaleAction(Flatten(eval_env), min_action=-1.0, max_action=1.0)
    eval_env.render("human")

    # eval_env.reset()
    for _ in range(10000):
        done = False
        obs = eval_env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

    eval_env.shutdown()

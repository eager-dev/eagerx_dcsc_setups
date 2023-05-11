import eagerx
from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_dcsc_setups.pendulum.envs import PendulumEnv
from eagerx.engines.openai_gym.engine import GymEngine
from eagerx_reality.engine import RealEngine
from eagerx.wrappers import Flatten
from eagerx_dcsc_setups.pendulum.nodes import ResetAngle

import stable_baselines3 as sb3
from stable_baselines3.common.logger import configure

import numpy as np
import gym.wrappers as w
import optuna

if __name__ == "__main__":
    rate = 20
    image_rate = 20
    seed = 1
    np.random.seed(seed)

    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, [])

    sensor_rate = rate
    actuator_rate = 20
    engine_rate = max(sensor_rate, actuator_rate)

    # Create pendulum object
    pendulum = Pendulum.make(
        "pendulum",
        actuators=["u"],
        sensors=["x", "action_applied"],
        states=["model_state", "max_speed", "length", "mass"],
        actuator_rate=actuator_rate,
        sensor_rate=sensor_rate,
        camera_index=0,
    )

    # Create graph
    graph = eagerx.Graph.create()
    graph.add(pendulum)

    graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", skip=True)
    graph.connect(source=pendulum.sensors.x, observation="angle_data", window=2)

    gains = [0.5, 0.1, 0.1]
    reset = ResetAngle.make("reset_angle", rate, gains=gains, u_range=[-2, 2], process=eagerx.NEW_PROCESS, timeout=1.0)
    graph.add(reset)

    graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
    graph.connect(action="voltage", target=reset.feedthroughs.u)
    graph.connect(source=reset.outputs.u, target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)

    from eagerx.backends.ros1 import Ros1
    backend = Ros1.make()

    gym_engine = GymEngine.make(rate=sensor_rate, process=eagerx.ENVIRONMENT)
    real_engine = RealEngine.make(rate=engine_rate, sync=True)

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

    def objective(trial):
        mass = trial.suggest_float('mass_low', 0.05, 0.08)
        length = trial.suggest_float('length_low', 0.07, 0.13)
        if trial.number > 0:
            print(f"Trial {trial.number} with mass {mass} and length {length}, best value: {study.best_value}, best params: {study.best_params}")

        # Create pendulum object
        pendulum = Pendulum.make(
            "pendulum",
            actuators=["u"],
            sensors=["x", "action_applied"],
            states=["model_state", "max_speed", "length", "mass"],
            actuator_rate=actuator_rate,
            sensor_rate=sensor_rate,
            camera_index=0,
        )
        pendulum.states.dt.space.low = 1 / engine_rate
        pendulum.states.dt.space.high = 1 / engine_rate
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

        # Create envs
        train_env = PendulumEnv(
            name=f"TrainEnv_{trial.number}",
            rate=rate,
            graph=graph,
            engine=gym_engine,
            backend=backend,
            evaluate=False,
            delay_low=0.03,
            delay_high=0.04,
        )
        train_env = w.rescale_action.RescaleAction(Flatten(train_env), min_action=-1.0, max_action=1.0)

        model = sb3.SAC("MlpPolicy", train_env, verbose=0, learning_rate=7e-4)
        model.set_logger(new_logger)
        model.learn(total_timesteps=50_000)
        episodic_rewards = []
        for _ in range(3):
            done = False
            rewards = 0
            obs = eval_env.reset()
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                rewards += reward
            episodic_rewards.append(rewards)
        # intermediate_value = np.infty
        #
        # total_steps = 0
        # for step, total_timesteps in enumerate([12_500, 2_500, 2_500, 2_500, 2_500, 2_500, 2_500, 2_500, 2_500, 2_500, 2_500, 2_500]):
        #     model.learn(total_timesteps=total_timesteps)
        #     episodic_rewards = []
        #     for _ in range(3):
        #         done = False
        #         rewards = 0
        #         obs = eval_env.reset()
        #         while not done:
        #             action, _ = model.predict(obs, deterministic=True)
        #             obs, reward, done, info = eval_env.step(action)
        #             rewards += reward
        #         episodic_rewards.append(rewards)
        #
        #     intermediate_value = -np.mean(episodic_rewards)
        #     trial.report(intermediate_value, step=step)
        #     total_steps += total_timesteps
        #     print(f"Trial {trial.number} at step {total_steps}/40000 with value {intermediate_value}, best value {study.best_value}, best params {study.best_params}")
        #     if trial.should_prune():
        #         raise optuna.TrialPruned()
        train_env.shutdown()

        value = -np.mean(episodic_rewards)
        if trial.number > 0 and value < study.best_value:
            model.save(f"model_real_{mass}_{length}_{value}")

        model.logger.close()
        del model
        return value


    study_name = 'real-study-8'  # Unique identifier of the study.
    study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db')
    # study = optuna.load_study(study_name=study_name, storage="sqlite:///example.db")
    study.optimize(objective, n_trials=100, n_jobs=1)

    study.best_params  # E.g. {'x': 2.002108042}

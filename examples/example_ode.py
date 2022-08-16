import eagerx
from eagerx.core.specs import EngineSpec
from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_ode.engine import OdeEngine
from eagerx.wrappers import Flatten
from eagerx.backends.single_process import SingleProcess

import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction
import numpy as np
from typing import Dict



class PendulumEnv(eagerx.BaseEnv):
    def __init__(self, name: str, rate: float, graph: eagerx.Graph, engine: EngineSpec):
        """Initializes an environment with EAGERx dynamics.

        :param name: The name of the environment. Everything related to this environment
                     (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
        :param rate: The rate (Hz) at which the environment will run.
        :param graph: The graph consisting of nodes and objects that describe the environment's dynamics.
        :param engine: The physics engine that will govern the environment's dynamics.
        :param eval: If True we will create an evaluation environment, i.e. not performing domain randomization.
        """
        # Make the backend specification
        backend = SingleProcess.make()

        self.eval = eval
        # Maximum episode length
        self.episode_length = 100

        # Step counter
        self.steps = None
        super().__init__(name, rate, graph, engine, backend, force_start=True)

    def step(self, action: Dict):
        """A method that runs one timestep of the environment's dynamics.

        :params action: A dictionary of actions provided by the agent.
        :returns: A tuple (observation, reward, done, info).

                  - observation: Dictionary of observations of the current timestep.

                  - reward: amount of reward returned after previous action

                  - done: whether the episode has ended, in which case further step() calls will return undefined results

                  - info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Take step
        obs = self._step(action)
        self.steps += 1

        # Extract observations
        cos_th, sin_th, thdot = obs["angle_data"][0]
        u = action["voltage"][0]

        # Calculate reward
        # We want to penalize the angle error, angular velocity and applied voltage
        th = np.arctan2(sin_th, cos_th)
        cost = th**2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u**2

        # Determine done flag
        done = self.steps > self.episode_length

        # Set info:
        info = {"TimeLimit.truncated": done}

        return obs, -cost, done, info

    def reset(self) -> Dict:
        """Resets the environment to an initial state and returns an initial observation.

        :returns: The initial observation.
        """
        # Determine reset states
        states = self.state_space.sample()

        # Perform reset
        obs = self._reset(states)

        # Reset step counter
        self.steps = 0
        return obs


if __name__ == "__main__":
    rate = 30
    image_rate = 15
    seed = 1
    np.random.seed(seed)

    # Create pendulum object
    pendulum = Pendulum.make("pendulum", actuators=["u"], sensors=["x", "image"], states=["model_state"])

    # Create graph
    graph = eagerx.Graph.create()
    graph.add(pendulum)
    graph.connect(action="voltage", target=pendulum.actuators.u, window=1)
    graph.connect(source=pendulum.sensors.x,  observation="angle_data", window=1)
    graph.render(source=pendulum.sensors.image, rate=image_rate)

    # Open gui
    pendulum.gui(OdeEngine)
    graph.gui()

    engine = OdeEngine.make(rate=rate)
    env = PendulumEnv(name="PendulumEnv", rate=rate, graph=graph, engine=engine)
    env = RescaleAction(Flatten(env), min_action=-1.0, max_action=1.0)

    # Check env validity
    check_env(env)

    # Set seed
    env.seed(seed)

    model = sb3.SAC("MlpPolicy", env, verbose=1, seed=seed, learning_rate=7e-4)

    # First train in simulation
    env.render("human")
    model.learn(total_timesteps=6000)
    env.close()

    # Toggle render
    env.render("human")

    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

    env.shutdown()
import eagerx
from eagerx.core.specs import EngineSpec
from eagerx.core.specs import BackendSpec
import numpy as np
from typing import Dict


class PendulumEnv(eagerx.BaseEnv):
    def __init__(
        self,
        name: str,
        rate: float,
        graph: eagerx.Graph,
        engine: EngineSpec,
        backend: BackendSpec,
        evaluate: bool = False,
        seed: int = 0,
        t_max: float = 5,
        mass_low: float = 0.04,
        mass_high: float = 0.06,
        length_low: float = 0.08,
        length_high: float = 0.12,
        delay_low: float = None,
        delay_high: float = None,
        dt: float = None,
    ):
        """Initializes an environment with EAGERx dynamics.

        :param name: The name of the environment. Everything related to this environment
                     (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
        :param rate: The rate (Hz) at which the environment will run.
        :param graph: The graph consisting of nodes and objects that describe the environment's dynamics.
        :param engine: The physics engine that will govern the environment's dynamics.
        :param backend: The backend that will run the environment.
        :param evaluate: If True we will create an evaluation environment, i.e. not performing domain randomization.
        :param seed: The seed used to initialize the environment.
        :param t_max: The maximum episode length (seconds).
        :param mass_low: The lower bound of the mass domain randomization.
        :param mass_high: The upper bound of the mass domain randomization.
        :param length_low: The lower bound of the length domain randomization.
        :param length_high: The upper bound of the length domain randomization.
        :param delay_low: The lower bound of the delay domain randomization.
        :param delay_high: The upper bound of the delay domain randomization.
        :param dt: The time step of the environment. If None, it will be not be set.
        """
        # Set evaluation flag
        self.evaluate = evaluate

        # Maximum episode length
        self.episode_length = t_max * rate

        # Set domain randomization parameters
        self.mass_low = mass_low
        self.mass_high = mass_high
        self.length_low = length_low
        self.length_high = length_high
        self.delay_low = delay_low
        self.delay_high = delay_high

        # Set time step
        self.dt = dt

        # Step counter
        self.steps = None
        super().__init__(name, rate, graph, engine, backend, force_start=True)

        self._state_space = self.state_space
        for key in self._state_space.spaces.keys():
            self._state_space.spaces[key]._space.seed(seed)
            seed += 1

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
        done = self.episode_length < self.steps

        # Set info:
        info = {"TimeLimit.truncated": done}

        return obs, -cost, done, info

    def reset(self) -> Dict:
        """Resets the environment to an initial state and returns an initial observation.

        :returns: The initial observation.
        """
        # Determine reset states
        states = self._state_space.sample()

        if self.evaluate:
            # During evaluation on the real system we cannot set the state to an arbitrary position and velocity
            offset = np.random.rand() - 0.5
            theta = np.pi - offset if offset > 0 else -np.pi - offset
            states["pendulum/model_state"] = np.array([theta, 0], dtype="float32")

        # Sample mass (kg)
        if "pendulum/mass" in states:
            states["pendulum/mass"] = np.random.uniform(low=self.mass_low, high=self.mass_high, size=()).astype(
                "float32"
            )  # Sample mass (kg)
        if "pendulum/length" in states:
            states["pendulum/length"] = np.random.uniform(low=self.length_low, high=self.length_high, size=()).astype(
                "float32"
            )  # Sample length (m)

        # Sample delay
        if self.delay_low is None or self.delay_high is None:
            if "pendulum/u/delay" in states:
                states["pendulum/u/delay"] = None
        elif self.delay_low == self.delay_high:
            states["pendulum/u/delay"] = np.array(self.delay_low, dtype="float32")
        else:
            actuator_delay = np.random.random(()) * (self.delay_high - self.delay_low) + self.delay_low
            states["pendulum/u/delay"] = np.array(actuator_delay, dtype="float32")

        # Set time step
        if self.dt is not None and "pendulum/dt" in states:
            states["pendulum/dt"] = np.array(self.dt, dtype="float32")

        # Perform reset
        obs = self._reset(states)

        # Reset step counter
        self.steps = 0
        for key, value in obs.items():
            # check if observation is NaN
            if np.isnan(value).any():
                obs[key] = np.zeros_like(value)
        return obs

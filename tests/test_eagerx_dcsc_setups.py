# ROS packages required
import eagerx
from eagerx.core.specs import EngineSpec
from eagerx.backends.single_process import SingleProcess

# Implementation specific
from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_ode.engine import OdeEngine

# Other
from eagerx.wrappers.flatten import Flatten
from gym.wrappers.rescale_action import RescaleAction
import numpy as np
from typing import Dict
import pytest

NP = eagerx.process.NEW_PROCESS
ENV = eagerx.process.ENVIRONMENT

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

    def reset(self, states=None) -> Dict:
        """Resets the environment to an initial state and returns an initial observation.

        :returns: The initial observation.
        """
        # Determine reset states
        if states is None:
            states = self.state_space.sample()

        # Perform reset
        obs = self._reset(states)

        # Reset step counter
        self.steps = 0
        return obs


@pytest.mark.parametrize(
    "eps, steps, sync, rtf, p",
    [(3, 3, True, 0, ENV)],
)
def test_pendulum_ode(eps, steps, sync, rtf, p):

    # Define unique name for test environment
    name = f"{eps}_{steps}_{sync}_{p}"

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
    graph.connect(source=pendulum.sensors.x, observation="angle_data", window=1)
    graph.render(source=pendulum.sensors.image, rate=image_rate)

    # Open gui
    # pendulum.gui(OdeEngine)
    # graph.gui()

    engine = OdeEngine.make(rate=rate)
    env = PendulumEnv(name=name, rate=rate, graph=graph, engine=engine)
    env = RescaleAction(Flatten(env), min_action=-1.0, max_action=1.0)

    # First reset
    env.reset()
    action = env.action_space.sample()
    for j in range(eps):
        print("\n[Episode %s]" % j)
        for i in range(steps):
            env.step(action)
        env.reset()
    print("\n[Finished]")
    env.shutdown()
    print("\n[Shutdown]")


@pytest.mark.parametrize(
    "eps, steps, sync, rtf, p",
    [(3, 30, True, 0, ENV)],
)
def test_dfun(eps, steps, sync, rtf, p):
    """
    Creates two environments, one uses a Jacobian function (Dfun) and the other not.
    Tests if the observations of the environments are close to eachother within a tolerance.

    :param eps: Number of episodes
    :param steps: Number of steps per episode
    :param sync: If True, the environment is reactive
    :param rtf: Real-time factor
    :param p: Process
    :return:
    """

    # Define unique name for test environment
    name = f"{eps}_{steps}_{sync}_{p}"
    engine_p = p
    rate = 30
    seed = 1

    # Initialize empty graphs
    graph = eagerx.Graph.create()
    graph2 = eagerx.Graph.create()

    # Create pendulum
    pendulum = Pendulum.make("pendulum")
    graph.add(pendulum)

    pendulum2 = Pendulum.make("pendulum", Dfun=None)
    graph2.add(pendulum2)

    # Connect the nodes
    graph.connect(action="voltage", target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, observation="angle_data", window=1)
    graph.render(pendulum.sensors.image, rate=10)

    graph2.connect(action="voltage", target=pendulum2.actuators.u)
    graph2.connect(source=pendulum2.sensors.x, observation="angle_data", window=1)
    graph2.render(pendulum2.sensors.image, rate=10)

    # Define engines
    engine = OdeEngine.make(
        rate=rate,
        sync=sync,
        real_time_factor=rtf,
        process=engine_p,
    )

    # Initialize Environment
    env = PendulumEnv(name=f"{name}", rate=rate, graph=graph, engine=engine)
    env = RescaleAction(Flatten(env), min_action=-1.0, max_action=1.0)
    env2 = PendulumEnv(name=f"{name}_2", rate=rate, graph=graph2, engine=engine)
    env2 = RescaleAction(Flatten(env2), min_action=-1.0, max_action=1.0)

    # First reset
    states = env.state_space.sample()
    env.reset(states=states)
    env2.reset(states=states)
    action = env.action_space.sample()
    for j in range(eps):
        print("\n[Episode %s]" % j)
        for i in range(steps):
            obs, _, _, _ = env.step(action)
            obs2, _, _, _ = env2.step(action)

            # Assert if result is the same with and without Jacobian
            assert np.allclose(obs, obs2)
        states = env.state_space.sample()
        env.reset(states=states)
        env2.reset(states=states)
    print("\n[Finished]")
    env.shutdown()
    env2.shutdown()
    print("\n[Shutdown]")

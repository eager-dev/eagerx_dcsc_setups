import eagerx
import eagerx.converters
import eagerx.nodes
import eagerx_dcsc_setups
from eagerx.wrappers import Flatten
import stable_baselines3 as sb3

import numpy as np
from typing import Dict


def make_graph(
    rate: float,
    image_rate: float,
    evaluation: bool,
):
    u_limit = 2.0

    # Create empty graph
    graph = eagerx.Graph.create()

    # Create pendulum
    pendulum = eagerx.Object.make(
        "Pendulum",
        "pendulum",
        render_shape=[480, 480],
        sensors=["x"],
        states=["model_state"],
        sensor_rate=rate,
        actuator_rate=rate,
        image_rate=image_rate,
    )
    graph.add(pendulum)

    # Create Butterworth filter
    bf = eagerx.Node.make(
        "ButterworthFilter",
        name="butterworth_filter",
        rate=rate,
        N=2,
        Wn=13,
        process=eagerx.process.NEW_PROCESS,
    )
    bf.inputs.signal.space_converter = pendulum.actuators.u.space_converter
    bf.outputs.filtered.space_converter = pendulum.actuators.u.space_converter
    graph.add(bf)

    if evaluation:
        reset = eagerx.ResetNode.make(
            "ResetAngle",
            "reset_angle",
            rate,
            u_range=[-u_limit, +u_limit],
            gains=[0.5, 0.4, 0.3],
        )
        graph.add(reset)

        graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
        graph.connect(action="voltage", target=reset.feedthroughs.u)
        graph.connect(source=reset.outputs.u, target=bf.inputs.signal)
        graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)
    else:
        graph.connect(action="voltage", target=bf.inputs.signal)
    graph.connect(source=bf.outputs.filtered, target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, observation="angle_data")
    graph.connect(source=bf.outputs.filtered, observation="action_applied", skip=True, initial_obs=[0], window=1)

    # Add rendering
    graph.add_component(pendulum.sensors.image)
    layover = eagerx.Node.make("Overlay", "overlay", rate=image_rate, process=eagerx.process.NEW_PROCESS)
    graph.add(layover)
    graph.connect(source=pendulum.sensors.x, target=layover.inputs.x)
    graph.connect(source=bf.outputs.filtered, target=layover.inputs.u)
    graph.connect(source=pendulum.sensors.image, target=layover.inputs.base_image)
    graph.render(source=layover.outputs.image, rate=image_rate, display=True, process=eagerx.process.NEW_PROCESS)

    return graph, pendulum


class PendulumEnv(eagerx.BaseEnv):
    def __init__(self, name: str, rate: float, graph: eagerx.Graph, engine: eagerx.Engine, eval=False):
        """Initializes an environment with EAGERx dynamics.

        :param name: The name of the environment. Everything related to this environment
                     (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
        :param rate: The rate (Hz) at which the environment will run.
        :param graph: The graph consisting of nodes and objects that describe the environment's dynamics.
        :param engine: The physics engine that will govern the environment's dynamics.
        :param eval: If True we will create an evaluation environment, i.e. not performing domain randomization.
        """
        self.eval = eval
        # Maximum episode length
        self.episode_length = 270 if self.eval else 100

        # Step counter
        self.steps = None
        super().__init__(name, rate, graph, engine, force_start=True)

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
    eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

    rate = 30
    image_rate = 15
    u_limit = 2
    seed = 1
    np.random.seed(seed)

    train_graph, pendulum = make_graph(rate=rate, image_rate=image_rate, evaluation=False)
    eval_graph, _ = make_graph(rate=rate, image_rate=image_rate, evaluation=True)

    ode_engine = eagerx.Engine.make("OdeEngine", rate=rate, sync=True, process=eagerx.process.NEW_PROCESS)
    real_engine = eagerx.Engine.make("RealEngine", rate=rate, sync=True, process=eagerx.process.NEW_PROCESS)

    train_env = Flatten(PendulumEnv(name="sim", rate=rate, graph=train_graph, engine=ode_engine))
    eval_env = Flatten(PendulumEnv(name="real", rate=rate, graph=eval_graph, engine=real_engine, eval=True))

    train_graph.gui()

    # Set seed
    train_env.seed(seed)
    eval_env.seed(seed)

    model = sb3.SAC("MlpPolicy", train_env, verbose=1, seed=seed, learning_rate=7e-4)

    # First train in simulation
    train_env.render("human")
    model.learn(total_timesteps=6000)
    train_env.close()

    # Toggle render
    eval_env.render("human")

    while True:
        done = False
        obs = eval_env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

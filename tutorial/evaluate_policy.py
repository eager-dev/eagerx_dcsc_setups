import eagerx
import eagerx_dcsc_setups
from eagerx.wrappers import Flatten
import stable_baselines3 as sb3

# Other
from typing import Dict
import numpy as np


class PendulumEnv(eagerx.BaseEnv):
    def __init__(self, name: str, rate: float, graph: eagerx.Graph, engine: eagerx.Engine):
        """Initializes an environment with EAGERx dynamics.

        :param name: The name of the environment. Everything related to this environment
                     (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
        :param rate: The rate (Hz) at which the environment will run.
        :param graph: The graph consisting of nodes and objects that describe the environment's dynamics.
        :param engine: The physics engine that will govern the environment's dynamics.
        :param eval: If True we will create an evaluation environment, i.e. not performing domain randomization.
        """
        # Maximum episode length
        self.episode_length = 270

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

    # user_name = "Halil"
    # file_name = "pendulum_halil.zip"
    user_name = "Jan"
    file_name = "pendulum_jan.zip"
    # user_name = "Ritesh"
    # file_name = "pendulum_ritesh.zip"

    model_path = f"/home/jelle/Downloads/{file_name}"

    # Define rate (depends on rate of ode)
    rate = 20
    image_rate = 10

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Select sensors, actuators and states of Pendulum
    sensors = ["x", "image"]
    actuators = ["u"]
    states = ["model_state"]

    # Make pendulum
    pendulum = eagerx.Object.make(
        "Pendulum", "pendulum", sensors=sensors, states=states, sensor_rate=rate, actuator_rate=rate, image_rate=image_rate
    )

    # Add pendulum to the graph
    graph.add(pendulum)

    # Connect the pendulum to an action and observations
    reset = eagerx.ResetNode.make("ResetAngle", "reset_angle", rate, u_range=[-2, +2], gains=[0.5, 0.4, 0.3])
    graph.add(reset)

    graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
    graph.connect(action="voltage", target=reset.feedthroughs.u)
    graph.connect(source=reset.outputs.u, target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)
    graph.connect(source=pendulum.sensors.x, observation="angle_data")

    # Render image
    overlay = eagerx.Node.make("Overlay", "overlay", user_name=user_name, rate=image_rate, process=eagerx.process.NEW_PROCESS)
    graph.add(overlay)
    graph.connect(source=pendulum.sensors.x, target=overlay.inputs.x)
    graph.connect(action="voltage", target=overlay.inputs.u)
    graph.connect(source=pendulum.sensors.image, target=overlay.inputs.base_image)
    graph.render(source=overlay.outputs.image, rate=image_rate, process=eagerx.process.NEW_PROCESS)

    engine = eagerx.Engine.make("RealEngine", rate=rate, sync=True, process=eagerx.process.NEW_PROCESS)

    env = Flatten(PendulumEnv(name="real", rate=rate, graph=graph, engine=engine))

    model = sb3.SAC.load(model_path)

    # Toggle render
    env.render("human")

    while True:
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

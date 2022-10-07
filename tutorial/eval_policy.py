import eagerx
from eagerx.wrappers import Flatten
from eagerx.backends.ros1 import Ros1
from eagerx.backends.single_process import SingleProcess


from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_dcsc_setups.pendulum.nodes import ResetAngle
from eagerx_dcsc_setups.pendulum.overlay import Overlay

from eagerx_reality.engine import RealEngine

import stable_baselines3 as sb3
from typing import Dict
import numpy as np
import rospy


class PendulumEnv(eagerx.BaseEnv):
    def __init__(self, name: str, rate: float, user_name: str = None):
        """Initializes an environment with EAGERx dynamics.

        :param name: The name of the environment. Everything related to this environment
                     (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
        :param rate: The rate (Hz) at which the environment will run.
        :param self.graph: The self.graph consisting of nodes and objects that describe the environment's dynamics.
        :param engine: The physics engine that will govern the environment's dynamics.
        :param eval: If True we will create an evaluation environment, i.e. not performing domain randomization.
        """
        self.rate = rate
        self.user_name = user_name
        # rospy.init_node("env")

        # Make the backend specification
        backend = Ros1.make()
        # backend = SingleProcess.make(log_level=eagerx.core.constants.INFO)

        # Create Engine
        engine = RealEngine.make(rate=rate, sync=True, process=eagerx.NEW_PROCESS)

        # Maximum episode length
        self.episode_length = 300

        # Make graph
        self._make_graph()

        # Step counter
        self.steps = None
        super().__init__(name, rate, self.graph, engine, backend, force_start=True)

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

    def _make_graph(self):
        u_limit = 2.0
        sensor_rate = self.rate
        actuator_rate = self.rate
        image_rate = self.rate / 2

        # Create empty self.graph
        self.graph = eagerx.Graph.create()

        # Create pendulum
        pendulum = Pendulum.make(
            "pendulum",
            render_shape=[480, 480],
            actuators=["u"],
            sensors=["x"],
            states=["model_state"],
            sensor_rate=sensor_rate,
            actuator_rate=actuator_rate,
            image_rate=image_rate,
        )
        self.graph.add(pendulum)

        # reset = ResetAngle.make("reset_angle", sensor_rate, u_range=[-u_limit, +u_limit], gains=[0.5, 0.2, 0.5])
        # self.graph.add(reset)

        # self.graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
        # self.graph.connect(action="voltage", target=reset.feedthroughs.u)
        self.graph.connect(action="voltage", target=pendulum.actuators.u)
        # self.graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)
        self.graph.connect(source=pendulum.sensors.x, observation="angle_data")

        # Add rendering
        self.graph.add_component(pendulum.sensors.image)
        overlay = Overlay.make("overlay", rate=image_rate, user_name=self.user_name, process=eagerx.NEW_PROCESS)
        self.graph.add(overlay)
        self.graph.connect(source=pendulum.sensors.x, target=overlay.inputs.x)
        self.graph.connect(action="voltage", target=overlay.inputs.u, skip=True)
        self.graph.connect(source=pendulum.sensors.image, target=overlay.inputs.base_image)
        self.graph.render(source=overlay.outputs.image, rate=image_rate, display=True)

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
    eagerx.set_log_level(eagerx.WARN)

    user_name = "George"
    file_name = "pendulum_george.zip"
    model_path = f"/home/jelle/Downloads/{file_name}"

    # Define rate (depends on rate of ode)
    rate = 20

    env = Flatten(PendulumEnv(name="real", rate=rate, user_name=user_name))

    model = sb3.SAC.load(model_path, device="cuda")

    # Toggle render
    env.render("human")

    while True:
        done = False
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
# ROS packages required
import eagerx
from eagerx.backends.single_process import SingleProcess
from eagerx.backends.ros1 import Ros1

# Environment
from eagerx.wrappers import Flatten

# Implementation specific
from eagerx_ode.engine import OdeEngine
from eagerx_reality.engine import RealEngine
from eagerx_dcsc_setups.pendulum.objects import Pendulum
from eagerx_dcsc_setups.pendulum.nodes import ResetAngle
from eagerx_dcsc_setups.pendulum.overlay import Overlay

# Stable Baselines 3 and Gym
import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction

# Other
from typing import Dict
import numpy as np
import rospy
from pathlib import Path


class PendulumEnv(eagerx.BaseEnv):
    def __init__(self, name: str, rate: float, eval=False):
        """Initializes an environment with EAGERx dynamics.

        :param name: The name of the environment. Everything related to this environment
                     (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
        :param rate: The rate (Hz) at which the environment will run.
        :param self.graph: The self.graph consisting of nodes and objects that describe the environment's dynamics.
        :param engine: The physics engine that will govern the environment's dynamics.
        :param eval: If True we will create an evaluation environment, i.e. not performing domain randomization.
        """
        self.rate = rate
        self.eval = eval

        # Make the backend specification
        # backend = Ros1.make(log_level=eagerx.core.constants.INFO) if eval else SingleProcess.make()
        backend = Ros1.make(log_level=eagerx.core.constants.INFO)

        # Create Engine
        engine = RealEngine.make(rate=rate, sync=True, process=eagerx.NEW_PROCESS) if eval else OdeEngine.make(rate=rate)
        # engine = OdeEngine.make(rate=rate, real_time_factor=1.0) if eval else OdeEngine.make(rate=rate)

        # Maximum episode length
        self.episode_length = 300 if eval else 100

        # Make graph
        self._make_graph()
        # self.graph.gui()

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

        # if self.eval:
        if True:
            reset = ResetAngle.make("reset_angle", sensor_rate, u_range=[-u_limit, +u_limit], process=eagerx.NEW_PROCESS)
            self.graph.add(reset)

            self.graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
            self.graph.connect(action="voltage", target=reset.feedthroughs.u)
            self.graph.connect(source=reset.outputs.u, target=pendulum.actuators.u)
            self.graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)
        else:
            self.graph.connect(action="voltage", target=pendulum.actuators.u)

        self.graph.connect(source=pendulum.sensors.x, observation="angle_data")

        # Add rendering
        # self.graph.add_component(pendulum.sensors.image)
        # overlay = Overlay.make("overlay", rate=image_rate)
        # self.graph.add(overlay)
        # self.graph.connect(source=pendulum.sensors.x, target=overlay.inputs.x)
        # self.graph.connect(action="voltage", target=overlay.inputs.u, skip=True)
        # self.graph.connect(source=pendulum.sensors.image, target=overlay.inputs.base_image)
        # self.graph.render(source=overlay.outputs.image, rate=image_rate, display=True)

    def reset(self) -> Dict:
        """Resets the environment to an initial state and returns an initial observation.

        :returns: The initial observation.
        """
        # Determine reset states
        states = self.state_space.sample()
        # states["pendulum/model_parameters"] = np.array(
        #     [
        #         0.000159931461600856,
        #         0.0508581731919534,
        #         0.0415233722862552,
        #         1.43298488358436e-05,
        #         0.0333391179016334,
        #         7.73125142447252,
        #         0.000975041213361349,
        #         165.417960777425,
        #     ],
        #     dtype="float32",
        # )
        if self.eval:
            # During evaluation on the real system we cannot set the state to an arbitrary position and velocity
            offset = np.random.rand() - 0.5
            theta = np.pi - offset if offset > 0 else -np.pi - offset
            states["pendulum/model_state"] = np.array([theta, 0], dtype="float32")

        # Perform reset
        obs = self._reset(states)

        # Reset step counter
        self.steps = 0
        return obs


if __name__ == "__main__":
    rate = 20
    seed = 1

    root = Path(__file__).parent.parent
    train_log_dir = root / "exps" / "train" / "runs" / f"domain_randomization_0"
    LOAD_DIR = str(train_log_dir) + f"/rl_model_5000_steps.zip"

    simulation_env = PendulumEnv("sim_env", rate, eval=False)
    real_env = PendulumEnv("real_env", rate, eval=True)

    # Flatten and scale envs
    simulation_env = RescaleAction(Flatten(simulation_env), min_action=-1.0, max_action=1.0)
    real_env = RescaleAction(Flatten(real_env), min_action=-1.0, max_action=1.0)

    # Seed envs
    simulation_env.seed(seed)
    real_env.seed(seed)

    # Check simulation env validity
    check_env(simulation_env)

    # First train in simulation
    simulation_env.render("human")
    model = sb3.SAC("MlpPolicy", simulation_env, verbose=1, seed=seed, learning_rate=7e-4)
    # model.learn(total_timesteps=4_000)
    simulation_env.close()

    # Evaluate for 30 seconds in simulation
    rospy.loginfo("Start simulation evaluation!")
    obs = simulation_env.reset()
    for i in range(int(30 * rate)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = simulation_env.step(action)
        if done:
            obs = simulation_env.reset()

    # model.save("simulation")
    simulation_env.shutdown()
    model = sb3.SAC.load(LOAD_DIR)

    # Evaluate on real system
    real_env.render("human")
    rospy.loginfo("Start zero-shot evaluation!")
    obs = real_env.reset()
    obs = real_env.reset()
    i = 0
    s = -1
    while True:
        if i % 10 == 0:
            s *= -1
        action, _states = model.predict(obs, deterministic=True)
        # action = s*((2*i/600)*(action / action) - 1)
        obs, reward, done, info = real_env.step(action)
        i += 1
        if done:
            i = 0
            s = -1
            obs = real_env.reset()

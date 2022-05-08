from eagerx import Object, Node, process, Graph, ResetNode
import numpy as np
import rospy
from time import sleep


# Define step function
def step_fn(prev_obs, obs, action, steps, length_eps):
    state = obs["observation"][0]
    u = action["action"][0]

    # Calculate reward
    sin_th, cos_th, thdot = state
    th = np.arctan2(sin_th, cos_th)
    cost = th ** 2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u ** 2

    # Determine done flag
    done = steps > length_eps
    # Set info:
    info = {"TimeLimit.truncated": done}
    return obs, -cost, done, info


def train_reset_fn(env, train_delay):
    states = env.state_space.sample()
    if train_delay is None:
        states["pendulum/pendulum_actuator/delay"] = None
    else:
        actuator_delay = train_delay if np.random.random() > 0.9 else 0.
        states["pendulum/pendulum_actuator/delay"] = np.array(actuator_delay, dtype="float32")
    return states


def eval_reset_fn(env, eval_delay):
    states = env.state_space.sample()
    # Delay
    if eval_delay is None:
        if "pendulum/pendulum_actuator/delay" in states:
            states["pendulum/pendulum_actuator/delay"] = None
    else:
        actuator_delay = eval_delay if np.random.random() > 0.9 else 0.
        states["pendulum/pendulum_actuator/delay"] = np.array(actuator_delay, dtype="float32")
    # Model state
    offset = np.random.rand() - 0.5
    theta = np.pi - offset if offset > 0 else -np.pi - offset
    states["pendulum/model_state"] = np.array([theta, 0], dtype="float32")
    return states


def eval_env(model, env, eval_eps, real_delay=None):
    cumulative_reward = 0
    rospy.loginfo("Start evaluation!")
    for episode in range(eval_eps):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            if real_delay is not None and np.random.random() > 0.9:
                sleep(real_delay)
    return float(cumulative_reward)


def make_graph(
        DR: bool,
        FA: bool,
        evaluation: bool,
        sensor_rate: float,
        actuator_rate: float,
        image_rate: float,
        render: bool = False,
):
    u_limit = 2.
    states = ["model_state"]

    if DR:
        states.append("model_parameters")

    # Create empty graph
    graph = Graph.create()

    # Create pendulum
    pendulum = Object.make(
        "Pendulum",
        "pendulum",
        render_shape=[480, 480],
        sensors=["x"],
        states=states,
        sensor_rate=sensor_rate,
        actuator_rate=actuator_rate,
        image_rate=image_rate,
        fixed_delay=0.0,
    )
    graph.add(pendulum)

    # Create Butterworth filter
    bf = Node.make(
        "ButterworthFilter",
        name="butterworth_filter",
        rate=sensor_rate,
        N=2,
        Wn=13,
        process=process.NEW_PROCESS,
    )
    bf.inputs.signal.space_converter = pendulum.actuators.u.space_converter
    bf.outputs.filtered.space_converter = pendulum.actuators.u.space_converter
    graph.add(bf)

    if evaluation:
        reset = ResetNode.make("ResetAngle", "reset_angle", sensor_rate, u_range=[-u_limit, +u_limit])
        graph.add(reset)

        graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
        graph.connect(action="action", target=reset.feedthroughs.u)
        graph.connect(source=reset.outputs.u, target=bf.inputs.signal)
        graph.connect(source=pendulum.sensors.x, target=reset.inputs.x)
    else:
        graph.connect(action="action", target=bf.inputs.signal)
    graph.connect(source=bf.outputs.filtered, target=pendulum.actuators.u)
    graph.connect(source=pendulum.sensors.x, observation="observation")

    if render:
        graph.add_component(pendulum.sensors.image)
        graph.render()

    if FA > 0:
        graph.connect(source=bf.outputs.filtered, observation="action_applied", skip=True, initial_obs=[0], window=FA)
    return graph

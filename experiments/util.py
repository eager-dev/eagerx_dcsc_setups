from eagerx import Object, Node, process, Graph, ResetNode
import eagerx_dcsc_setups
import numpy as np
import rospy
from time import sleep
from matplotlib import animation
import matplotlib.pyplot as plt
from typing import List
import os


def save_frames_as_gif(dt: float, frames: List, filename="some_episode.gif", dpi: int = 72):
    """
    :param dt: 1/rate of the environment
    :param frames: list containing all the rgb arrays
    :param path: Path to the log directory where the gif is going to be saved
    :param filename: name of the gif e.g. "NAME_eps_1.gif"
    :param dpi: 72 dpi is ok.
    :return:
    """

    # Mess with this to change frame size
    fig = plt.figure(figsize=(frames[0].shape[1] / 72, frames[0].shape[0] / 72.0), dpi=dpi)
    ax = fig.gca()
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(filename, writer="Pillow", fps=int(1 / dt))
    plt.close(fig)
    print("Gif saved to %s" % (filename))


# Define step function
def step_fn(prev_obs, obs, action, steps, length_eps):
    state = obs["observation"][0]
    u = action["action"][0]

    # Calculate reward
    cos_th, sin_th, thdot = state
    th = np.arctan2(sin_th, cos_th)
    cost = th**2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u**2

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
        # actuator_delay = train_delay if np.random.random() > 0.9 else 0.
        actuator_delay = train_delay
        states["pendulum/pendulum_actuator/delay"] = np.array(actuator_delay, dtype="float32")
    return states


def eval_reset_fn(env, eval_delay):
    states = env.state_space.sample()
    # Delay
    if eval_delay is None:
        if "pendulum/pendulum_actuator/delay" in states:
            states["pendulum/pendulum_actuator/delay"] = None
    else:
        # actuator_delay = eval_delay if np.random.random() > 0.9 else 0.
        actuator_delay = eval_delay
        states["pendulum/pendulum_actuator/delay"] = np.array(actuator_delay, dtype="float32")
    # Model state
    offset = np.random.rand() - 0.5
    theta = np.pi - offset if offset > 0 else -np.pi - offset
    states["pendulum/model_state"] = np.array([theta, 0], dtype="float32")
    return states


def eval_env(model, env, eval_eps, real_delay=None, gif_file=None):
    img_array = []
    cumulative_reward = 0
    rospy.loginfo("Start evaluation!")
    i = 0
    for episode in range(eval_eps):
        done = False
        obs = env.reset()
        while not done:
            i += 1
            if real_delay is not None:
                sleep(real_delay)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if gif_file is not None and i % 2 == 0:
                img = env.render("rgb_array")
                if not 0 in img.shape:
                    img_array.append(img)
            cumulative_reward += reward
    if gif_file is not None:
        save_frames_as_gif(1 / 15, img_array, dpi=72, filename=gif_file)
    return float(cumulative_reward)


def make_graph(
    DR: bool,
    FA: int,
    evaluation: bool,
    sensor_rate: float,
    actuator_rate: float,
    image_rate: float,
    render: bool = False,
    FD=0.0,
):
    u_limit = 2.0
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
        fixed_delay=FD,
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
        reset = ResetNode.make("ResetAngle", "reset_angle", sensor_rate, u_range=[-u_limit, +u_limit], gains=[1.0, 0.4, 0.5])
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
        overlay = Node.make("Overlay", "overlay", rate=image_rate, process=process.NEW_PROCESS)
        graph.add(overlay)
        graph.connect(source=pendulum.sensors.x, target=overlay.inputs.x)
        graph.connect(source=bf.outputs.filtered, target=overlay.inputs.u)
        graph.connect(source=pendulum.sensors.image, target=overlay.inputs.base_image)
        graph.render(source=overlay.outputs.image, rate=image_rate, display=False, process=process.NEW_PROCESS)

    if FA > 0:
        graph.connect(source=bf.outputs.filtered, observation="action_applied", skip=True, initial_obs=[0], window=FA)
    return graph

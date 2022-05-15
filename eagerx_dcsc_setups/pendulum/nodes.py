import rospy
from typing import Optional, List
import eagerx
from eagerx.utils.utils import Msg
from std_msgs.msg import Float32MultiArray, Float32, Bool
import numpy as np
from time import sleep


class FixedDelayAction(eagerx.Node):
    @staticmethod
    @eagerx.register.spec("FixedDelayAction", eagerx.Node)
    def spec(
        spec,
        name: str,
        rate: float,
        fixed_delay: float,
        process: Optional[int] = eagerx.process.NEW_PROCESS,
        color: Optional[str] = "green",
    ):
        """DelayAction spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(FixedDelayAction)

        # Modify default node params
        params = dict(
            name=name, rate=rate, process=process, color=color, inputs=["observation", "action"], outputs=["delayed_action"]
        )
        spec.config.update(params)

        # Custom params
        spec.config.fixed_delay = fixed_delay

    def initialize(self, fixed_delay):
        assert self.sync, "This node only works in sync mode."
        # todo: check that observation, action rate are the same, and equal to the rate of this node.
        self.fixed_delay = fixed_delay

    @eagerx.register.states()
    def reset(self):
        pass

    @eagerx.register.inputs(observation=Float32MultiArray, action=Float32MultiArray)
    @eagerx.register.outputs(delayed_action=Float32MultiArray)
    def callback(self, t_n: float, observation: Msg, action: Msg):
        if self.simulate_delays:
            return dict(delayed_action=action.msgs[-1])

        obs_ts = observation.info.t_in[-1].wc_stamp
        act_ts = action.info.t_in[-1].wc_stamp
        dt = act_ts - obs_ts
        sleep_duration = self.fixed_delay - dt

        if sleep_duration > 0:
            sleep(sleep_duration)
        else:
            rospy.logwarn("[DelayAction] Delay is already larger than desired delay.")
        return dict(delayed_action=action.msgs[-1])


def wrap_angle(angle):
    return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))


class ResetAngle(eagerx.ResetNode):
    @staticmethod
    @eagerx.register.spec("ResetAngle", eagerx.ResetNode)
    def spec(
        spec,
        name: str,
        rate: float,
        threshold: float = 0.1,
        timeout: float = 5.0,
        gains: Optional[List[float]] = None,
        u_range: Optional[List[float]] = None,
    ):
        """This AngleReset node resets the pendulum to a desired angle with zero angular velocity. Note that this controller
        only works properly when resetting the pendulum near the downward facing equilibrium.

        :param spec: Not provided by the user. Contains the configuration of this node to initialize it at run-time.
        :param name: Node name
        :param rate: Rate at which callback is called. Must be equal to the rate of the nodes that are connect to the feedthroughs.
        :param threshold: Absolute difference between the desired and goal state before considering the reset complete.
        :param timeout: Maximum time (seconds) before considering the reset finished (regardless whether the goal was reached).
        :param gains: Gains of the PID controller used to reset.
        :param u_range: Min and max action.
        :return:
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        # Note: not to be confused with the initialize method of this node.
        spec.initialize(ResetAngle)

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.process.ENVIRONMENT, color="grey")
        spec.config.update(inputs=["x"], targets=["goal"], outputs=["u"])
        spec.config.update(u_range=u_range, threshold=threshold, timeout=timeout)
        spec.config.gains = gains if isinstance(gains, list) else [2.0, 0.2, 1.0]

        # Add space_converter
        c = eagerx.SpaceConverter.make("Space_Float32MultiArray", [u_range[0]], [u_range[1]], dtype="float32")
        spec.outputs.u.space_converter = c

    def initialize(self, threshold: float, timeout: float, gains: List[float], u_range: List[float]):
        self.threshold = threshold
        self.timeout = timeout
        self.u_min, self.u_max = u_range

        # Creat a simple PID controller
        from eagerx_dcsc_setups.pendulum.pid import PID

        self.controller = PID(u0=0.0, kp=gains[0], kd=gains[1], ki=gains[2], dt=1 / self.rate)

    @eagerx.register.states()
    def reset(self):
        # Reset the internal state of the PID controller (ie the error term).
        self.controller.reset()
        self.ts_start_routine = None

    @eagerx.register.inputs(x=Float32MultiArray)
    @eagerx.register.targets(goal=Float32MultiArray)
    @eagerx.register.outputs(u=Float32MultiArray)
    def callback(self, t_n: float, goal: Msg, x: Msg):
        if self.ts_start_routine is None:
            self.ts_start_routine = t_n

        # Convert messages to floats and numpy array
        theta, dtheta = x.msgs[-1].data
        goal = np.array(goal.msgs[-1].data, dtype="float32")  # Take the last received message

        # Define downward angle as theta=0 (resolve downward discontinuity)
        theta += np.pi
        goal[0] += np.pi

        # Wrap angle between [-pi, pi]
        theta = wrap_angle(theta)
        goal[0] = wrap_angle(goal[0])

        # Overwrite the desired velocity to be zero.
        goal[1] = 0.0

        # Calculate the action using the PID controller
        # Select random actions instead.
        u = self.controller.next_action(theta, ref=goal[0])
        u = np.clip(u, self.u_min, self.u_max)  # Clip u to range

        # Determine if we have reached our goal state
        done = np.isclose(np.array([theta, dtheta]), goal, atol=self.threshold).all()

        # If the reset routine takes too long, we timeout the routine and simply assume that we are done.
        done = done or (t_n - self.ts_start_routine) > self.timeout

        # Prepare output message for transmission.
        # This must contain a message for every registered & selected output and target.
        # For targets, this message decides whether the goal state has been reached (or we, for example, timeout the reset).
        # The name for this target message is the registered target name + "/done".
        output_msgs = {"u": Float32MultiArray(data=[u]), "goal/done": Bool(data=done)}
        return output_msgs

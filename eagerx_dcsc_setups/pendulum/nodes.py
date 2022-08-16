import eagerx
from eagerx import register, Space
from eagerx.utils.utils import Msg
from eagerx.core.specs import ResetNodeSpec
import numpy as np
from typing import Optional, List


def wrap_angle(angle):
    return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))


class ResetAngle(eagerx.ResetNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        threshold: float = 0.1,
        timeout: float = 5.0,
        gains: Optional[List[float]] = None,
        u_range: Optional[List[float]] = None,
    ) -> ResetNodeSpec:
        """This ResetAngle node resets the pendulum to a desired angle with zero angular velocity. Note that this controller
        only works properly when resetting the pendulum near the downward facing equilibrium.

        :param spec: Not provided by the user. Contains the configuration of this node to initialize it at run-time.
        :param name: Node name
        :param rate: Rate at which callback is called. Must be equal to the rate of the nodes that are connect to the feedthroughs.
        :param threshold: Absolute difference between the desired and goal state before considering the reset complete.
        :param timeout: Maximum time (seconds) before considering the reset finished (regardless whether the goal was reached).
        :param gains: Gains of the PID controller used to reset.
        :param u_range: Min and max action.
        :return: Specification.
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        # Note: not to be confused with the initialize method of this node.
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.process.ENVIRONMENT, color="grey")
        spec.config.update(inputs=["x"], targets=["goal"], outputs=["u"])
        spec.config.update(u_range=u_range, threshold=threshold, timeout=timeout)
        spec.config.gains = gains if isinstance(gains, list) else [2.0, 0.2, 1.0]

        # Add space_converter
        c = Space(low=[u_range[0]], high=[u_range[1]], dtype="float32")
        spec.outputs.u.space = c
        return spec

    def initialize(self, spec: ResetNodeSpec):
        self.threshold = spec.config.threshold
        self.timeout = spec.config.timeout
        self.u_min, self.u_max = spec.config.u_range

        # Creat a simple PID controller
        from eagerx_dcsc_setups.pendulum.pid import PID

        gains = spec.config.gains
        self.controller = PID(u0=0.0, kp=gains[0], kd=gains[1], ki=gains[2], dt=1 / self.rate)

    @register.states()
    def reset(self):
        # Reset the internal state of the PID controller (ie the error term).
        self.controller.reset()
        self.ts_start_routine = None

    @register.inputs(x=Space(dtype="float32"))
    @register.targets(goal=Space(low=[-3.14, -9.0], high=[3.14, 9.0], dtype="float32"))
    @register.outputs(u=Space(dtype="float32"))
    def callback(self, t_n: float, goal: Msg, x: Msg):
        if self.ts_start_routine is None:
            self.ts_start_routine = t_n

        # Convert messages to floats and numpy array
        cos_theta, sin_theta, dtheta = x.msgs[-1]
        goal = np.array(goal.msgs[-1], dtype="float32")  # Take the last received message

        # Define downward angle as theta=0 (resolve downward discontinuity)
        theta = np.arctan2(sin_theta, cos_theta)
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
        output_msgs = {"u": np.array([u], dtype="float32"), "goal/done": bool(done)}
        return output_msgs

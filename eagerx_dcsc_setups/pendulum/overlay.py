import eagerx
from eagerx import register, Space
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg
import cv2
import numpy as np


class Overlay(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: int = eagerx.process.ENVIRONMENT,
        color: str = "cyan",
        user_name: str = None,
    ):
        """Overlay spec"""
        # Fills spec with defaults parameters
        spec = cls.get_specification()

        # Adjust default params
        spec.config.update(
            name=name, rate=rate, process=process, color=color, inputs=["base_image", "u", "x"], outputs=["image"]
        )

        spec.config.user_name = user_name
        return spec

    def initialize(self, spec: NodeSpec):
        self.user_name = spec.config.user_name

    @register.states()
    def reset(self):
        pass

    @register.inputs(base_image=Space(dtype="uint8"), u=Space(low=[-2], high=[2]), x=Space(dtype="float32"))
    @register.outputs(image=Space(dtype="uint8"))
    def callback(self, t_n: float, base_image: Msg, u: Msg, x: Msg):
        if len(base_image.msgs[-1]) > 1:
            u = u.msgs[-1][0] if u and u.msgs and u.msgs[-1] else 0
            theta = x.msgs[-1][0]
            theta = theta - 2 * np.pi * np.floor((theta + np.pi) / (2 * np.pi))

            # Set background image from base_image
            img = base_image.msgs[-1]
            height, width, _ = img.shape
            side_length = min(width, height)

            # Put text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Applied Voltage"
            text_size = cv2.getTextSize(text, font, 2 / 3, 2)[0]
            text_x = int((width - text_size[0]) / 2)
            text_y = int(text_size[1])
            img = cv2.rectangle(
                img,
                (text_x, 0),
                (int((width + text_size[0]) / 2), int(text_y * 1.5)),
                (255, 255, 255),
                -1,
            )
            img = cv2.putText(img, text, (text_x, text_y), font, 2 / 3, (0, 0, 0), thickness=2)

            # Draw grey bar
            img = cv2.rectangle(
                img,
                (width // 2 - side_length * 4 // 10, text_y * 2),
                (width // 2 + 4 * side_length // 10, text_y * 3),
                (125, 125, 125),
                -1,
            )

            # Fill bar proportional to the action that is applied
            p1 = (width // 2, text_y * 2)
            p2 = (width // 2 + int(side_length * u * 2 / 15), text_y * 3)
            img = cv2.rectangle(img, p1, p2, (0, 0, 255), -1)

            # Add info
            info = f"t ={t_n: .2f} s"
            if self.user_name is not None:
                info += f", user = {self.user_name}"

            info_width = cv2.getTextSize(info, font, 2 / 3, 2)[0][0]
            info_height = cv2.getTextSize(info, font, 2 / 3, 2)[0][1]

            info_x = (width - info_width) // 2
            info_y = height - info_height // 2

            img = cv2.rectangle(img, (info_x, height - 2 * info_height), (info_x + info_width, height), (255, 255, 255), -1)
            img = cv2.putText(img, info, (info_x, info_y), font, 2 / 3, (0, 0, 0), thickness=2)
            return dict(image=img)
        else:
            return dict(image=np.zeros((0, 0, 3), dtype="uint8"))

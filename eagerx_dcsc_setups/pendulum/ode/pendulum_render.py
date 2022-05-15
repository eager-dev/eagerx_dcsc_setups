import cv2
import numpy as np


def pendulum_render_fn(img, observation, action):
    height, width, _ = img.shape
    side_length = min(width, height)
    state = observation.msgs[-1].data

    img += 255
    length = 2 * side_length // 9
    sin_theta, cos_theta = np.sin(state[0]), np.cos(state[0])

    img = cv2.circle(img, (width // 2, height // 2), side_length // 3, (255, 0, 0), -1)
    img = cv2.circle(img, (width // 2, height // 2), side_length // 12, (192, 192, 192), -1)
    img = cv2.circle(
        img,
        (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)),
        side_length // 9,
        (192, 192, 192),
        -1,
    )

    # Draw velocity vector
    img = cv2.arrowedLine(
        img,
        (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)),
        (
            width // 2 + int(length * (sin_theta + state[1] * cos_theta / 5)),
            height // 2 + int(length * (-cos_theta + state[1] * sin_theta / 5)),
        ),
        (0, 0, 0),
        2,
    )
    return img

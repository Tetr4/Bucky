import cv2
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark  # type: ignore
from bucky.common.simple_types import ColorBGR, Point3D
from bucky.common.math_utils import clamp


def to_point_3d(landmark: NormalizedLandmark) -> Point3D:
    return Point3D(landmark.x, landmark.y, landmark.z)


def to_cv_coords(frame: np.ndarray, point: NormalizedLandmark | Point3D) -> cv2.typing.Point:
    h, w, _ = frame.shape
    return (int(point.x * w), int(point.y * h))


def probability_to_color(prob: float) -> ColorBGR:
    prob = clamp(prob, 0, 1)
    red = ColorBGR(0, 0, 255)
    yellow = ColorBGR(0, 255, 255)
    green = ColorBGR(0, 255, 0)
    if prob <= 0.5:
        # Interpolate between red and yellow
        t = prob / 0.5
        b = int(red.b * (1 - t) + yellow.b * t)
        g = int(red.g * (1 - t) + yellow.g * t)
        r = int(red.r * (1 - t) + yellow.r * t)
    else:
        # Interpolate between yellow and green
        t = (prob - 0.5) / 0.5
        b = int(yellow.b * (1 - t) + green.b * t)
        g = int(yellow.g * (1 - t) + green.g * t)
        r = int(yellow.r * (1 - t) + green.r * t)
    return ColorBGR(b, g, r)

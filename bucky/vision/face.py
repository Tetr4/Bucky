from __future__ import annotations
import cv2
import math
import numpy as np
from typing import Optional
from google._upb._message import RepeatedCompositeContainer
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark  # type: ignore
from dataclasses import dataclass
from bucky.common.conversion import to_cv_coords, to_point_3d
import bucky.common.math_utils as math_utils
from bucky.common.simple_types import Point3D, ColorBGR, EulerAngles


@dataclass
class FaceRawData:
    camera_image: np.ndarray
    landmarks: RepeatedCompositeContainer


@dataclass
class Eye:
    raw_data: FaceRawData
    contour_ids: list[int]
    iris_id: int

    @property
    def contour(self) -> list[Point3D]:
        result = []
        for i in range(len(self.contour_ids)):
            result.append(to_point_3d(self.raw_data.landmarks[self.contour_ids[i]]))
        return result

    @property
    def iris_center(self) -> Point3D:
        return to_point_3d(self.raw_data.landmarks[self.iris_id])

    def draw(self, image: np.ndarray, color: ColorBGR):
        c: list[Point3D] = self.contour
        for i in range(len(c)):
            a = to_cv_coords(image, c[i])
            b = to_cv_coords(image, c[(i + 1) % len(c)])
            cv2.line(image, a, b, color, 2)

        iris_cv_pos: cv2.typing.Point = to_cv_coords(image, self.raw_data.landmarks[self.iris_id])
        cv2.circle(image, iris_cv_pos, 4, color, 1)


class Face:
    IMPORTANT_LANDMARKS_IDS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 263,
        "right_eye_outer": 33,
        "left_mouth": 287,
        "right_mouth": 57
    }

    def __init__(self, raw_data: FaceRawData):
        self.raw_data = raw_data

        self.left_eye = Eye(
            raw_data=raw_data,
            contour_ids=[133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173],
            iris_id=468)

        self.right_eye = Eye(
            raw_data=raw_data,
            contour_ids=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            iris_id=473)

    @property
    def important_landmarks(self) -> dict[str, Point3D]:
        result: dict[str, Point3D] = {}
        for name, idx in Face.IMPORTANT_LANDMARKS_IDS.items():
            result[name] = to_point_3d(self.raw_data.landmarks[idx])
        return result

    @property
    def rotation(self) -> Optional[EulerAngles]:
        # 3D model points of a generic face
        model_points = np.array([
            [0.0, 0.0, 0.0],         # Nose tip
            [0.0, -63.6, -12.5],     # Chin
            [-43.3, 32.7, -26.0],    # Left eye left corner
            [43.3, 32.7, -26.0],     # Right eye right corner
            [-28.9, -28.9, -24.1],   # Left Mouth corner
            [28.9, -28.9, -24.1]     # Right mouth corner
        ])

        points: dict[str, Point3D] = self.important_landmarks
        image_points = [
            to_cv_coords(self.raw_data.camera_image, points["nose_tip"]),
            to_cv_coords(self.raw_data.camera_image, points["chin"]),
            to_cv_coords(self.raw_data.camera_image, points["left_eye_outer"]),
            to_cv_coords(self.raw_data.camera_image, points["right_eye_outer"]),
            to_cv_coords(self.raw_data.camera_image, points["left_mouth"]),
            to_cv_coords(self.raw_data.camera_image, points["right_mouth"])]
        image_points = np.array(image_points, dtype="double")

        # Image size
        size = self.raw_data.camera_image.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Get rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)

        # Convert to Euler angles
        def rotationMatrixToEulerAngles(R):
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            return np.degrees([x, y, z])  # pitch, yaw, roll

        pitch, yaw, roll = rotationMatrixToEulerAngles(rotation_mat)
        return EulerAngles(pitch, yaw, roll)

    @property
    def attention(self) -> float:
        rot = self.rotation
        if not rot:
            return 0.0

        # print(f"Pitch: {rot.pitch:.2f}, Yaw: {rot.yaw:.2f}, Roll: {rot.roll:.2f}")

        yaw_prob = math_utils.smoothstep(math_utils.remap(abs(rot.yaw), 0, 90, 1, 0), 0, 1)
        pitch_prob = math_utils.smoothstep(math_utils.remap(abs(rot.pitch), 0, 20, 1, 0), 0, 1)
        return yaw_prob * pitch_prob

    def draw(self, image: np.ndarray, color: ColorBGR):
        self.left_eye.draw(image, color)
        self.right_eye.draw(image, color)

        for pos in self.important_landmarks.values():
            cv2.circle(image, to_cv_coords(image, pos), 3, color, -1)

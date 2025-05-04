from __future__ import annotations
import cv2
import numpy as np
import logging
import time
import queue
from threading import RLock, Thread, Event
from typing import Literal, Optional, Callable
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
from bucky.common.conversion import probability_to_color
from bucky.common.simple_types import ColorBGR, Point2D
from bucky.vision import CameraStream
from bucky.vision.face import Face, FaceRawData

logger = logging.getLogger(__name__)


class UserTracker:
    def __init__(self,
                 cam_stream_factory: Callable[[], CameraStream],
                 max_num_faces: int = 2,
                 debug_mode: bool = True):
        self._data_lock = RLock()
        self._face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_num_faces)
        self._cam_stream_factory = cam_stream_factory
        self._faces: list[Face] = []

        self._debug_mode = debug_mode
        self._debug_queue = queue.Queue(maxsize=3)
        self._debug_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=0, circle_radius=0, color=(128, 128, 128))
        self._debug_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(64, 64, 64))

        self._continue = Event()
        self._thread_lock = RLock()
        self._update_thread: Optional[Thread] = None

        if self._debug_mode:
            cv2.namedWindow("user tracking", cv2.WINDOW_NORMAL)
            self._debug_window_thread = Thread(target=self._debug_window_proc, daemon=True)
            self._debug_window_thread.start()

    @property
    def face_with_max_attention(self) -> Optional[Face]:
        with self._data_lock:
            return max(self._faces, key=lambda face: face.attention, default=None)

    @property
    def max_attention(self) -> float:
        with self._data_lock:
            if face := self.face_with_max_attention:
                logger.info(f"max_attention={face.attention:.2f}")
                return face.attention
            return 0.0

    @property
    def user_direction(self) -> Literal["unknown", "front", "left", "right"]:
        if face := self.face_with_max_attention:
            pos: Point2D = face.position
            if pos.x < -0.3:
                return "left"
            elif pos.x > 0.3:
                return "right"
            else:
                return "front"
        return "unknown"

    def start(self):
        with self._thread_lock:
            self.stop()
            self._continue.set()
            self._update_thread = Thread(target=self._update_proc, daemon=True)
            self._update_thread.start()

    def stop(self):
        with self._thread_lock:
            if self._update_thread:
                self._continue.clear()
                self._update_thread.join()
                self._update_thread = None

    def _update_proc(self):
        while self._continue.is_set():
            cam_stream: CameraStream = self._cam_stream_factory()
            with cam_stream:
                while self._continue.is_set():
                    camera_image = cam_stream.read()
                    if camera_image is not None:
                        start: float = time.time()
                        self._update(camera_image)
                        dt: float = time.time() - start
                        time.sleep(max(0.0, 30/1000 - dt))
                    else:
                        time.sleep(3.0)
                        break

    def _update(self, camera_image: np.ndarray):
        rgb = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        with self._data_lock:
            self._faces.clear()
            if results.multi_face_landmarks:  # type: ignore
                for i, face_landmarks in enumerate(results.multi_face_landmarks):  # type: ignore
                    face = Face(FaceRawData(camera_image, face_landmarks.landmark))
                    self._faces.append(face)
                    if not self._debug_mode:
                        continue

                    # mp_drawing.draw_landmarks(camera_image, face_landmarks,
                    #                           mp_face_mesh.FACEMESH_TESSELATION,  # type: ignore
                    #                           landmark_drawing_spec=self._debug_landmark_drawing_spec,
                    #                           connection_drawing_spec=self._debug_connection_drawing_spec)

                    col: ColorBGR = probability_to_color(face.attention)
                    face.draw(camera_image, col)

                if self._debug_mode and (face := self.face_with_max_attention):
                    max_att: float = face.attention
                    pos: Point2D = face.position
                    col: ColorBGR = probability_to_color(max_att)

                    def draw_text(txt: str, pos: cv2.typing.Point, font_scale: float):
                        cv2.putText(camera_image, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 5)
                        cv2.putText(camera_image, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 2)

                    draw_text(f"Position: x={pos.x:.2f} y={pos.y:.2f} {self.user_direction}", (50, 30), 0.8)
                    draw_text(f"Attention: {int(max_att * 100)}%", (50, 70), 1)

            if self._debug_mode:
                self._debug_queue.put(camera_image)

    def _debug_window_proc(self):
        while True:
            camera_image = self._debug_queue.get()
            if camera_image is None:
                break
            cv2.imshow(f"user tracking", camera_image)


# pdm run python -m bucky.vision.user_tracking
if __name__ == "__main__":
    from bucky.vision.webcam import WebCameraStream
    tracker = UserTracker(lambda: WebCameraStream(800, 600), max_num_faces=2, debug_mode=True)
    tracker.start()
    input("press ENTER to exit")
    tracker.stop()

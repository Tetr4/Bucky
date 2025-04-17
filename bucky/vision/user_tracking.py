from __future__ import annotations
import cv2
import numpy as np
import logging
import time
import queue
from threading import RLock, Thread, Event
from typing import Optional, Callable
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
from bucky.common.conversion import probability_to_color
from bucky.common.simple_types import ColorBGR
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
        self._debug_face_colors: list[ColorBGR] = [ColorBGR(0, 255, 0), ColorBGR(255, 0, 255), ColorBGR(255, 255, 0)]
        self._debug_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=0, circle_radius=0, color=(128, 128, 128))
        self._debug_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(64, 64, 64))

        self._continue = Event()
        self._thread_lock = RLock()
        self._update_thread: Optional[Thread] = None

        if self._debug_mode:
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
                logger.info(f"max_attention={face.attention}")
                return face.attention
            return 0.0

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
        cam_stream: CameraStream = self._cam_stream_factory()
        with cam_stream:
            while self._continue.is_set():
                with self._data_lock:
                    camera_image = cam_stream.read()
                    if camera_image is not None:
                        self._update(camera_image)
                        time.sleep(0.001)
                        continue
                time.sleep(1.0)

    def _update(self, camera_image: np.ndarray):
        with self._data_lock:
            self._faces.clear()
            rgb = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if results.multi_face_landmarks:  # type: ignore
                for i, face_landmarks in enumerate(results.multi_face_landmarks):  # type: ignore
                    face = Face(FaceRawData(camera_image, face_landmarks.landmark))
                    self._faces.append(face)
                    if not self._debug_mode:
                        continue

                    mp_drawing.draw_landmarks(camera_image, face_landmarks,
                                              mp_face_mesh.FACEMESH_TESSELATION,  # type: ignore
                                              landmark_drawing_spec=self._debug_landmark_drawing_spec,
                                              connection_drawing_spec=self._debug_connection_drawing_spec)

                    col: ColorBGR = probability_to_color(face.attention)
                    face.draw(camera_image, col)

                if self._debug_mode and (face := self.face_with_max_attention):
                    max_att: float = face.attention
                    col: ColorBGR = probability_to_color(max_att)
                    label: str = f"Attention: {int(max_att * 100)}%"
                    cv2.putText(camera_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                    cv2.putText(camera_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

            if self._debug_mode:
                self._debug_queue.put(camera_image)

    def _debug_window_proc(self):
        while True:
            camera_image = self._debug_queue.get()
            if camera_image is None:
                break
            cv2.imshow(f"user tracking", camera_image)
            cv2.waitKey(1)

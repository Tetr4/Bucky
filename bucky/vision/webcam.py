from typing import Optional
import cv2
import numpy as np
from bucky.vision import CameraStream
import logging


logger = logging.getLogger(__name__)

class WebCameraStream(CameraStream):
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._cam = cv2.VideoCapture(0)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __exit__(self, type, value, traceback):
        self._cam.release()

    def read(self) -> Optional[np.ndarray]:
        if self._cam.isOpened():
            ret, frame = self._cam.read()
            if ret:
                return frame
            logger.error(f"{self._cam.getBackendName()}: Capturing camera image failed.")
        else:
            logger.error("Failed to open camera.")
        return None
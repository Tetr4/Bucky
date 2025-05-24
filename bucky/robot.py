from threading import Thread, RLock
from typing import Optional
import base64
import httpx
import queue
import numpy as np
import requests
import cv2
import time
from abc import ABC, abstractmethod
from bucky.vision import CameraStream
import logging


logger = logging.getLogger(__name__)


class IRobot(ABC):
    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_happy(self, delay: float = 0.0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_angry(self, delay: float = 0.0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_tired(self, delay: float = 0.0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_doze(self, delay: float = 0.0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_idle(self, delay: float = 0.0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_attention(self, delay: float = 0.0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def turn_left(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def turn_right(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def take_image(self, width: int = 640, height: int = 480) -> str:
        raise NotImplementedError()

    @abstractmethod
    def open_camera_stream(self, width: int = 800, height: int = 600) -> CameraStream:
        raise NotImplementedError()


class FakeBot(IRobot):
    def __init__(self):
        self._lock = RLock()
        self._cam: Optional[cv2.VideoCapture] = None
        self._cam_resolution: tuple[int, int] = (0, 0)
        self._cam_streams: int = 0

    def _add_ref_camera(self):
        with self._lock:
            if self._cam is None:
                logger.debug("creating VideoCapture(0) instance")
                self._cam = cv2.VideoCapture(0)
            self._cam_streams += 1

    def _release_camera(self):
        with self._lock:
            self._cam_streams = max(0, self._cam_streams - 1)
            if self._cam_streams == 0 and self._cam is not None:
                logger.debug("releasing VideoCapture(0) instance")
                self._cam_resolution = (0, 0)
                try:
                    self._cam.release()
                except Exception as error:
                    logger.error(error)
                finally:
                    self._cam = None

    def _read_camera_frame(self, width: int, height: int) -> Optional[np.ndarray]:
        with self._lock:
            if self._cam is not None:
                if self._cam_resolution != (width, height):
                    logger.debug(f"changing camera resolution to {width}x{height}")
                    self._cam_resolution = (width, height)
                    self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    time.sleep(0.2)  # wait for camera to adjust to lighting

                if self._cam.isOpened():
                    ret, frame = self._cam.read()
                    if ret:
                        return frame
                    logger.error(f"{self._cam.getBackendName()}: Capturing camera image failed.")
                else:
                    logger.error("Failed to open camera.")
        return None

    def release(self) -> None:
        pass

    def emote_happy(self, delay: float = 0.0) -> None:
        print("😀")

    def emote_angry(self, delay: float = 0.0) -> None:
        print("😡")

    def emote_tired(self, delay: float = 0.0) -> None:
        print("😩")

    def emote_doze(self, delay: float = 0.0) -> None:
        pass

    def emote_idle(self, delay: float = 0.0) -> None:
        pass

    def emote_attention(self, delay: float = 0.0) -> None:
        pass

    def turn_left(self) -> None:
        pass

    def turn_right(self) -> None:
        pass

    def take_image(self, width=640, height=480) -> str:
        with self._lock:
            cam_stream = self.open_camera_stream(width, height)
            with cam_stream:
                frame: Optional[cv2.typing.MatLike] = cam_stream.read()
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame)
                    jpeg_bytes = jpeg.tobytes()
                    logger.info(f"Captured camera image size {len(jpeg_bytes)} bytes.")
                    # with open("assets/images/captured_image.jpg", "wb") as file:
                    #     file.write(jpeg_bytes)
                    return base64.b64encode(jpeg_bytes).decode("utf-8")

        default_img_path = "assets/images/horse.jpg"
        logger.error(f"Capturing camera image failed. Using default image: {default_img_path}")
        with open(default_img_path, "rb") as image_file:
            print(default_img_path)
            return base64.b64encode(image_file.read()).decode("utf-8")

    def open_camera_stream(self, width: int = 800, height: int = 600) -> CameraStream:
        class FakeBotCameraStream(CameraStream):
            def __init__(self, bot: FakeBot, width: int, height: int):
                self._bot = bot
                self._width = width
                self._height = height
                self._bot._add_ref_camera()

            def __exit__(self, type, value, traceback):
                self._bot._release_camera()

            def read(self) -> Optional[np.ndarray]:
                return self._bot._read_camera_frame(self._width, self._height)

        return FakeBotCameraStream(self, width, height)


class BuckyBot(IRobot):
    def __init__(self, url: str):
        self.url = url
        self.job_queue = queue.Queue()

        def thread_proc():
            while func := self.job_queue.get():
                try:
                    func()
                except Exception as ex:
                    print("ERROR", str(ex))

        self.thread = Thread(target=thread_proc, daemon=True)
        self.thread.start()

    def __run_async(self, func) -> None:
        self.job_queue.put(func)

    def release(self) -> None:
        if self.thread.is_alive():
            self.job_queue.put(None)
            self.thread.join(5.0)

    def emote_happy(self, delay: float = 0.0) -> None:
        def func():
            time.sleep(delay)
            requests.get(f"{self.url}/eyes/set_mood?mood=HAPPY")
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/anim_laugh")
            time.sleep(2)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_mood?mood=NEUTRAL")
        self.__run_async(func)

    def emote_angry(self, delay: float = 0.0) -> None:
        def func():
            time.sleep(delay)
            requests.get(f"{self.url}/eyes/set_mood?mood=ANGRY")
            requests.get(f"{self.url}/eyes/set_colors?main=FF0F0F")
            requests.get(f"{self.url}/eyes/anim_confused")
            time.sleep(2)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_mood?mood=NEUTRAL")
        self.__run_async(func)

    def emote_tired(self, delay: float = 0.0) -> None:
        def func():
            time.sleep(delay)
            requests.get(f"{self.url}/eyes/set_mood?mood=TIRED")
            requests.get(f"{self.url}/eyes/set_colors?main=AAAAAA")
            time.sleep(2)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_mood?mood=NEUTRAL")
        self.__run_async(func)

    def emote_doze(self, delay: float = 0.0) -> None:
        def func():
            time.sleep(delay)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_height?left=80&right=80")
            requests.get(f"{self.url}/eyes/open?left=true&right=true")
            requests.get(f"{self.url}/eyes/set_idlemode?on=false")
            requests.get(f"{self.url}/eyes/set_autoblinker?on=true&interval=10&variation=5")
            time.sleep(2.0)
            requests.get(f"{self.url}/eyes/set_position?position=CENTER")
            requests.get(f"{self.url}/eyes/set_height?left=10&right=10")
        self.__run_async(func)

    def emote_idle(self, delay: float = 0.0) -> None:
        def func():
            time.sleep(delay)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_height?left=120&right=120")
            requests.get(f"{self.url}/eyes/set_width?left=90&right=90")
            requests.get(f"{self.url}/eyes/open?left=true&right=true")
            requests.get(f"{self.url}/eyes/set_idlemode?on=true")
        self.__run_async(func)

    def emote_attention(self, delay: float = 0.0) -> None:
        def func():
            time.sleep(delay)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_height?left=150&right=150")
            requests.get(f"{self.url}/eyes/set_width?left=95&right=95")
            requests.get(f"{self.url}/eyes/set_idlemode?on=false")
            requests.get(f"{self.url}/eyes/set_position?position=CENTER")
            requests.get(f"{self.url}/eyes/open?left=true&right=true")
            requests.get(f"{self.url}/eyes/set_autoblinker?on=true")
        self.__run_async(func)

    def turn_left(self) -> None:
        def func():
            requests.get(f"{self.url}/motors/set_speed?left=-0.8&right=0.8")
            time.sleep(0.2)
            requests.get(f"{self.url}/motors/set_speed?left=0&right=0")
        self.__run_async(func)

    def turn_right(self) -> None:
        def func():
            requests.get(f"{self.url}/motors/set_speed?left=0.8&right=-0.8")
            time.sleep(0.2)
            requests.get(f"{self.url}/motors/set_speed?left=0&right=0")
        self.__run_async(func)

    def take_image(self, width=640, height=480) -> str:
        bytes = httpx.get(f"{self.url}/cam/still?width={width}&height={height}").content
        # with open(f"{time.time()}.jpg", "wb") as file:
        #    file.write(bytes)
        return base64.b64encode(bytes).decode("utf-8")

    def open_camera_stream(self, width: int = 800, height: int = 600) -> CameraStream:
        class HttpCameraStream(CameraStream):
            def __init__(self, url: str):
                self._url = url
                logger.debug(f"creating video stream: {self._url}")
                self._cam: cv2.VideoCapture = cv2.VideoCapture(url)

            def __exit__(self, type, value, traceback):
                logger.debug(f"releasing video stream: {self._url}")
                try:
                    self._cam.release()
                except Exception as error:
                    logger.error(error)

            def get_camera_matrix(self, w: int, h: int) -> np.ndarray:
                # Intrinsic camera matrix
                K = np.array([[509.03709729, 0, 487.01091754],
                              [0, 510.47051325, 410.00552861],
                              [0, 0, 1]], dtype=np.float32)
                scale_x = w / 1024
                scale_y = h / 768
                K[0, 0] *= scale_x  # Scale fx
                K[0, 2] *= scale_x  # Scale cx
                K[1, 1] *= scale_y  # Scale fy
                K[1, 2] *= scale_y  # Scale cy
                return K

            def undistort(self, frame: np.ndarray) -> np.ndarray:
                h, w = frame.shape[:2]
                K = self.get_camera_matrix(w, h)
                D = np.array([-3.28286122e-01, 1.32135416e-01,
                              -1.09980142e-03, -1.49094943e-04,
                              -2.59619213e-02], dtype=np.float32)
                opt_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.2, (w, h))
                return cv2.undistort(frame, K, D, None, opt_K)

            def read(self) -> Optional[np.ndarray]:
                if self._cam.isOpened():
                    ret, frame = self._cam.read()
                    if ret:
                        return self.undistort(frame)
                    logger.error(f"Capturing video image failed.")
                else:
                    logger.error("Failed to open video stream.")
                return None

        return HttpCameraStream(f"{self.url}/cam/live?width={width}&height={height}")

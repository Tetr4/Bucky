import base64
import httpx
import threading
import queue
import requests
import time
from abc import ABC, abstractmethod

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
    def take_image(self, width=640, height=480) -> str:
        raise NotImplementedError()

class FakeBot(IRobot):
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

    def take_image(self, width=640, height=480) -> str:
        try:
            import cv2        
            cam = cv2.VideoCapture(0)
            try:
                if cam.isOpened():
                    name = cam.getBackendName()
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    ret, frame = cam.read()
                    if ret:
                        _, jpeg = cv2.imencode('.jpg', frame)
                        jpeg_bytes = jpeg.tobytes()
                        print(f"{name}: Captured camera image size {len(jpeg_bytes)} bytes.")
                        # with open("assets/images/captured_image.jpg", "wb") as file:
                        #     file.write(jpeg_bytes)
                        return base64.b64encode(jpeg_bytes).decode("utf-8")
                    print("{name}: Capturing camera image failed.")
                else:
                    print("No camera found.")    
            finally:
                cam.release()       
        except ImportError as ex:
            print(str(ex))
        
        default_img_path = "assets/images/horse.jpg"
        with open(default_img_path, "rb") as image_file:
            print(default_img_path)
            return base64.b64encode(image_file.read()).decode("utf-8")

class BuckyBot(IRobot):
    def __init__(self, url:str):
        self.url = url
        self.job_queue = queue.Queue()

        def thread_proc():
            while func := self.job_queue.get():                
                try:
                    func()
                except Exception as ex:
                    print("ERROR", str(ex))

        self.thread = threading.Thread(target=thread_proc, daemon=True)
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

    def take_image(self, width=640, height=480) -> str:
        bytes = httpx.get(f"{self.url}/cam/still?width={width}&height={height}").content
        # with open(f"{time.time()}.jpg", "wb") as file:
        #    file.write(bytes)
        return base64.b64encode(bytes).decode("utf-8")

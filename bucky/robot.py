import base64
import httpx
import threading
import queue
import requests
import time
from abc import ABC, abstractmethod

class IRobot(ABC):
    @abstractmethod
    def emote_happy(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_angry(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_tired(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_sleeping(self, blocking=True) -> None:
        raise NotImplementedError()

    @abstractmethod
    def emote_attention(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def take_image(self, width=640, height=480) -> str:
        raise NotImplementedError()

class FakeBot(IRobot):
    def emote_happy(self) -> None:
        print("😀")

    def emote_angry(self) -> None:
        print("😡")

    def emote_tired(self) -> None:
        print("😩")

    def emote_sleeping(self, blocking=True) -> None:
        pass

    def emote_attention(self) -> None:
        pass

    def take_image(self, width=640, height=480) -> str:
        return "" # TODO take webcam image

class BuckyBot(IRobot):
    def __init__(self, url:str):
        self.url = url
        self.job_queue = queue.Queue()

        def thread_proc():
            while True:
                func = self.job_queue.get()
                try:
                    func()
                except Exception as ex:
                    print("ERROR", str(ex))

        self.thread = threading.Thread(target=thread_proc, daemon=True)
        self.thread.start()

    def __run_async(self, func) -> None:
        self.job_queue.put(func)

    def emote_happy(self) -> None:
        def func():
            requests.get(f"{self.url}/eyes/set_mood?mood=HAPPY")
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/anim_laugh")
            time.sleep(2)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_mood?mood=NEUTRAL")
        self.__run_async(func)

    def emote_angry(self) -> None:
        def func():
            requests.get(f"{self.url}/eyes/set_mood?mood=ANGRY")
            requests.get(f"{self.url}/eyes/set_colors?main=FF0000")
            requests.get(f"{self.url}/eyes/anim_confused")
            time.sleep(2)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_mood?mood=NEUTRAL")
        self.__run_async(func)

    def emote_tired(self) -> None:
        def func():
            requests.get(f"{self.url}/eyes/set_mood?mood=TIRED")
            requests.get(f"{self.url}/eyes/set_colors?main=0000FF")
            time.sleep(2)
            requests.get(f"{self.url}/eyes/set_colors?main=FFFFFF")
            requests.get(f"{self.url}/eyes/set_mood?mood=NEUTRAL")
        self.__run_async(func)

    def emote_sleeping(self, blocking: bool = False) -> None:
        def func():
            requests.get(f"{self.url}/eyes/set_autoblinker?on=false")
            requests.get(f"{self.url}/eyes/close?left=true&right=true")
        if blocking:
            func()
        else:
            self.__run_async(func)

    def emote_attention(self) -> None:
        def func():
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

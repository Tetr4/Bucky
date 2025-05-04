import time
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from bucky.fx_player import FxPlayer
import threading


class TimerToolInput(BaseModel):
    duration: str = Field(description='The duration for the countdown. e.g. 5')
    unit: str = Field(
        description='The unit of the duration. Only Seconds, Minutes or Hours are valid. Must always be in english. ')


class TimerTool(BaseTool):
    name: str = "countdown"
    description: str = "Use this to set a countdown that plays an alarm after the duration expired."
    fx_player: FxPlayer

    def __init__(self, fx_player: FxPlayer):
        super().__init__(fx_player=fx_player, args_schema=TimerToolInput)

    def _run(self, duration: str, unit: str) -> str:
        sleep_duration: float = float(duration)
        if unit.lower().startswith("h"):
            sleep_duration *= 3600
        elif unit.lower().startswith("m"):
            sleep_duration *= 60
        elif unit.lower().startswith("s"):
            sleep_duration *= 1
        else:
            raise ValueError("invalid unit")

        def func():
            alarm_time = time.time() + sleep_duration
            while True:
                eta = alarm_time - time.time()
                print("Alarm in", round(max(0.0, eta)), "seconds ...")
                time.sleep(min(5.0 if eta > 10 else 1.0, eta))
                if time.time() >= alarm_time:
                    print("Alarm!")
                    assert self.fx_player
                    self.fx_player.play_timer_alarm()
                    break

        threading.Thread(target=func, daemon=True).start()

        return "timer created"

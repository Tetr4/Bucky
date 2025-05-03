import time
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from bucky.fx_player import FxPlayer
import threading


class TimerToolInput(BaseModel):
    duration: str = Field(description='The duration for the timer. e.g. 5')
    unit: str = Field(
        description='The unit of the duration. Only Seconds, Minutes or Hours are valid. Must always be in english. ')


class TimerTool(BaseTool):
    name: str = "timer"
    description: str = "Use this set a timer that plays an alarm after duration expired."
    args_schema: Type[BaseModel] = TimerToolInput  # type: ignore
    fx_player: Optional[FxPlayer] = None

    def __init__(self, fx_player: FxPlayer):
        super().__init__()
        self.fx_player = fx_player

    def _run(self, duration: str, unit: str) -> str:
        assert self.fx_player

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
                time.sleep(min(5.0, eta))
                if time.time() >= alarm_time:
                    print("Alarm!")
                    assert self.fx_player
                    self.fx_player.play_timer_alarm()
                    break

        threading.Thread(target=func, daemon=True).start()

        return "timer created"

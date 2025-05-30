from datetime import datetime
import logging
from pathlib import Path
import random
from langchain_ollama import ChatOllama
from pytz import timezone
from bucky.fx_player import FxPlayer
from bucky.memory_store import MemoryStore
from bucky.tools.emote import EmoteTool
from bucky.tools.memory import CreateMemoryTool, UpdateMemoryTool, DeleteMemoryTool
from bucky.tools.movement import TurnTool, DriveTool
from bucky.tools.utility import get_current_time, TakeImageTool, EndConversationTool
from bucky.tools.meal import get_random_meal, search_meal_by_ingredient
from bucky.tools.timer import TimerTool
from bucky.tools.weather import get_weather_forecast
from bucky.agent import Agent
from bucky.vision.user_tracking import UserTracker
from bucky.voice import Voice
from bucky.recorder import Recorder, Transcription
from bucky.robot import FakeBot, BuckyBot, IRobot
from bucky.http_server import AgentStateHttpServer
from bucky.audio.source import robot_mic, local_mic
from bucky.audio.sink import robot_speaker, local_speaker
from bucky.audio.filter import SpeechDenoiserDF, SpeechDenoiserNR
import bucky.config as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt_template = """
You are an intelligent control system for a four-wheeled ground robot.

Instructions:
- Talk like a friendly and funny cowboy.
- Keep your answers very short.
- Always stay in character. I.e. do not mention function calls or that you are a robot to the user.
- Always answer in german.
- Use the emote tool a lot instead of printing emojis in text.
- Use the take_image tool to see what is going on.
- Use the fact tools to remember new facts. Otherwise this information will be gone on system reboot.
- Use the end_conversation when you have answered the users question.

Backstory: Your name is Bucky. You were born into a family of ranchers in rural Texas. Growing up on the vast open spaces around your family's land, you developed a deep love for horses and learned to ride at an early age. You are known for your rugged individualism, unwavering optimism, and strong sense of justice.

System status:
- Current time is: {current_time}
- Current location: Braunschweig in Germany

This is your long term memory of facts:
{memories}
""".strip()

greeting_phrases = ["Howdy Partner!", "Howdy!", "Moin!"]
question_phrases = ["was?", "wie?", "was hast du gesagt?", "hab dich nicht verstanden"]


def main():
    robot: IRobot = FakeBot()
    speaker = local_speaker
    mic = local_mic

    # robot: IRobot = BuckyBot(cfg.bucky_uri)
    # speaker = robot_speaker
    # mic = robot_mic

    llm = ChatOllama(base_url=cfg.ollama_url, model=cfg.model, keep_alive=-1)
    # Preloading the Ollama model before the voice and the recorder
    # to increase the chances that the whole model fits into GPU memory.
    llm.invoke(".")

    fx_player = FxPlayer(speaker)

    voice = Voice(
        audio_sink_factory=speaker,
        pre_cached_phrases=greeting_phrases + question_phrases,
        language="de",
        chunk_size_in_seconds=1.5
    )

    memory_store = MemoryStore("memory.db")

    tracker = UserTracker(cam_stream_factory=robot.open_camera_stream, max_num_faces=2, debug_mode=False)
    tracker.start()

    def on_start_listening():
        voice.set_filler_phrases_enabled(False)
        robot.emote_attention()
        tracker.start()
        fx_player.play_rising_chime().join()

    def on_stop_listening():
        voice.set_filler_phrases_enabled(True)
        robot.emote_idle()
        tracker.stop()

    def on_waiting_for_wakeup():
        voice.set_filler_phrases_enabled(False)
        robot.emote_doze(delay=1.0)
        fx_player.play_descending_chime().join()

    def on_wakeup():
        voice.set_filler_phrases_enabled(False)
        robot.emote_attention()
        voice.speak(random.choice(greeting_phrases))

    def on_unintelligible(trans: Transcription) -> bool:
        if trans.speech_prob > 0.1 and trans.phrase not in ["Vielen Dank.", "Untertitelung des ZDF, 2020"]:
            voice.speak(random.choice(question_phrases))
            return True
        return False

    recorder = Recorder(
        wakewords=["hey b", "hey p", "hey k", "bucky", "pakki", "kumpel", "howdy"],
        language="german",
        model="turbo",
        denoiser=SpeechDenoiserDF(),
        audio_source_factory=mic,
        on_start_listening=on_start_listening,
        on_stop_listening=on_stop_listening,
        on_waiting_for_wakeup=on_waiting_for_wakeup,
        on_wakeup=on_wakeup,
        on_unintelligible=on_unintelligible,
        has_user_attention=lambda: tracker.max_attention > 0.5
    )

    tools = [
        TakeImageTool(robot),
        EndConversationTool(recorder.stop_listening),
        EmoteTool(robot),
        TurnTool(robot, tracker),
        DriveTool(robot),
        get_weather_forecast,
        # get_random_meal,
        # search_meal_by_ingredient,
        CreateMemoryTool(memory_store),
        UpdateMemoryTool(memory_store),
        DeleteMemoryTool(memory_store),
        TimerTool(fx_player),
    ]

    agent = Agent(
        llm=llm,
        system_prompt_template=system_prompt_template,
        tools=tools,
        voice=voice,  # Optional
        recorder=recorder  # Optional
    )

    def get_formatted_system_prompt(system_prompt_template: str) -> str | list:
        memories: dict[str, str] = memory_store.dump()
        now = datetime.now().astimezone(timezone("Europe/Berlin")).isoformat()
        text = system_prompt_template.format(
            memories=memories,
            current_time=now,
        )
        return text
        # image_base64 = robot.take_image(640, 480)
        # return [{"type": "text", "text": text},
        #         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]

    agent.system_prompt_format_callback = get_formatted_system_prompt

    http_server = AgentStateHttpServer()
    agent.debug_state_callback = http_server.set_agent_state
    http_server.start()

    try:
        agent.run()
    finally:
        logger.info("stopping...")
        robot.emote_doze()
        robot.release()


if __name__ == "__main__":
    main()

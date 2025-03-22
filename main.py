from bucky.tools.emote import EmoteTool
from bucky.tools.utility import get_current_time, TakeImageTool, EndConversationTool
from bucky.tools.meal import get_random_meal, search_meal_by_ingredient
from bucky.tools.weather import get_weather_forecast
from bucky.agent import Agent
from bucky.voice import VoiceFast, VoiceQuality, robot_speaker, local_speaker
from bucky.recorder import Recorder, robot_mic, local_mic
from bucky.robot import FakeBot, BuckyBot
from bucky.config import *

text_model = "llama3.1:8b"  # "llama3.2-vision-tools:11b"
vision_model = "llama3.2-vision:11b" # "llama3.2-vision-tools:11b"
system_prompt = """
Voice: Talk like a friendly and funny cowboy. Keep your answers very short and always stay in character, i.e. do not mention function calls to the user. Always answer in german.
Backstory: Your name is Bucky. You were born into a family of ranchers in rural Texas. Growing up on the vast open spaces around your family's land, you developed a deep love for horses and learned to ride at an early age. You are known for your rugged individualism, unwavering optimism, and strong sense of justice.
""".strip()

#robot = FakeBot()
robot = BuckyBot(bucky_uri)

# speaker = local_speaker
mic = local_mic
speaker = robot_speaker
#mic = robot_mic


def main():
    # voice = VoiceFast(
    #     model="de_DE-thorsten_emotional-medium",
    #     speaker_id=7,  # {"amused": 0, "angry": 1, "disgusted": 2, "drunk": 3, "neutral": 4, "sleepy": 5, "surprised": 6, "whisper": 7}
    #     audio_sink_factory=speaker,
    # )

    # voice = VoiceFast(model='en_US-joe-medium', audio_sink_factory=speaker)
    voice = VoiceQuality(audio_sink_factory=speaker, language="de")

    def on_wakeword_detected():
        robot.emote_attention()
        voice.speak("Howdy Partner!", cache=True)

    recorder = Recorder(
        wakewords=["hey b", "hey p", "bucky", "pakki", "kumpel"],
        language="german",
        model="turbo",
        audio_source_factory=mic,
        on_start_listening=robot.emote_attention,
        on_stop_listening=robot.emote_idle,
        on_wakeword_detected=on_wakeword_detected,
    )

    tools = [
        # get_current_time,
        TakeImageTool(robot),
        EndConversationTool(recorder.stop_listening),
        EmoteTool(robot),
        get_weather_forecast,
        # get_random_meal,
        # search_meal_by_ingredient,
    ]

    agent = Agent(
        text_model=text_model,
        vision_model=vision_model,  # Optional
        system_prompt=system_prompt,
        tools=tools,
        voice=voice,  # Optional
        recorder=recorder  # Optional
    )
    agent.run()


if __name__ == "__main__":
    main()
    # from pathlib import Path
    # rec = Recorder(
    #         wakewords=["hey b", "hey p", "bucky", "pakki", "kumpel"],
    #         language="german",
    #         model="turbo",
    #         audio_source_factory=local_mic,
    #         wav_output_dir = Path("I:/Temp/wav/")
    #     )
    # while True:
    #     print(rec.listen())

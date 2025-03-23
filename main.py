from bucky.fx_player import FxPlayer
from bucky.tools.emote import EmoteTool
from bucky.tools.utility import get_current_time, TakeImageTool, EndConversationTool
from bucky.tools.meal import get_random_meal, search_meal_by_ingredient
from bucky.tools.weather import get_weather_forecast
from bucky.agent import Agent
from bucky.voice import Voice, VoiceFast, VoiceQuality, robot_speaker, local_speaker
from bucky.recorder import Recorder, robot_mic, local_mic
from bucky.robot import FakeBot, BuckyBot, IRobot
from bucky.config import *

text_model = "PetrosStav/gemma3-tools:4b" # "llama3.2-vision-tools:11b" # "llama3.1:8b"  # "llama3.2-vision-tools:11b"
vision_model = "PetrosStav/gemma3-tools:4b" # "llama3.2-vision-tools:11b" # "llama3.2-vision:11b"
system_prompt = """
Voice: Talk like a friendly and funny cowboy. Keep your answers very short and always stay in character, i.e. do not mention function calls to the user. Always answer in german.
Backstory: Your name is Bucky. You were born into a family of ranchers in rural Texas. Growing up on the vast open spaces around your family's land, you developed a deep love for horses and learned to ride at an early age. You are known for your rugged individualism, unwavering optimism, and strong sense of justice.
Important! Always answer in German!
""".strip()

robot: IRobot = FakeBot()
#robot: IRobot = BuckyBot(bucky_uri)

speaker = local_speaker
mic = local_mic
#speaker = robot_speaker
#mic = robot_mic


def main():
    fx_player = FxPlayer(speaker)

    # voice = VoiceFast(model='en_US-joe-medium', audio_sink_factory=speaker)
    voice: Voice = VoiceQuality(audio_sink_factory=speaker, pre_cached_phrases=["Howdy Partner!"], language="de")

    def on_start_listening():
        voice.set_filler_phrases_enabled(False)
        robot.emote_attention()
        fx_player.play_rising_chime()

    def on_stop_listening():
        voice.set_filler_phrases_enabled(True)
        robot.emote_idle()

    def on_waiting_for_wakewords():
        voice.set_filler_phrases_enabled(False)
        fx_player.play_descending_chime()       
        robot.emote_doze(delay=1.0)

    def on_wakeword_detected():
        voice.set_filler_phrases_enabled(False)
        robot.emote_attention()
        voice.speak("Howdy Partner!", cache=True)


    recorder = Recorder(
        wakewords=["hey b", "hey p", "bucky", "pakki", "kumpel", "howdy"],
        language="german",
        model="turbo",
        audio_source_factory=mic,
        on_start_listening=on_start_listening,
        on_stop_listening=on_stop_listening,
        on_waiting_for_wakewords=on_waiting_for_wakewords,
        on_wakeword_detected=on_wakeword_detected,
    )

    tools = [
        get_current_time,
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
    
    try:
        agent.run()
    finally:
        robot.emote_doze()
        robot.release()


if __name__ == "__main__":
    main()
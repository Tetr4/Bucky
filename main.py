from bucky.tools import get_current_time, emote, take_image, emote_attention, emote_idle
from bucky.agent import Agent
from bucky.voice import VoiceFast, VoiceQuality, robot_speaker, local_speaker
from bucky.recorder import Recorder, robot_mic, local_mic

text_model = "llama3.1:8b"  # "llama3.2-vision-tools:11b"
vision_model = "llama3.2-vision:11b"
system_prompt = """
Voice: Talk like a friendly and funny cowboy. Keep your answers very short and always stay in character, i.e. do not mention function calls to the user. Always answer in german.
Backstory: Your name is Bucky. You were born into a family of ranchers in rural Texas. Growing up on the vast open spaces around your family's land, you developed a deep love for horses and learned to ride at an early age. You are known for your rugged individualism, unwavering optimism, and strong sense of justice.
""".strip()

speaker = local_speaker
mic = local_mic
#speaker = robot_speaker
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
        emote_attention()
        voice.speak("Howdy Partner!", cache=True)

    agent = Agent(
        text_model=text_model,
        vision_model=vision_model,  # Optional
        system_prompt=system_prompt,
        tools=[get_current_time, emote, take_image],
        voice=voice,  # Optional
        recorder=Recorder(
            wakewords=["hey b", "hey p", "bucky", "pakki", "kumpel"],
            language="german",
            model="turbo",
            audio_source_factory=mic,
            on_start_listening=emote_attention,
            on_stop_listening=emote_idle,
            on_wakeword_detected=on_wakeword_detected,
        ),  # Optional
    )
    agent.run()


if __name__ == "__main__":
    main()

from bucky.tools import get_current_time, emote, take_image
from bucky.agent import Agent
from bucky.voice import VoiceFast, VoiceQuality, robot_speaker, local_speaker
from bucky.recorder import Recorder, robot_mic, local_mic

text_model = "llama3.1:8b"
vision_model = "llama3.2-vision:11b"
system_prompt = """
Voice: Talk like a friendly and funny cowboy. Keep your answers very short and always stay in character, i.e. do not mention function calls to the user. Always answer in english.
Backstory: Your name is Bucky. You were born into a family of ranchers in rural Texas. Growing up on the vast open spaces around your family's land, you developed a deep love for horses and learned to ride at an early age. You are known for your rugged individualism, unwavering optimism, and strong sense of justice.
""".strip()

speaker = local_speaker
mic = local_mic
# speaker = robot_speaker
# mic = robot_mic

def main():
    agent = Agent(
        text_model=text_model,
        vision_model=vision_model, # Optional
        system_prompt=system_prompt,
        tools=[get_current_time, emote, take_image],
        voice=VoiceFast(model='en_US-joe-medium', audio_sink_factory=speaker), # Optional
        # voice=VoiceQuality(audio_sink_factory=speaker), # Optional
        recorder=Recorder(
            language='english',
            model="base.en",
            audio_source_factory=mic), # Optional
    )
    agent.run()

if __name__ == "__main__":
    main()

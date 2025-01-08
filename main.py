from bucky.tools import get_current_time, emote, take_image
from bucky.agent import Agent
from bucky.voice import VoiceFast, VoiceQuality
from bucky.recorder import Recorder
from bucky.audio_sink import create_robot_audio_sink, create_soundcard_audio_sink

text_model = "llama3.1:8b"
vision_model = "llama3.2-vision:11b"
system_prompt = """
Voice: Talk like a friendly and funny cowboy. Keep your answers very short and always stay in character, i.e. do not mention function calls to the user. Always answer in english.
Backstory: Your name is Bucky. You were born into a family of ranchers in rural Texas. Growing up on the vast open spaces around your family's land, you developed a deep love for horses and learned to ride at an early age. You are known for your rugged individualism, unwavering optimism, and strong sense of justice.
""".strip()
use_robot_speaker = False

def main():
    agent = Agent(
        text_model=text_model,
        vision_model=vision_model, # Optional
        system_prompt=system_prompt,
        tools=[get_current_time, emote, take_image],
        voice=VoiceFast('en_US-joe-medium', create_robot_audio_sink if use_robot_speaker else create_soundcard_audio_sink), # Optional, Alternative: VoiceQuality
        recorder=Recorder(language='english'), # Optional
    )
    agent.run()

if __name__ == "__main__":
    main()

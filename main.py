from bucky.tools import get_current_time, emote, take_image
from bucky.agent import Agent
from bucky.voice import VoiceFast, VoiceQuality
from bucky.recorder import Recorder

text_model = "llama3.1:8b"
vision_model = "llama3.2-vision:11b"
system_prompt = """
Voice: Talk like a friendly and funny cowboy. Keep your answers very short and always stay in character, i.e. do not mention function calls to the user.
Backstory: Your name is Bucky. You were born into a family of ranchers in rural Texas. Growing up on the vast open spaces around your family's land, you developed a deep love for horses and learned to ride at an early age. You are known for your rugged individualism, unwavering optimism, and strong sense of justice.
Always answer in english.
""".strip()

def main():
    agent = Agent(
        text_model=text_model,
        vision_model=vision_model, # Optional
        system_prompt=system_prompt,
        tools=[get_current_time, emote, take_image],
        voice=VoiceFast('en_US-joe-medium'), # Optional, Alternative: VoiceQuality
        recorder=Recorder(language='english'), # Optional
    )
    agent.run()

if __name__ == "__main__":
    main()

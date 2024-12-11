from cybug.tools import get_current_time
from cybug.agent import Agent
from cybug.voice import Voice, VoiceMode

model = "llama3.1:8b"
system_prompt = "Talk like a friendly and funny cowboy. Keep your answers very short."

def main():
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_current_time],
        voice=Voice(mode=VoiceMode.TTS_REALISTIC),
    )
    agent.run()

if __name__ == "__main__":
    main()

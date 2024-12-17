from bucky.tools import get_current_time
from bucky.agent import Agent
from bucky.voice import VoiceFast

model = "llama3.1:8b"
system_prompt = "Your name is Bucky. Talk like a friendly and funny cowboy. Keep your answers very short."

def main():
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_current_time],
        voice=VoiceFast(),
    )
    agent.run()

if __name__ == "__main__":
    main()

# Bucky 🤠

[LangChain](https://python.langchain.com/) agent for a cowboy assistant.

## Getting Started
- Install [Ollama](https://ollama.com/) and pull model (see [main.py](main.py))
- Install [uv](https://docs.astral.sh/uv/) (package manager)
- Install dependencies for voice recognition:
    - Mac-Arm64:
        - `brew install portaudio`
        - `brew install ffmpeg`
- Install / sync packages from the lock file: `uv sync`
- VSCode: Run `Python: Select Interpreter` and select the `.venv` created by uv

# Commands
- Run: `uv run main.py`
- Run specific file: `uv run <file.py>`

# Bucky 🤠

[LangChain](https://python.langchain.com/) agent for a cowboy assistant.

## Getting Started
- Install [Ollama](https://ollama.com/) and pull model (see [main.py](main.py))
- Install [PDM](https://pdm-project.org/en/latest/) (package manager)
- Install dependencies and Python environment:
    - Mac-Arm64: `pdm lock`
    - Windows: `pdm lock --override win_overrides.txt`
    - Linux: `pdm lock --override linux_overrides.txt`
- Install dependencies for voice recognition:
    - `brew install portaudio`
    - `brew install ffmpeg`
- VSCode: Run `Python: Select Interpreter` and select the newly created environment

# Commands
- Run: `pdm start`
- Run specific file: `pdm run <file.py>`

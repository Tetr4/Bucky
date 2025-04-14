# Bucky 🤠

[LangChain](https://python.langchain.com/) agent for a cowboy assistant.

## Getting Started
- Install [Ollama](https://ollama.com/) and pull model (see [main.py](main.py))
- Install [PDM](https://pdm-project.org/en/latest/) (package manager)
- Install dependencies for voice recognition:
    - Mac-Arm64: 
        - `brew install portaudio`
        - `brew install ffmpeg`
- Optional: Create or update the lock file:
    - Mac-Arm64: `pdm lock`
    - Windows: `pdm lock --override win_overrides.txt --lockfile pdm_win32.lock`
    - Linux: `pdm lock --override linux_overrides.txt --lockfile pdm_linux.lock`
- Install packages from existing lock file:
    - Mac-Arm64: `pdm sync --dev --clean`
    - Windows: `pdm sync --lockfile pdm_win32.lock --dev --clean`
    - Linux: `pdm sync --lockfile pdm_linux.lock --dev --clean`
- To add new packages to the project:
    - Mac-Arm64: `pdm add <mypackage>`
    - Windows: `pdm add <mypackage> --override win_overrides.txt --lockfile pdm_win32.lock`
    - Linux: `pdm add <mypackage> --override linux_overrides.txt --lockfile pdm_linux.lock`
- VSCode: Run `Python: Select Interpreter` and select the newly created environment

# Commands
- Run: `pdm start`
- Run specific file: `pdm run <file.py>`

# Socially Intelligent Humanoid Robot — AI Brain

## Purpose
A terminal-based conversational AI module for our humanoid robot project.
It uses a local Ollama model and keeps conversation history during the session.

## Requirements
- Python 3
- Ollama installed and running
- The `llama3.2:3b` model downloaded

## Run on macOS
From the project directory:

```bash
source .venv/bin/activate
python robot_brain.py
```

Type `exit` or `quit` to stop the conversation.

## API
The program connects to `http://localhost:11434/api/chat`.
No third-party Python packages or API keys are required.

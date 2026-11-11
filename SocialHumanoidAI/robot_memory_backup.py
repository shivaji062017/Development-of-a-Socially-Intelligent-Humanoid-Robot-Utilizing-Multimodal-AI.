
import json
from pathlib import Path

MEMORY_FILE = Path(__file__).resolve().parent / "robot_memory.json"


def load_memory():
    """Load saved robot memory from disk."""
    if not MEMORY_FILE.exists():
        return []

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (OSError, json.JSONDecodeError):
        return []


def save_memory(messages):
    """Save the latest conversation messages."""
    try:
        with MEMORY_FILE.open("w", encoding="utf-8") as file:
            json.dump(messages, file, ensure_ascii=False, indent=2)
        return True

    except OSError as error:
        print("Memory save error:", error)
        return False


def clear_memory():
    """Clear saved conversation memory."""
    save_memory([])

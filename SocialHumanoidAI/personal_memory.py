
import json
from pathlib import Path

FACTS_FILE = Path(__file__).resolve().parent / "personal_facts.json"


def load_facts():
    """Load saved personal facts."""
    if not FACTS_FILE.exists():
        return {}

    try:
        with FACTS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_fact(key, value):
    """Save or update one personal fact."""
    key = key.strip().lower()
    value = value.strip()

    if not key or not value:
        return False

    facts = load_facts()
    facts[key] = value

    try:
        with FACTS_FILE.open("w", encoding="utf-8") as file:
            json.dump(facts, file, ensure_ascii=False, indent=2)
        return True

    except OSError as error:
        print("Personal memory save error:", error)
        return False


def get_fact(key):
    """Retrieve one saved personal fact."""
    return load_facts().get(key.strip().lower())


def delete_fact(key):
    """Delete one saved personal fact."""
    facts = load_facts()
    key = key.strip().lower()

    if key not in facts:
        return False

    del facts[key]

    try:
        with FACTS_FILE.open("w", encoding="utf-8") as file:
            json.dump(facts, file, ensure_ascii=False, indent=2)
        return True

    except OSError as error:
        print("Personal memory delete error:", error)
        return False

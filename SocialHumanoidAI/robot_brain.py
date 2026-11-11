import json
import urllib.request
import urllib.error
import socket

API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a friendly, honest humanoid robot assistant. "
    "Give clear, helpful answers. Never claim to see or do things "
    "that you cannot actually see or do."
)


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("🤖 Robot Brain is ready!")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRobot: Goodbye, Shiva!")
            break

        if user_text.lower() in ("exit", "quit"):
            print("Robot: Goodbye, Shiva!")
            break

        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        payload = json.dumps({
            "model": MODEL,
            "messages": messages,
            "stream": False
        }).encode("utf-8")

        request = urllib.request.Request(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

            reply = result["message"]["content"]
            messages.append({"role": "assistant", "content": reply})
            print(f"\nRobot: {reply}\n")

        except urllib.error.HTTPError as error:
            messages.pop()
            details = error.read().decode("utf-8", errors="replace")
            print(f"HTTP error {error.code}: {details}")
        except urllib.error.URLError as error:
            messages.pop()
            print(f"Cannot connect to Ollama: {error.reason}")
            print("Check that Ollama is running.")
        except (TimeoutError, socket.timeout):
            messages.pop()
            print("Ollama took too long to respond. Please try again.")
        except (KeyError, json.JSONDecodeError) as error:
            messages.pop()
            print(f"Unexpected Ollama response: {error}")


if __name__ == "__main__":
    main()

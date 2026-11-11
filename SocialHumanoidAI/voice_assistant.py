
import subprocess
import json
import urllib.request
import sounddevice as sd
import mlx_whisper
from scipy.io.wavfile import write

# Ollama local AI configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

# Audio configuration
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_FILE = "live_input.wav"

# Conversation memory: lasts only while this program is running
conversation_history = []

SYSTEM_PROMPT = (
    "You are a friendly, socially intelligent humanoid robot. "
    "Speak English only. Keep answers concise and natural. "
    "Use previous conversation messages to answer follow-up questions."
)


def record_audio():
    """Record five seconds of audio from the microphone."""
    input("\nPress ENTER to start speaking, or type exit to stop: ")
    print(f"Listening for {RECORD_SECONDS} seconds...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )
    sd.wait()

    write(AUDIO_FILE, SAMPLE_RATE, audio)
    print("Recording complete.")


def transcribe_audio():
    """Convert the recorded audio into English text."""
    result = mlx_whisper.transcribe(
        AUDIO_FILE,
        path_or_hf_repo="mlx-community/whisper-tiny-mlx",
        language="en"
    )
    return result["text"].strip()


def ask_robot(user_text):
    """Ask Ollama a question while preserving session conversation history."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_text}
    ]

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": messages
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    answer = data["message"]["content"].strip()

    # Save the exchange only after Ollama responds successfully
    conversation_history.append(
        {"role": "user", "content": user_text}
    )
    conversation_history.append(
        {"role": "assistant", "content": answer}
    )

    return answer


def speak(text):
    """Print the robot's response and speak it aloud on macOS."""
    print(f"\nRobot: {text}")
    subprocess.run(["say", text], check=False)


def main():
    print("=== Social Humanoid Robot: English Voice Mode ===")
    print("Local AI model:", MODEL)
    print("Conversation memory: enabled for this session")
    print("Type 'exit' when prompted to stop.\n")

    while True:
        command = input(
            "Press ENTER to talk, or type exit to stop: "
        )

        if command.strip().lower() == "exit":
            print("Robot shutting down. Goodbye!")
            break

        try:
            print(f"\nListening... Speak for {RECORD_SECONDS} seconds.")

            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16"
            )
            sd.wait()
            write(AUDIO_FILE, SAMPLE_RATE, audio)

            user_text = transcribe_audio()
            print(f"\nYou said: {user_text}")

            if not user_text or len(user_text) < 2:
                print("I could not understand clearly. Please try again.")
                continue

            answer = ask_robot(user_text)
            speak(answer)

        except Exception as error:
            print(f"Error: {error}")
            print("Please check that Ollama is running and try again.")


if __name__ == "__main__":
    main()

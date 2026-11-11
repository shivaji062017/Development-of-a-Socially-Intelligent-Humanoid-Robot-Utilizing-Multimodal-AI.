
import subprocess
import json
import urllib.request
import os
import sounddevice as sd
import numpy as np
import mlx_whisper

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_FILE = "live_input.wav"


def record_audio():
    input("\nPress ENTER, then speak for 5 seconds...")

    print("Listening... Speak now!")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )
    sd.wait()

    from scipy.io.wavfile import write
    write(AUDIO_FILE, SAMPLE_RATE, audio)

    print("Recording complete.")


def transcribe_audio():
    result = mlx_whisper.transcribe(
        AUDIO_FILE,
        path_or_hf_repo="mlx-community/whisper-tiny-mlx",
        language="en"
    )
    return result["text"].strip()


def ask_robot(user_text):
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly socially intelligent humanoid robot. "
                    "Speak English only. Keep answers concise and natural."
                )
            },
            {"role": "user", "content": user_text}
        ]
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data["message"]["content"].strip()


def speak(text):
    print(f"\nRobot: {text}")
    subprocess.run(["say", text], check=False)


def main():
    print("=== Social Humanoid Robot: English Voice Mode ===")
    print("Type 'exit' to stop.")

    while True:
        command = input("\nPress ENTER to talk, or type exit: ")

        if command.strip().lower() == "exit":
            print("Robot shutting down. Goodbye!")
            break

        try:
            print("Listening... Speak for 5 seconds.")
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16"
            )
            sd.wait()

            from scipy.io.wavfile import write
            write(AUDIO_FILE, SAMPLE_RATE, audio)

            user_text = transcribe_audio()
            print(f"\nYou said: {user_text}")

            if not user_text or len(user_text) < 2:
                print("I could not understand clearly. Please try again.")
                continue

            answer = ask_robot(user_text)
            speak(answer)

        except Exception as error:
            print("Error:", error)


if __name__ == "__main__":
    main()

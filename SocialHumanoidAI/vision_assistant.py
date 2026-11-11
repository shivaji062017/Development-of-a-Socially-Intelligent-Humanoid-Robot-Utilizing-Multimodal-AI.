
import json
import os
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import cv2
import sounddevice as sd
from scipy.io.wavfile import write
import mlx_whisper
from PIL import Image, ImageTk

from robot_memory import load_memory, save_memory, clear_memory
from personal_memory import load_facts, save_fact


BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """
You are a friendly, helpful socially intelligent humanoid robot assistant.
Answer naturally and clearly. Keep responses concise and conversational.
You can respond to questions about what the camera can see, but do not
claim to identify a person's real-world identity. Do not claim to perform
physical actions unless they are actually available and confirmed.
"""


class VisionAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("Social Humanoid AI Assistant")
        self.root.geometry("1050x720")
        self.root.minsize(800, 600)

        self.running = True
        self.busy = False
        self.conversation_history = load_memory()[-MAX_HISTORY_MESSAGES:]

        self.camera = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.status_var = tk.StringVar(value="Ready")
        self.user_var = tk.StringVar(value="Your speech will appear here.")
        self.robot_var = tk.StringVar(value="Hello! I am your robot assistant.")

        self.build_ui()

        if not self.camera.isOpened():
            self.status_var.set("Camera unavailable. Check camera permissions.")
        else:
            self.update_camera()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Socially Intelligent Humanoid Robot",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=10)

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        camera_frame = tk.LabelFrame(
            main, text="Live Camera & Face Detection",
            font=("Arial", 12, "bold")
        )
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.camera_label = tk.Label(
            camera_frame,
            text="Starting camera...",
            bg="black",
            fg="white",
            font=("Arial", 12)
        )
        self.camera_label.pack(fill="both", expand=True, padx=6, pady=6)

        chat_frame = tk.LabelFrame(
            main, text="Robot Conversation",
            font=("Arial", 12, "bold")
        )
        chat_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

        tk.Label(
            chat_frame, text="You said:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", padx=10, pady=(12, 3))

        self.user_text = tk.Text(
            chat_frame, height=5, wrap="word",
            font=("Arial", 11), state="disabled"
        )
        self.user_text.pack(fill="x", padx=10)

        tk.Label(
            chat_frame, text="Robot response:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", padx=10, pady=(12, 3))

        self.robot_text = tk.Text(
            chat_frame, height=9, wrap="word",
            font=("Arial", 11), state="disabled"
        )
        self.robot_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.voice_button = tk.Button(
            button_frame,
            text="🎤 Speak to Robot",
            font=("Arial", 12, "bold"),
            command=self.start_voice,
            width=18,
            height=2
        )
        self.voice_button.pack(side="left", padx=6)

        tk.Button(
            button_frame,
            text="Clear Conversation",
            font=("Arial", 11),
            command=self.clear_conversation,
            width=18,
            height=2
        ).pack(side="left", padx=6)

        tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            font=("Arial", 10),
            relief="sunken"
        ).pack(fill="x", side="bottom")

    def update_camera(self):
        if not self.running:
            return

        if self.camera.isOpened():
            success, frame = self.camera.read()

            if success:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40)
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, "Face detected",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)

                width = max(self.camera_label.winfo_width(), 320)
                height = max(self.camera_label.winfo_height(), 240)
                image.thumbnail((width, height))

                photo = ImageTk.PhotoImage(image=image)
                self.camera_label.configure(image=photo, text="")
                self.camera_label.image = photo

        self.root.after(30, self.update_camera)

    def set_status(self, text):
        if self.running:
            self.status_var.set(text)

    def set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def start_voice(self):
        if self.busy:
            return

        self.busy = True
        self.voice_button.configure(state="disabled")
        self.set_status("Recording for 5 seconds... Please speak.")
        threading.Thread(target=self.voice_worker, daemon=True).start()

    def voice_worker(self):
        temp_path = None

        try:
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16"
            )
            sd.wait()

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_file:
                temp_path = temp_file.name

            write(temp_path, SAMPLE_RATE, audio)

            self.root.after(
                0, self.set_status, "Converting speech to text..."
            )

            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo="mlx-community/whisper-tiny-mlx",
                language="en"
            )
            user_text = result.get("text", "").strip()

            if not user_text:
                self.root.after(
                    0, self.set_status,
                    "No speech recognized. Please try again."
                )
                return

            self.root.after(0, self.set_text, self.user_text, user_text)
            self.root.after(0, self.set_status, "Thinking...")

            facts = load_facts()

            if facts:
                facts_text = "\n".join(
                    f"- {key}: {value}" for key, value in facts.items()
                )
            else:
                facts_text = "No personal facts saved yet."

            system_message = (
                SYSTEM_PROMPT
                + "\n\nSaved personal facts about the user:\n"
                + facts_text
                + "\nUse these facts when relevant. Do not invent facts."
            )

            messages = [{"role": "system", "content": system_message}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_text})

            payload = json.dumps({
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False
            }).encode("utf-8")

            request = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

            answer = result.get("message", {}).get(
                "content", "Sorry, I could not generate a response."
            ).strip()


            
            
            # Save a personal fact only when the user explicitly asks.
            import re

            normalized_text = re.sub(r"[,.!?]+", " ", user_text.lower())
            normalized_text = " ".join(normalized_text.split())

            prefixes = (
                "remember that my ",
                "remember my ",
                "remember that ",
                "please remember that my ",
                "please remember my ",
                "please remember that ",
            )

            for prefix in prefixes:
                if normalized_text.startswith(prefix):
                    fact = normalized_text[len(prefix):].strip()

                    if " is " in fact:
                        key, value = re.split(r"\s+is\s+", fact, maxsplit=1)
                        key = re.sub(r"^(that\s+)?my\s+", "", key).strip()
                        value = value.strip(" .,!?:;")

                        if save_fact(key, value):
                            answer += "\nI've saved that in my personal memory."
                        else:
                            answer += "\nSorry, I couldn't save that memory."
                    else:
                        answer += "\nPlease say: Remember that my favourite language is Python."

                    break




            self.conversation_history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer}
            ])
            self.conversation_history = self.conversation_history[
                -MAX_HISTORY_MESSAGES:
            ]

            saved = save_memory(self.conversation_history)

            self.root.after(0, self.set_text, self.robot_text, answer)

            if saved:
                self.root.after(
                    0, self.set_status,
                    "Response ready. Conversation memory saved."
                )
            else:
                self.root.after(
                    0, self.set_status,
                    "Response ready, but memory could not be saved."
                )

            if self.running:
                self.root.after(
                    0, self.set_status, "Speaking response..."
                )
                subprocess.run(
                    ["say", answer],
                    check=False,
                    timeout=120
                )
                self.root.after(0, self.set_status, "Ready")

        except sd.PortAudioError as error:
            self.root.after(
                0, self.set_status, f"Microphone error: {error}"
            )
        except urllib.error.URLError as error:
            self.root.after(
                0, self.set_status,
                f"Cannot connect to Ollama. Is it running? {error}"
            )
        except Exception as error:
            self.root.after(
                0, self.set_status, f"Error: {error}"
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            self.busy = False
            if self.running:
                self.root.after(
                    0, lambda: self.voice_button.configure(state="normal")
                )

    def clear_conversation(self):
        if self.busy:
            messagebox.showinfo(
                "Please wait",
                "Wait until the current conversation finishes."
            )
            return

        if not messagebox.askyesno(
            "Clear Conversation",
            "Clear the conversation from the app and saved memory?"
        ):
            return

        self.conversation_history = []
        clear_memory()
        self.set_text(self.user_text, "")
        self.set_text(self.robot_text, "Conversation cleared. Let's start again.")
        self.set_status("Conversation and saved memory cleared.")

    def close_app(self):
        self.running = False

        if self.camera and self.camera.isOpened():
            self.camera.release()

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VisionAssistant(root)
    root.mainloop()

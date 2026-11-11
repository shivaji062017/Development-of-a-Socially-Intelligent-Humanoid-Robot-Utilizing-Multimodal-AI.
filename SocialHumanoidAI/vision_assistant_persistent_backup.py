
import tkinter as tk
from tkinter import ttk
import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import mlx_whisper
import urllib.request
import urllib.error
import json
import threading
import subprocess
import tempfile
import os
from PIL import Image, ImageTk


class HumanoidRobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Social Humanoid Robot | Local AI")
        self.root.geometry("1100x760")
        self.root.minsize(850, 600)

        self.running = True
        self.busy = False
        self.conversation_history = []

        self.camera = cv2.VideoCapture(0)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.status = tk.StringVar(value="System ready")
        self.face_status = tk.StringVar(value="Searching for face...")
        self.user_text = tk.StringVar(
            value="Your speech will appear here."
        )
        self.robot_text = tk.StringVar(
            value="Hello! Click Start Listening to talk to me."
        )

        self.build_ui()

        if not self.camera.isOpened():
            self.status.set("Camera unavailable")
            self.camera.release()
            self.camera = None

        self.update_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def build_ui(self):
        self.root.configure(bg="#101820")

        header = tk.Frame(self.root, bg="#101820")
        header.pack(fill="x", padx=18, pady=14)

        tk.Label(
            header,
            text="SOCIAL HUMANOID ROBOT",
            font=("Arial", 22, "bold"),
            fg="#65d6ce",
            bg="#101820"
        ).pack(side="left")

        tk.Label(
            header,
            text="LOCAL MULTIMODAL AI",
            font=("Arial", 10, "bold"),
            fg="white",
            bg="#101820"
        ).pack(side="right", padx=8)

        main = tk.Frame(self.root, bg="#101820")
        main.pack(fill="both", expand=True, padx=18, pady=(0, 14))

        camera_panel = tk.Frame(main, bg="#1b2a38")
        camera_panel.pack(
            side="left", fill="both", expand=True, padx=(0, 10)
        )

        tk.Label(
            camera_panel,
            text="LIVE CAMERA",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#1b2a38"
        ).pack(anchor="w", padx=12, pady=10)

        self.camera_label = tk.Label(
            camera_panel,
            bg="#05090d",
            fg="white",
            text="Opening camera...",
            font=("Arial", 14)
        )
        self.camera_label.pack(
            fill="both", expand=True, padx=10, pady=8
        )

        tk.Label(
            camera_panel,
            textvariable=self.face_status,
            font=("Arial", 11, "bold"),
            fg="#65d6ce",
            bg="#1b2a38"
        ).pack(anchor="w", padx=12, pady=10)

        right = tk.Frame(main, bg="#1b2a38", width=370)
        right.pack(side="right", fill="both", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(
            right,
            text="ROBOT STATUS",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#1b2a38"
        ).pack(anchor="w", padx=12, pady=(12, 4))

        tk.Label(
            right,
            textvariable=self.status,
            font=("Arial", 11),
            fg="#65d6ce",
            bg="#1b2a38",
            wraplength=330,
            justify="left"
        ).pack(anchor="w", padx=12, pady=8)

        tk.Label(
            right,
            text="YOU SAID",
            font=("Arial", 10, "bold"),
            fg="#a8b9c9",
            bg="#1b2a38"
        ).pack(anchor="w", padx=12, pady=(14, 4))

        tk.Label(
            right,
            textvariable=self.user_text,
            font=("Arial", 11),
            fg="white",
            bg="#263746",
            wraplength=325,
            justify="left",
            anchor="nw",
            padx=10,
            pady=10,
            height=5
        ).pack(fill="x", padx=12)

        tk.Label(
            right,
            text="AI RESPONSE",
            font=("Arial", 10, "bold"),
            fg="#a8b9c9",
            bg="#1b2a38"
        ).pack(anchor="w", padx=12, pady=(14, 4))

        tk.Label(
            right,
            textvariable=self.robot_text,
            font=("Arial", 11),
            fg="white",
            bg="#263746",
            wraplength=325,
            justify="left",
            anchor="nw",
            padx=10,
            pady=10
        ).pack(fill="both", expand=True, padx=12, pady=(0, 12))

        controls = tk.Frame(self.root, bg="#101820")
        controls.pack(fill="x", padx=18, pady=(0, 18))

        self.listen_button = ttk.Button(
            controls,
            text="🎤 Start Listening",
            command=self.start_listening
        )
        self.listen_button.pack(side="left")

        ttk.Button(
            controls,
            text="Clear Conversation",
            command=self.clear_text
        ).pack(side="left", padx=10)

        ttk.Button(
            controls,
            text="Exit Robot",
            command=self.close
        ).pack(side="right")

    def update_camera(self):
        if not self.running:
            return

        if self.camera is not None:
            ok, frame = self.camera.read()

            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40)
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Face Detected",
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                self.face_status.set(
                    f"Face detected: {len(faces)}"
                    if len(faces)
                    else "No face detected"
                )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image.thumbnail((720, 520))

                photo = ImageTk.PhotoImage(image=image)
                self.camera_label.configure(image=photo, text="")
                self.camera_label.image = photo

        self.root.after(30, self.update_camera)

    def start_listening(self):
        if self.busy:
            return

        self.busy = True
        self.listen_button.configure(state="disabled")
        self.status.set("Listening: speak now...")
        self.user_text.set("Recording audio...")
        self.robot_text.set("Waiting for your message...")

        threading.Thread(
            target=self.voice_worker,
            daemon=True
        ).start()

    def voice_worker(self):
        temp_path = None

        try:
            duration = 5
            sample_rate = 16000

            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16"
            )
            sd.wait()

            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            wav.write(temp_path, sample_rate, audio)

            self.root.after(
                0,
                lambda: self.status.set("Transcribing speech...")
            )

            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo="mlx-community/whisper-tiny-mlx",
                language="en"
            )

            recognized = result.get("text", "").strip()

            if not recognized:
                self.root.after(
                    0,
                    lambda: self.status.set(
                        "No speech recognized. Please try again."
                    )
                )
                return

            self.root.after(
                0,
                lambda text=recognized: self.user_text.set(text)
            )
            self.root.after(
                0,
                lambda: self.status.set("Thinking with local AI...")
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly socially intelligent humanoid "
                        "robot. Reply naturally and concisely. Remember "
                        "details from the current conversation and use them "
                        "when relevant. Do not claim to see or do things "
                        "unsupported by available sensors."
                    )
                }
            ]

            messages.extend(self.conversation_history)
            messages.append({
                "role": "user",
                "content": recognized
            })

            payload = json.dumps({
                "model": "llama3.2:3b",
                "stream": False,
                "messages": messages
            }).encode("utf-8")

            request = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))

            answer = data.get("message", {}).get(
                "content",
                "Sorry, I could not generate a response."
            ).strip()

            # Store both sides of the conversation.
            self.conversation_history.extend([
                {"role": "user", "content": recognized},
                {"role": "assistant", "content": answer}
            ])

            # Keep the latest 10 messages (5 exchanges).
            self.conversation_history = self.conversation_history[-10:]

            self.root.after(
                0,
                lambda text=answer: self.robot_text.set(text)
            )
            self.root.after(
                0,
                lambda: self.status.set("Speaking response...")
            )

            subprocess.run(
                ["say", answer],
                check=False,
                timeout=120
            )

            self.root.after(
                0,
                lambda: self.status.set("System ready")
            )

        except urllib.error.URLError as error:
            message = (
                "Cannot reach Ollama. Check that Ollama is running. "
                + str(error)
            )
            self.root.after(
                0,
                lambda msg=message: self.status.set(msg[:180])
            )
            self.root.after(
                0,
                lambda: self.robot_text.set(
                    "Local AI is unavailable. Start Ollama and try again."
                )
            )

        except Exception as error:
            message = str(error)
            self.root.after(
                0,
                lambda msg=message: self.status.set("Error: " + msg[:180])
            )
            self.root.after(
                0,
                lambda: self.robot_text.set(
                    "Something went wrong. Check the status and terminal."
                )
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            self.busy = False
            self.root.after(
                0,
                lambda: self.listen_button.configure(state="normal")
                if self.running else None
            )

    def clear_text(self):
        self.conversation_history.clear()
        self.user_text.set("Your speech will appear here.")
        self.robot_text.set("Conversation memory cleared.")

    def close(self):
        self.running = False

        if self.camera is not None:
            self.camera.release()

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HumanoidRobotGUI(root)
    root.mainloop()

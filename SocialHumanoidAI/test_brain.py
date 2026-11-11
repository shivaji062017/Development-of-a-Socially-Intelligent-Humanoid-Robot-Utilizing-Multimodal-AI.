import ollama

print("Connecting to the robot's AI brain...")

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are the AI brain of a socially intelligent "
                "humanoid robot. Be friendly and concise."
            ),
        },
        {
            "role": "user",
            "content": "Introduce yourself in two sentences.",
        },
    ],
)

print("\nRobot:", response["message"]["content"])
print("\nAI brain test completed.")

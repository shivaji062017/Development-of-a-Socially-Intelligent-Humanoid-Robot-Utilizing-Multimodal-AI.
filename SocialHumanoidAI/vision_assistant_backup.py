import cv2
import time

camera = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if not camera.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit(1)

print("Robot vision started.")
print("Press Q to close.")

last_face_time = 0

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            cv2.putText(
                frame, "Person detected", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        cv2.imshow("Social Humanoid Robot - Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Robot vision stopped.")

import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO pose model
model = YOLO("C:/Users/saisa/Desktop/AI-Basketball-Referee/yolov8s-pose.pt")

# Open the webcam
cap = cv2.VideoCapture('C:/Users/saisa/Downloads/4962327_close_up_man_basketball_h1895b38823082202Basketball Player4k026720p5000br.mp4')

# Define the body part indices
body_index = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

def calculate_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def check_foul(defender, shooter):
    def_parts = ["left_wrist", "right_wrist"]
    shoot_parts = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]
    for dpart in def_parts:
        for spart in shoot_parts:
            if (
                calculate_distance(defender[dpart], shooter[spart]) < 40
            ):  # arbitrary threshold
                return True
    return False

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Reduce the frame size to half
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        results = model(frame, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Resize annotated_frame to match the resized input
        annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if len(results) >= 2:
            humans = np.round(results[0].keypoints.numpy(), 1)
            if humans.shape[0] >= 2:
                shooter, defender = humans[:2]

                persons = []
                for human in [shooter, defender]:
                    parts = {}
                    for part, index in body_index.items():
                        try:
                            parts[part] = human[index]
                        except:
                            parts[part] = None
                    persons.append(parts)

                if check_foul(persons[1], persons[0]):
                    blue_frame = cv2.addWeighted(
                        frame, 0.7, np.zeros(frame.shape, frame.dtype) + [0, 0, 255], 0.3, 0
                    )
                    cv2.imshow("Foul Detected", blue_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

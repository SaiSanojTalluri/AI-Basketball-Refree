import cv2
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
from playsound import playsound
import tempfile

# Load the YOLO pose model
model = YOLO("yolov8s-pose.pt")

# Open the video file or webcam
cap = cv2.VideoCapture("C:/Users/saisa/Desktop/AI-Basketball-Referee/video2.mp4")

# Define the body part indices
body_index = {"left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}

# Initialize step count, previous positions, and thresholds
step_count = 0
prev_left_ankle_y = None
prev_right_ankle_y = None
step_threshold = 12
min_wait_frames = 8
wait_frames = 0

# Generate the 'Step' audio file once (optional use)
tts = gTTS(text="Step", lang="en")
temp_file = tempfile.NamedTemporaryFile(delete=False)
tts.save(temp_file.name)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run pose detection
    results = model(frame, verbose=False, conf=0.5)
    annotated_frame = results[0].plot()

    # Dynamically resize to 70%
    frame_height, frame_width = frame.shape[:2]
    scale_factor = 0.7
    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)

    annotated_frame_resized = cv2.resize(annotated_frame, (new_width, new_height))

    # Show the annotated and resized frame
    cv2.imshow("YOLOv8 Inference", annotated_frame_resized)


    try:
        # Check if keypoints exist and get first detected person's keypoints
        keypoints_obj = results[0].keypoints
        if keypoints_obj is not None and len(keypoints_obj.xy) > 0:
            keypoints_xy = keypoints_obj.xy[0].cpu().numpy()
            keypoints_conf = keypoints_obj.conf[0].cpu().numpy()

            # Combine xy and confidence into one array
            rounded_results = np.hstack([keypoints_xy, keypoints_conf[:, None]])

            left_knee = rounded_results[body_index["left_knee"]]
            right_knee = rounded_results[body_index["right_knee"]]
            left_ankle = rounded_results[body_index["left_ankle"]]
            right_ankle = rounded_results[body_index["right_ankle"]]

            if (
                (left_knee[2] > 0.5)
                and (right_knee[2] > 0.5)
                and (left_ankle[2] > 0.5)
                and (right_ankle[2] > 0.5)
            ):
                if prev_left_ankle_y is not None and prev_right_ankle_y is not None and wait_frames == 0:
                    left_diff = abs(left_ankle[1] - prev_left_ankle_y)
                    right_diff = abs(right_ankle[1] - prev_right_ankle_y)

                    if max(left_diff, right_diff) > step_threshold:
                        step_count += 1
                        print(f"Step taken: {step_count}")
                        wait_frames = min_wait_frames
                        # Optional: play sound on step detected
                        # playsound(temp_file.name)

                prev_left_ankle_y = left_ankle[1]
                prev_right_ankle_y = right_ankle[1]

                if wait_frames > 0:
                    wait_frames -= 1

        else:
            print("No human detected.")

    except Exception as e:
        print("Error processing keypoints:", e)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

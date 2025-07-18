import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import os
import time

# Load the YOLO models
ball_model = YOLO("C:/Users/saisa/Desktop/AI-Basketball-Referee/basketballModel.pt")
pose_model = YOLO("C:/Users/saisa/Desktop/AI-Basketball-Referee/yolov8s-pose.pt")

# Open the video file
cap = cv2.VideoCapture("C:/Users/saisa/Desktop/AI-Basketball-Referee/video2.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize counters and positions
dribble_count = 0
step_count = 0
prev_x_center = None
prev_y_center = None
prev_left_ankle_y = None
prev_right_ankle_y = None
prev_delta_y = None
ball_not_detected_frames = 0
max_ball_not_detected_frames = 20  # Adjust based on your requirement
dribble_threshold = 18  # Adjust based on observations
step_threshold = 5
min_wait_frames = 7
wait_frames = 0
travel_detected = False
travel_timestamp = None
total_dribble_count = 0
total_step_count = 0

# Define the body part indices
body_index = {"left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}

# Define the frame dimensions and fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Make directory to save travel footage
if not os.path.exists("travel_footage"):
    os.makedirs("travel_footage")

# Initialize frame buffer and frame saving settings
frame_buffer = deque(maxlen=30)  # Buffer to hold frames
save_frames = 60  # Number of frames to save after travel is detected
frame_save_counter = 0
saving = False
out = None

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Append the frame to buffer
    frame_buffer.append(frame)

    # Ball detection
    ball_results_list = ball_model(frame, verbose=False, conf=0.65)
    ball_detected = False

    for results in ball_results_list:
        for bbox in results.boxes.xyxy:
            x1, y1, x2, y2 = bbox[:4]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            if prev_y_center is not None:
                delta_y = y_center - prev_y_center
                if (
                    prev_delta_y is not None
                    and prev_delta_y > dribble_threshold
                    and delta_y < -dribble_threshold
                ):
                    dribble_count += 1
                    total_dribble_count += 1
                prev_delta_y = delta_y

            prev_x_center = x_center
            prev_y_center = y_center

            ball_detected = True
            ball_not_detected_frames = 0

        annotated_frame = results.plot()

    if not ball_detected:
        ball_not_detected_frames += 1

    if ball_not_detected_frames >= max_ball_not_detected_frames:
        step_count = 0

    # Pose detection
    pose_results = pose_model(frame, verbose=False, conf=0.5)

    keypoints = pose_results[0].keypoints
    rounded_results = []

    if keypoints is not None:
        for i in range(len(keypoints.xy)):
            xy = np.round(keypoints.xy[i], 1)
            conf = np.round(keypoints.conf[i], 2).reshape(-1, 1)
            combined = np.hstack((xy, conf))
            rounded_results.append(combined)
        rounded_results = np.array(rounded_results)
    else:
        rounded_results = []

    try:
        if len(rounded_results) > 0:
            left_knee = rounded_results[0][body_index["left_knee"]]
            right_knee = rounded_results[0][body_index["right_knee"]]
            left_ankle = rounded_results[0][body_index["left_ankle"]]
            right_ankle = rounded_results[0][body_index["right_ankle"]]

            if (
                (left_knee[2] > 0.5)
                and (right_knee[2] > 0.5)
                and (left_ankle[2] > 0.5)
                and (right_ankle[2] > 0.5)
            ):
                if (
                    prev_left_ankle_y is not None
                    and prev_right_ankle_y is not None
                    and wait_frames == 0
                ):
                    left_diff = abs(left_ankle[1] - prev_left_ankle_y)
                    right_diff = abs(right_ankle[1] - prev_right_ankle_y)

                    if max(left_diff, right_diff) > step_threshold:
                        step_count += 1
                        total_step_count += 1
                        print(f"Step taken: {step_count}")
                        wait_frames = min_wait_frames  # Update wait_frames

                prev_left_ankle_y = left_ankle[1]
                prev_right_ankle_y = right_ankle[1]

                if wait_frames > 0:
                    wait_frames -= 1
        else:
            print("No human detected.")

    except Exception as e:
        print("Error processing keypoints:", e)

    pose_annotated_frame = pose_results[0].plot()

    # Combine annotated frames
    combined_frame = cv2.addWeighted(
        annotated_frame, 0.6, pose_annotated_frame, 0.4, 0
    )

    # Draw dribble count
    cv2.putText(
        combined_frame,
        f"Dribble count: {total_dribble_count}",
        (50, frame_height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    # Travel detection
    if ball_detected and step_count >2 and dribble_count == 0:
        print("Travel detected!")
        step_count = 0  # reset step count
        travel_detected = True
        travel_timestamp = time.time()
        if not saving:
            filename = os.path.join(
                "travel_footage",
                "travel_{}.mp4".format(time.strftime("%Y%m%d-%H%M%S")),
            )
            out = cv2.VideoWriter(filename, fourcc, 9, (frame_width, frame_height))
            for f in frame_buffer:
                out.write(f)
            saving = True

    if travel_detected and time.time() - travel_timestamp > 3:
        travel_detected = False
        total_dribble_count = 0
        total_step_count = 0

    # Tint frame if travel detected
    if travel_detected:
        blue_tint = np.full_like(combined_frame, (255, 0, 0), dtype=np.uint8)
        combined_frame = cv2.addWeighted(combined_frame, 0.7, blue_tint, 0.3, 0)
        cv2.putText(
            combined_frame,
            "Travel Detected!",
            (frame_width - 600, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )

    # Reset counts on dribble detection
    if dribble_count > 0:
        step_count = 0
        dribble_count = 0

    # Save frames after travel detected
    if saving:
        out.write(frame)
        frame_save_counter += 1
        if frame_save_counter >= save_frames:
            saving = False
            frame_save_counter = 0
            out.release()

    # Resize frame and show output window **always**
    display_frame = cv2.resize(combined_frame, (int(frame_width * 0.7), int(frame_height * 0.7)))
    cv2.imshow("Travel Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release VideoWriter if still open
if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()

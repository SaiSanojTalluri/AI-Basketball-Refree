import cv2
from ultralytics import YOLO
import numpy as np
import time

class BallHoldingDetector:
    def __init__(self):
        # Load the YOLO models for pose estimation and ball detection
        self.pose_model = YOLO("yolov8s-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")

        # Open the webcam or video
        self.cap = cv2.VideoCapture("C:/Users/saisa/Desktop/AI-Basketball-Referee/video2.mp4")

        # Define the body part indices (switch left and right for mirrored image)
        self.body_index = {
            "left_wrist": 10,  # switched
            "right_wrist": 9,  # switched
        }

        # Holding state variables
        self.hold_start_time = None
        self.is_holding = False

        # Holding duration threshold (in seconds)
        self.hold_duration = 0.85

        # Distance threshold to consider the ball is being held
        self.hold_threshold = 300

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                pose_annotated_frame, ball_detected = self.process_frame(frame)

                # Resize to 70%
                scale_percent = 70  # percent of original size
                width = int(pose_annotated_frame.shape[1] * scale_percent / 100)
                height = int(pose_annotated_frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized_frame = cv2.resize(pose_annotated_frame, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow("YOLOv8 Inference", resized_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        # Pose estimation
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()

        # Extract keypoints safely
        keypoints = pose_results[0].keypoints.xy  # (num_persons, num_keypoints, 2)

        if keypoints.shape[0] == 0:
            print("No human detected.")
            return pose_annotated_frame, False

        # For first detected person
        rounded_results = np.round(keypoints[0].cpu().numpy(), 1)

        left_wrist = rounded_results[self.body_index["left_wrist"]]
        right_wrist = rounded_results[self.body_index["right_wrist"]]

        # Ball detection
        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)
        ball_detected = False

        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2

                print(f"Ball coordinates: (x={ball_x_center:.2f}, y={ball_y_center:.2f})")
                ball_detected = True

                left_distance = np.hypot(ball_x_center - left_wrist[0], ball_y_center - left_wrist[1])
                right_distance = np.hypot(ball_x_center - right_wrist[0], ball_y_center - right_wrist[1])

                self.check_holding(left_distance, right_distance)

                # Annotate ball detection
                cv2.rectangle(
                    pose_annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    pose_annotated_frame,
                    f"Ball: ({ball_x_center:.2f}, {ball_y_center:.2f})",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    pose_annotated_frame,
                    f"Left Wrist: ({left_wrist[0]:.2f}, {left_wrist[1]:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    pose_annotated_frame,
                    f"Right Wrist: ({right_wrist[0]:.2f}, {right_wrist[1]:.2f})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    pose_annotated_frame,
                    f"Differentials: ({min(left_distance, right_distance):.2f})",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    pose_annotated_frame,
                    f"Holding: {'Yes' if self.is_holding else 'No'}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

                if self.is_holding:
                    blue_tint = np.full_like(pose_annotated_frame, (255, 0, 0), dtype=np.uint8)
                    pose_annotated_frame = cv2.addWeighted(
                        pose_annotated_frame, 0.7, blue_tint, 0.3, 0
                    )

        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False

        return pose_annotated_frame, ball_detected

    def check_holding(self, left_distance, right_distance):
        if min(left_distance, right_distance) < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif (
                time.time() - self.hold_start_time > self.hold_duration
                and not self.is_holding
            ):
                print("The ball is being held.")
                self.is_holding = True
        else:
            self.hold_start_time = None
            self.is_holding = False

if __name__ == "__main__":
    ball_detection = BallHoldingDetector()
    ball_detection.run()

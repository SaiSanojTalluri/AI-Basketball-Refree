import cv2
from ultralytics import YOLO
import numpy as np
import time

class BallHoldingDetector:
    def __init__(self):
        self.pose_model = YOLO("C:/Users/saisa/Desktop/AI-Basketball-Referee/yolov8s-pose.pt")
        self.ball_model = YOLO("C:/Users/saisa/Desktop/AI-Basketball-Referee/basketballModel.pt")

        self.cap = cv2.VideoCapture("C:/Users/saisa/Desktop/AI-Basketball-Referee/video3.mp4")

        self.body_index = {
            "left_wrist": 10,
            "right_wrist": 9,
        }

        self.hold_start_time = None
        self.is_holding = False
        self.hold_duration = 2
        self.hold_threshold = 300

        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.dribble_threshold = 18

        self.double_dribble_count = 0
        self.was_holding = False

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                pose_annotated_frame, ball_detected = self.process_frame(frame)

                scale_percent = 70
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
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()

        keypoints = pose_results[0].keypoints.xy
        if keypoints.shape[0] == 0:
            return pose_annotated_frame, False

        rounded_results = np.round(keypoints[0].cpu().numpy(), 1)
        left_wrist = rounded_results[self.body_index["left_wrist"]]
        right_wrist = rounded_results[self.body_index["right_wrist"]]

        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)
        ball_detected = False

        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2

                ball_detected = True

                left_distance = np.hypot(ball_x_center - left_wrist[0], ball_y_center - left_wrist[1])
                right_distance = np.hypot(ball_x_center - right_wrist[0], ball_y_center - right_wrist[1])

                self.check_holding(left_distance, right_distance)
                self.update_dribble_count(ball_y_center)

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(pose_annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(pose_annotated_frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Annotations
        cv2.putText(pose_annotated_frame, f"Dribble Count: {self.dribble_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(pose_annotated_frame, f"Holding: {'Yes' if self.is_holding else 'No'}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False

        return pose_annotated_frame, ball_detected

    def check_holding(self, left_distance, right_distance):
        if min(left_distance, right_distance) < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif time.time() - self.hold_start_time > self.hold_duration and not self.is_holding:
                self.is_holding = True
        else:
            self.hold_start_time = None
            self.is_holding = False

    def update_dribble_count(self, y_center):
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center

            if (self.prev_delta_y is not None and
                self.prev_delta_y > self.dribble_threshold and
                delta_y < -self.dribble_threshold):

                self.dribble_count += 1

                # Double dribble logic
                if self.was_holding:
                    self.double_dribble_count += 1
                    print(f"Double Dribble Detected! Count: {self.double_dribble_count}")
                    self.was_holding = False

                    # Reset dribble count after double dribble
                    self.dribble_count = 0

                    # Reset holding
                    self.is_holding = False
                    # print("Holding reset after double dribble increment. Waiting for next holding event.")

                if self.is_holding:
                    self.is_holding = False

            self.prev_delta_y = delta_y

        self.prev_y_center = y_center

        if self.is_holding:
            self.was_holding = True

if __name__ == "__main__":
    ball_detection = BallHoldingDetector()
    ball_detection.run()

import cv2
import numpy as np
from ultralytics import YOLO
import time

class DoubleDribbleDetector:
    def __init__(self):
        # Load models
        self.pose_model = YOLO("yolov8s-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")

        # Body keypoints
        self.body_index = {
            "left_wrist": 9,
            "right_wrist": 10,
        }

        # Video source
        self.cap = cv2.VideoCapture("C:/Users/saisa/Desktop/AI-Basketball-Referee/video3.mp4")

        # Tracking variables
        self.double_dribble_count = 0

        self.is_dribbling = False
        self.was_dribbling = False
        self.dribble_stopped = False

        self.prev_ball_y = None
        self.ball_moving_down = False
        self.ball_moving_up = False

        self.dribble_threshold_distance = 80  # Distance threshold between ball and wrist to consider "holding"

        # For smoothing ball y movement (simple low pass filter)
        self.prev_ball_y_filtered = None
        self.alpha = 0.6

    def run(self):
        cv2.namedWindow("Double Dribble Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Double Dribble Detector", 960, 540)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated_frame = self.process_frame(frame)

            cv2.imshow("Double Dribble Detector", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()

        # Ball detection
        ball_results = self.ball_model(frame, verbose=False, conf=0.5)

        # Extract keypoints safely
        keypoints = pose_results[0].keypoints
        if keypoints is None or len(keypoints.xy) == 0:
            # No person detected
            return pose_annotated_frame

        keypoints_np = keypoints.xy.cpu().numpy()
        if keypoints_np.shape[0] == 0:
            return pose_annotated_frame

        # Use first person detected
        keypoints_person = keypoints_np[0]
        left_wrist = keypoints_person[self.body_index["left_wrist"]]
        right_wrist = keypoints_person[self.body_index["right_wrist"]]

        ball_detected = False
        ball_x = None
        ball_y = None

        for result in ball_results:
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = bbox
                ball_x = (x1 + x2) / 2
                ball_y = (y1 + y2) / 2
                ball_detected = True

                # Draw ball box
                cv2.rectangle(pose_annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(pose_annotated_frame, (int(ball_x), int(ball_y)), 5, (0, 0, 255), -1)
                break  # Only use first detected ball

        if not ball_detected:
            # Reset state if ball not found
            self.is_dribbling = False
            self.was_dribbling = False
            self.dribble_stopped = False
            self.prev_ball_y = None
            self.prev_ball_y_filtered = None
            return pose_annotated_frame

        # Smooth ball y position
        if self.prev_ball_y_filtered is None:
            self.prev_ball_y_filtered = ball_y
        else:
            self.prev_ball_y_filtered = self.alpha * ball_y + (1 - self.alpha) * self.prev_ball_y_filtered

        # Check ball vertical movement direction
        if self.prev_ball_y is not None:
            diff = self.prev_ball_y_filtered - self.prev_ball_y

            # Threshold for movement to filter noise
            movement_threshold = 2  

            if diff > movement_threshold:
                self.ball_moving_up = True
                self.ball_moving_down = False
            elif diff < -movement_threshold:
                self.ball_moving_down = True
                self.ball_moving_up = False
            else:
                self.ball_moving_up = False
                self.ball_moving_down = False

        self.prev_ball_y = self.prev_ball_y_filtered

        # Check distance between ball and wrists
        dist_left = np.linalg.norm(ball_x - left_wrist[0]) + np.linalg.norm(ball_y - left_wrist[1])
        dist_right = np.linalg.norm(ball_x - right_wrist[0]) + np.linalg.norm(ball_y - right_wrist[1])
        min_dist = min(dist_left, dist_right)

        # Determine if ball is held (close to wrist and little vertical movement)
        ball_held = min_dist < self.dribble_threshold_distance and not (self.ball_moving_down or self.ball_moving_up)

        # Determine if dribbling (ball bouncing up and down)
        # Simplified logic: if ball is moving up or down and NOT held, dribbling
        self.is_dribbling = (self.ball_moving_down or self.ball_moving_up) and not ball_held

        # Detect dribble stopped event
        if self.was_dribbling and not self.is_dribbling and ball_held:
            # Dribble stopped - ball held after dribbling
            self.dribble_stopped = True

        # Detect double dribble: dribble stopped then dribble started again
        if self.dribble_stopped and self.is_dribbling:
            self.double_dribble_count += 1
            print(f"Double Dribble detected! Count: {self.double_dribble_count}")
            self.dribble_stopped = False  # Reset to wait for next stop

        self.was_dribbling = self.is_dribbling

        # Display info on frame
        cv2.putText(pose_annotated_frame, f"Double Dribble Count: {self.double_dribble_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(pose_annotated_frame, f"Dribbling: {'Yes' if self.is_dribbling else 'No'}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(pose_annotated_frame, f"Held: {'Yes' if ball_held else 'No'}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return pose_annotated_frame


if __name__ == "__main__":
    detector = DoubleDribbleDetector()
    detector.run()

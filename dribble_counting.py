import cv2
from ultralytics import YOLO
import numpy as np

class DribbleCounter:
    def __init__(self):
        # Load the YOLO model for ball detection
        self.model = YOLO("C:/Users/saisa/Desktop/AI-Basketball-Referee/basketballModel.pt")
        
        # Open the video file
        self.cap = cv2.VideoCapture("C:/Users/saisa/Desktop/AI-Basketball-Referee/video2.mp4")

        # Initialize variables to store the previous position of the basketball
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None

        # Initialize the dribble counter
        self.dribble_count = 0

        # Threshold for the y-coordinate change to be considered as a dribble
        self.dribble_threshold = 18

    def run(self):
        # Process frames until video ends or user quits
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            results_list = self.model(frame, verbose=False, conf=0.65)

            for results in results_list:
                for bbox in results.boxes.xyxy:
                    x1, y1, x2, y2 = bbox[:4]

                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    print(f"Ball coordinates: (x={x_center:.2f}, y={y_center:.2f})")

                    self.update_dribble_count(x_center, y_center)

                    self.prev_x_center = x_center
                    self.prev_y_center = y_center

                annotated_frame = results.plot()

                # Draw dribble count on frame
                cv2.putText(
                    annotated_frame,
                    f"Dribble Count: {self.dribble_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )

                # Dynamically resize window to 70% of the annotated frame size
                frame_height, frame_width = annotated_frame.shape[:2]
                scale_factor = 0.7
                new_width = int(frame_width * scale_factor)
                new_height = int(frame_height * scale_factor)
                small_frame = cv2.resize(annotated_frame, (new_width, new_height))

                cv2.imshow("YOLOv8 Basketball Dribble Counter", small_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

    def update_dribble_count(self, x_center, y_center):
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center

            if (
                self.prev_delta_y is not None
                and self.prev_delta_y > self.dribble_threshold
                and delta_y < -self.dribble_threshold
            ):
                self.dribble_count += 1

            self.prev_delta_y = delta_y


if __name__ == "__main__":
    dribble_counter = DribbleCounter()
    dribble_counter.run()

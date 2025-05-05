from ultralytics import YOLO
import os
import cv2
import numpy as np
import math
import time
# from ultralytics import YOLO

# Load your trained YOLOv8-pose model
# model = YOLO("bestv7-2.pt")

# Export the model to TensorRT engine format
# model.export(format="engine")


def visual_region(center_point, angle):
    length = center_point[1] / 3
    angle_rad = np.radians(angle)
    left_end_point = (
        int(center_point[0] - length * np.sin(angle_rad)),
        int(center_point[1] - length * np.cos(angle_rad))
    )
    right_end_point = (
        int(center_point[0] + length * np.sin(angle_rad)),
        int(center_point[1] - length * np.cos(angle_rad))
    )
    return left_end_point, right_end_point

def get_angle(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    angle_rad = np.arctan2(y2 - y1, x2 - x1) * (180 / np.pi)
    return angle_rad

if __name__ == "__main__":
    model = YOLO("bestv7-2_b10.engine", task="pose")  # Load TensorRT engine
    # model = YOLO("bestv7-2.pt", task="pose")  # Load TensorRT engine
    video_path = "E:/Wajahat/la_chunks/test_bench_3/c13-1.avi"
    cap = cv2.VideoCapture(video_path)
    ymin, ymax = 120, 900
    allowed_angle = 60

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        results = model([frame]*10)

        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                keypoints_data = keypoints.data
                for person_keypoints in keypoints_data:
                    for kp in person_keypoints:
                        x, y, confidence = kp
                        if confidence > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                    A = person_keypoints[2]
                    B = person_keypoints[3]
                    Ax, Ay = A[0].item(), A[1].item()
                    Bx, By = B[0].item(), B[1].item()
                    cv2.line(frame, (int(Ax), int(Ay)), (int(Bx), int(By)), (0, 255, 0), 2)

                    angle = get_angle(Ax, Ay, Bx, By)
                    center = (int((Ax + Bx) / 2), int((Ay + By) / 2))
                    diff = (ymax - ymin) / 2
                    LA_angle_threshold = allowed_angle - int(((By + Ay) / 2 - diff) / (ymax - diff) * 30)
                    left_point, right_point = visual_region(center, LA_angle_threshold)
                    cv2.line(frame, center, left_point, (255, 255, 255), 4)
                    cv2.line(frame, center, right_point, (255, 255, 255), 4)

                    if abs(angle) >= LA_angle_threshold:
                        cv2.putText(frame, "Looking around", (center[0], center[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        # fps = 1.0 / (time.time() - start_time + 1e-4)
        # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        resized_frame = cv2.resize(frame, (1280, 640))
        cv2.imshow('Pose Detection', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        print("fps =", 1.0 / (time.time() - start_time + 0.0001))
        

    cap.release()
    cv2.destroyAllWindows()

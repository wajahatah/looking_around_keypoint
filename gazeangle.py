""" Calculate the gaze angle using ear keypoints and compare the angle 
with the visual field angle. If angle doesn't lie with in the visual field 
predict looking around. Visual Field is dynamic, change with the position 
of student """

from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
import math
import time

import matplotlib.pyplot as plt

def visual_region(center_point, angle):
    length = center_point[1]/3  # Length of the lines

    # Convert degrees to radians
    angle_rad = angle*(3.142/180)

    # Calculate the endpoint for the 30-degree line
    left_end_point_1 = (
        int(center_point[0] - (length * math.sin(angle_rad))),
        int(center_point[1] - length * math.cos(angle_rad))
    )

    # Calculate the endpoint for the -30-degree line
    right_end_point_2 = (
        int(center_point[0] + (length * math.sin(angle_rad))),
        int(center_point[1] - (length * math.cos(angle_rad)))
    )

    return left_end_point_1, right_end_point_2


def get_angle(x1, y1, x2, y2):
        # Avoid division by zero (vertical line case)
        if x1 == x2:
            return 0  # Line coincides with vertical line, angle is 0

        # Calculate slope
        m = (y2 - y1) / (x2 - x1)

        # Calculate complementary angle using arctangent
        angle_rad = np.arctan2([y2 - y1], [x2- x1]) * (180 / np.pi)

        # angle_rad += 90

        return angle_rad

if __name__ == "__main__":

    # model = YOLO("runs/pose/trail32/weights/best_yv8.pt")
    # video_path = "Cam_19_10.mp4"
    # video_path = "video_5_tablet.mp4"
    video_path = "C:/Users/LAMBDA THETA/Videos/evaluation/chunk_06-03-25_13-32-desk21-22-23-24 - Trim.avi"#"la_chunks_4.mp4" #"C:/Users/LAMBDA THETA/Downloads/chunk_31-10-24_14-24.avi"
    # video_path = "C:/Users/LAMBDA THETA/Downloads/test_bench_02/test_bench_02/Cam_104_batch_5_56.mp4"
    # video_path = "C:/Users/LAMBDA THETA/Downloads/test_bench_02/test_bench_02/Cam_19_10.mp4"
    model = YOLO("bestv7-2.pt")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        start_time=time.time()
        # Run inference on the current frame
        global center,ymax,ymin,allowed_angle

        #take ymax and ymin from the desk roi
        ymin = 120
        ymax = 900
        allowed_angle = 60
        results = model(frame)

        # Iterate over each detected person and print their keypoints
        for result in results:
            keypoints = result.keypoints  # Get keypoints as a numpy array or tensor
            if keypoints is not None: 
                keypoints_data=keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    for kp in person_keypoints:
                        x, y, confidence = kp
                        if confidence > 0.5:  # Optional: Only draw keypoints with sufficient confidence
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw the keypoint

                    C = person_keypoints[0]  # Keypoint 0
                    # A = person_keypoints[1]  # Keypoint 1
                    # B = person_keypoints[2]  # Keypoint 2
                    # for v11 v1 model
                    A = person_keypoints[2]  # Keypoint 2
                    B = person_keypoints[3]  # Keypoint 3

                    Cx=int(C[0].item())
                    Cy=int(C[1].item())
                    Ax=A[0].item() 
                    Ay=A[1].item()
                    Bx=B[0].item()
                    By=B[1].item()

                    diff = (ymax - ymin) / 2

                    cv2.line(frame, (int(Ax), int(Ay)), (int(Bx), int(By)), (0, 255, 0),2)  # Green line

                    angle = get_angle(Ax,Ay,Bx,By)
                    # print("angle", angle)

                    # center = (int((Ax + Bx) / 2), int((Ay + By) / 2))

                    # Calculate the endpoint for the gaze line
                    # length = 100  # Length of the gaze line
                    # angle_radians = np.radians(angle)
                    # gaze_end_x = int(center[0])# + 1 * np.cos(angle_radians))
                    # gaze_end_y = int(center[1] + length * np.sin(angle_radians))

                    # Draw the gaze line
                    # cv2.line(frame, (0,ymin), (1200,ymin), (0, 0, 255), 2)  # Red line for gaze
                    # cv2.line(frame, (0,ymax), (1200,ymax), (0, 0, 255), 2)  # Red line for gaze

                    LA_angle_threshold = allowed_angle - int(((By+Ay)/2 - diff) / (ymax - diff) * 30)
                    # LA_angle_threshold = 60
                    center = (int((Ax+Bx)/2), int((Ay+By)/2))

                    frame_visual_area = frame
                    left_point, right_point = visual_region(center, LA_angle_threshold)
                    frame_visual_area = cv2.line(frame_visual_area, center, left_point,  (255, 255, 255), 4) #,(0, 0, 0), 1)
                    frame_visual_area = cv2.line(frame_visual_area, center, right_point,  (255, 255, 255), 4) #,(0, 0, 0), 1)
                    if abs(angle)<LA_angle_threshold:
                        pass
                    else:
                        cv2.putText(frame, "Looking around", (int(center[0]), int(center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


            # for result in results:
                # print("results:", result)
                # annotated_frame = result.plot()  # Draw keypoints and bounding boxes on the frame
            pp = cv2.resize(frame, (1280, 640))
            cv2.imshow('Pose Detection', pp)#annotated_frame)

            # Press 'q' to quit the video display
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close display window
        print("fps =", 1.0 / (time.time() - start_time + 0.0001))
    cap.release()
    cv2.destroyAllWindows()
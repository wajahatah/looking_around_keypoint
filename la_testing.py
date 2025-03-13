from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def visual_region(center_point, angle):
    length = 150  # Length of the lines
    angle_rad = angle*(3.142/180)
    left_end_point_1 = (
        int(center_point[0] - (length * math.sin(angle_rad))),
        int(center_point[1] - length * math.cos(angle_rad))
    )
    right_end_point_2 = (
        int(center_point[0] + (length * math.sin(angle_rad))),
        int(center_point[1] - (length * math.cos(angle_rad)))
    )
    return left_end_point_1, right_end_point_2

def get_angle(x1, y1, x2, y2):
        if x1 == x2:
            return 0  # Line coincides with vertical line, angle is 0
        m = (y2 - y1) / (x2 - x1)
        angle_rad = np.arctan2([y2 - y1], [x2- x1]) * (180 / np.pi)
        angle_rad += 90
        return angle_rad

if __name__ == "__main__":

    model = YOLO("runs/pose/trail52/weights/best_v11_2.pt")
    video_path = "Cam_19_10.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        results = model(frame)
        for result in results:
            keypoints = result.keypoints  # Get keypoints as a numpy array or tensor
            if keypoints is not None: 
                keypoints_data=keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    print("person:", person_idx)

                    #for v11 v2 model
                    C = person_keypoints[0]  # Keypoint 0
                    A = person_keypoints[2]  # Keypoint 2
                    B = person_keypoints[3]  # Keypoint 3

                    Cx=int(C[0].item())
                    Cy=int(C[1].item())
                    Ax=A[0].item() 
                    Ay=A[1].item()
                    Bx=B[0].item()
                    By=B[1].item()

                    x_D = int((Ax+Bx)/2)
                    y_D = int((Ay+By)/2)

                    angle = get_angle(x_D,y_D,Cx,Cy)

                    cv2.line(frame, (int(Ax), int(Ay)), (int(Bx), int(By)), (0, 255, 0),2)  # Green line
                
                    # Draw line from C to D
                    cv2.line(frame, (int(Cx), int(Cy)), (int(x_D) ,int(y_D)), (255, 0, 0),4)  # Blue line

                    # cv2.line(frame, (int(x_D), int(y_D)), (int(x_D) ,int(y_D - 100)), (255, 0, 0),4)  # Blue line
                    
                    center = int(x_D) , int(y_D)
                
                    frame_visual_area = frame
                    # diff = (ymax - ymin) / 2

                    # if Ay and By < left_y or d and b < desk_roi[str(la)]['right_y']:
                    #     allowed_angle = e_angle
                    # else:
                    #     allowed_angle = i_angle

                    # LA_angle_threshold = 60 - int(((By+Ay)/2 - diff) / (ymax - diff) * 30)
                    # LA_angle_threshold = 60 

                    # left_point, right_point = visual_region(center, LA_angle_threshold)
                    # frame_visual_area = cv2.line(frame_visual_area, center, left_point, (255, 255, 255), 4)#, (0, 0, 0), 1)
                    # frame_visual_area = cv2.line(frame_visual_area, center, right_point, (255, 255, 255), 4)#, (0, 0, 0), 1)

                    # if abs(angle)<LA_angle_threshold:
                    #     pass
                    # else:
                    #     cv2.putText(frame, "Looking around", (int(center[0]), int(center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            for result in results:
                annotated_frame = result.plot()  # Draw keypoints and bounding boxes on the frame
            pp = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow('Pose Detection', pp)

            # Press 'q' to quit the video display
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close display window
    cap.release()
    cv2.destroyAllWindows()
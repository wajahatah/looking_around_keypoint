""" Calculate the area of 4 triangles LRD,LCD,LRC,RDCand compare them, 
if the sum of LCD,LRC,RDC is not equal to LRD predict looking around.
The triangular area is dynamic, change with the position of student"""

from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
                    

def visual_region(center_point, angle):
    length = 150  # Length of the lines

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

def calculate_area(x1, y1, x2, y2, x3, y3):
    """
    Calculate the area of a triangle given its vertices.
    """
    return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0)

def calculate_angle(d):
    if d <= 150:
        return 90  # Maximum angle
    elif d >= 290:
        return 30  # Minimum angle
    else:
        return 30 + 60 * (190 - d) / 120

if __name__ == "__main__":

    # model = YOLO("runs/pose/trail32/weights/best_yv8.pt")
    model = YOLO("runs/pose/trail4/weights/best_y11.pt")
    # video_path = "Cam_19_10.mp4"
    video_path = "la_chunks_1.mp4" #"C:/Users/LAMBDA THETA/Downloads/chunk_31-10-24_14-24.avi"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        global center,ymax,ymin

        #take ymax and ymin from the desk roi
        ymax = 880
        ymin = 120

        # Run inference on the current frame
        results = model(frame)
        # Iterate over each detected person and print their keypoints
        for result in results:
            keypoints = result.keypoints  # Get keypoints as a numpy array or tensor
            if keypoints is not None: 
                keypoints_data=keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    
                    C = person_keypoints[0]  # Keypoint 0
                    A = person_keypoints[2]  # Keypoint 2
                    B = person_keypoints[3]  # Keypoint 3

                    Cx=int(C[0].item())
                    Cy=int(C[1].item())
                    Ax=A[0].item() 
                    Ay=A[1].item()
                    Bx=B[0].item()
                    By=B[1].item()

                    AB_x = Bx - Ax
                    AB_y = By - Ay
                    
                    # Calculate the vector from A to C
                    AC_x = Cx - Ax
                    AC_y = Cy - Ay
                    
                    # Projection formula to find the scalar t for point D on AB
                    if all(value != 0 for value in [Cx, Cy, Ax, Ay, Bx, By]):
                        t = (AC_x * AB_x + AC_y * AB_y) / (AB_x ** 2 + AB_y ** 2)
                    
                    # Calculate the coordinates of the perpendicular point D
                    x_D = int(Ax + t * AB_x)
                    y_D = int(Ay + t * AB_y)

                    cv2.line(frame, (int(Ax), int(Ay)), (int(Bx), int(By)), (0, 255, 0),2)  # Green line
                
                    # Draw line from C to D
                    cv2.line(frame, (int(Cx), int(Cy)), (int(x_D) ,int(y_D)), (255, 0, 0),4)  # Blue line

                    # Draw the gaze line
                    cv2.line(frame, (0,ymin), (1200,ymin), (0, 0, 255), 2)  # Red line for gaze
                    cv2.line(frame, (0,ymax), (1200,ymax), (0, 0, 255), 2)  # Red line for gaze
                    
                    
                    center = int(x_D) , int(y_D)
                
                    frame_visual_area = frame
                    diff = (ymax - ymin) / 2

                    LA_angle_threshold = 60 - int(((By+Ay)/2 - diff) / (ymax - diff) * 30)

                    left_point, right_point = visual_region(center, LA_angle_threshold)
                    frame_visual_area = cv2.line(frame_visual_area, center, left_point, (255, 255, 255), 4)#, (0, 0, 0), 1)
                    frame_visual_area = cv2.line(frame_visual_area, center, right_point, (255, 255, 255), 4)#, (0, 0, 0), 1)

                    Lx, Ly = int(left_point[0]), int(left_point[1])
                    Rx , Ry = int(right_point[0]), int(right_point[1])
                   
                    area_LRD = calculate_area(Lx, Ly, Rx, Ry, x_D, y_D)

                    area_LRC= calculate_area(Lx, Ly, Rx, Ry, Cx, Cy)
                    area_CRD = calculate_area(Cx, Cy, Rx, Ry, x_D, y_D)
                    area_LCD = calculate_area(Lx, Ly, Cx, Cy, x_D, y_D)

                    if area_LRD == (area_CRD + area_LCD + area_LRC):
                        pass
                    else:
                        cv2.putText(frame, "Looking around", (int(center[0]), int(center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

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
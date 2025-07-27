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
import json

import matplotlib.pyplot as plt
la = False

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

def get_position(x,roi_data_list):
    for roi in roi_data_list:
        if roi['xmin'] <= x <= roi['xmax']:
            return roi['position'] 
    return None

# if __name__ == "__main__":

    # model = YOLO("runs/pose/trail32/weights/best_yv8.pt")
    # video_path = "F:/Wajahat/LA_FP_v8/1751974933.827761.mp4"
    # video_path = "C:/Users/LAMBDA THETA/Videos/evaluation/chunk_06-03-25_13-32-desk21-22-23-24 - Trim.avi"#"la_chunks_4.mp4" #"C:/Users/LAMBDA THETA/Downloads/chunk_31-10-24_14-24.avi"
    # video_path = "C:/Users/LAMBDA THETA/Downloads/test_bench_02/test_bench_02/Cam_104_batch_5_56.mp4"
    # video_path = "C:/Users/LAMBDA THETA/Downloads/test_bench_02/test_bench_02/Cam_19_10.mp4"
video_folder = "F:/Wajahat/looking_around_panic/may_11/Looking Around/TP"
# "F:\Wajahat\looking_around_panic\may_11\Looking Around\TP"
# video_folder = "C:/Users/LAMBDA THETA/Downloads/test"
json_file = "qiyas_multicam.camera_final.json"
model = YOLO("bestv8-2.pt")

video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4") or ('.avi')]

for video_file in video_files:
    video_path = os.path.join(video_folder,video_file)
    print(f"processing video {video_file}")

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        continue
    
    # frame = cv2.resize(frame, (1280,720))
    frame = cv2.resize(frame, (500,500))
    cv2.imshow("select camera", frame)
    cv2.waitKey(1)

    with open(json_file, "r") as f:
        camera_config = json.load(f)

    skip_video= False
    while True:
        cam_id = input("Enter camera num or s to skip: ")
        if cam_id.lower() == 's':
            skip_video = True
            cap.release()
            cv2.destroyWindow("select camera")
            break
        
        cam_key = f"camera_{cam_id}"
        camera_data = next((cam for cam in camera_config if cam["_id"] == cam_key), None)
        if camera_data:
            break
        print("Invalid Camera ID")

    if skip_video:
        continue

    cv2.destroyWindow("select camera")

    roi_data_list = list(camera_data['data'].values())
    roi_lookup = {roi['position']: roi for roi in roi_data_list}


# cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        frame = cv2.resize(frame, (1280, 720))
        display_frame = frame.copy()
        display_frame = cv2.resize(display_frame, (840,500))
        allowed_angle = 60
        results = model(frame)

        for result in results:
            if not hasattr(result, "keypoints") or result.keypoints is None:
                continue
            keypoints = result.keypoints.data  # Get keypoints as a numpy array or tensor
            for person_idx, person_keypoints in enumerate(keypoints):
                keypoints_list = []
                # for kp_tensor in keypoints:
                    # for kp in kp_tensor[:10]:
                for kp in person_keypoints[:10]:
                    x, y, confidence = kp[:3].cpu().numpy()                        
                    if confidence >=0.5:
                        keypoints_list.append((x,y))
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw the keypoint
                    else:
                        keypoints_list.append((0,0))

                # print(f"Keypoint_list: {keypoints_list}")

                if not keypoints_list or all (x==0 and y==0 for x,y in keypoints_list):
                    continue

                person_x = keypoints_list[0][0]
                position = get_position(person_x, roi_data_list)
                if position is None:
                    continue
                a = int(float(keypoints_list[2][0]))
                b = int(float(keypoints_list[2][1]))
                c = int(float(keypoints_list[3][0]))
                d = int(float(keypoints_list[3][1]))
                cv2.circle(frame,(a,b), 5, (0,0,255), -1)
                cv2.circle(frame,(c,d), 5, (255,0,255), -1)

                if d and b > 0:
                    angle = get_angle(a,b,c,d)

                    roi = roi_lookup.get(position)
                    if not roi:
                        continue

                    xmin, xmax = roi['xmin'], roi['xmax']
                    ymin, ymax = roi['ymin'], roi['ymax']
                    box_x = (xmin,ymin)
                    box_y = (xmax,ymax)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, lineType=1)
                    left_x, right_x, left_y, right_y = roi['left_x'], roi['right_x'], roi['left_y'], roi['right_y']

                    gy = int((left_y + right_y) / 2)
                    x = float((((d+b)/2)-gy) / (ymax - gy))
                    # LA_angle_threshold = (149.15)*x**6 - (105.85)*x**5 - (212.34)*x**4 + (145.46)*x**3 + (87.967)*x**2 - (93.82)*x + 50.668
                    LA_angle_threshold = (10.581)*x**6 - (25.393)*x**5 - (36.068)*x**4 + (47.076)*x**3 + (30.868)*x**2 - (45.176)*x + 61.568
                    la = False
                    if abs(angle) > LA_angle_threshold:
                        la = True
                        cv2.putText(frame, "Looking Around", (a,b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        # cv2.putText(display_frame, "Looking Around", (a,b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)


                    center = (int((a+c)/2), int((b+d)/2))
                    left_point, right_point = visual_region(center, LA_angle_threshold)
                    cv2.line(frame, center, left_point, (255,255,255),4)
                    cv2.line(frame, center, right_point, (255,255,255), 4)

                        # Cx=int(C[0].item())
                        # Cy=int(C[1].item())
                        # Ax=A[0].item() 
                        # Ay=A[1].item()
                        # Bx=B[0].item()
                        # By=B[1].item()

                        # diff = (ymax - ymin) / 2

                        # cv2.line(frame, (int(Ax), int(Ay)), (int(Bx), int(By)), (0, 255, 0),2)  # Green line

                        # angle = get_angle(Ax,Ay,Bx,By)
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

                        # LA_angle_threshold = allowed_angle - int(((By+Ay)/2 - diff) / (ymax - diff) * 30)
                        # # LA_angle_threshold = 60
                        # center = (int((Ax+Bx)/2), int((Ay+By)/2))

                        # frame_visual_area = frame
                        # left_point, right_point = visual_region(center, LA_angle_threshold)
                        # frame_visual_area = cv2.line(frame_visual_area, center, left_point,  (255, 255, 255), 4) #,(0, 0, 0), 1)
                        # frame_visual_area = cv2.line(frame_visual_area, center, right_point,  (255, 255, 255), 4) #,(0, 0, 0), 1)
                        # if abs(angle)<LA_angle_threshold:
                        #     pass
                        # else:
                        #     cv2.putText(frame, "Looking around", (int(center[0]), int(center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


                # for result in results:
                    # print("results:", result)
                    # annotated_frame = result.plot()  # Draw keypoints and bounding boxes on the frame
                # pp = cv2.resize(frame, (1280, 640))
                # frame = cv2.resize(frame, (840,500))
                # if la == True:
                #     cv2.imshow('Pose Detection', frame)
                #     if cv2.waitKey(0) & 0xff ==ord('q'):
                #         break

                # else:
        cv2.imshow('Pose Detection', frame)#annotated_frame)

        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release video capture and close display window
            # print("fps =", 1.0 / (time.time() - start_time + 0.0001))
cap.release()
cv2.destroyAllWindows()
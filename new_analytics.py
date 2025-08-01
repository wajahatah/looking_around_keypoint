# Qiyas multi cam codes module changed for debugging can't be used for production as a backup saving here

from collections import deque
import re
import math
import constants
from datetime import datetime
import cv2
import numpy as np
from analytics.head_buffer import Head_buffer

class analytics:
    def __init__(self, cam_id, number_of_students, i_angle, e_angle):
        self.buffer_size = constants.action_window_buffer_size
        self.action_buffer = deque(maxlen=self.buffer_size)
        self.initial_appends = True
        self.number_of_students = number_of_students
        self.before = ["Normal"] * self.number_of_students
        # self.allowed_angle = 60
        self.before_abset = []
        self.cam_id = cam_id
        self.candidate_list = [i + 1 for i in range(self.number_of_students)]
        self.head_buffer = Head_buffer(self.candidate_list)
        self.i_angle = i_angle
        self.e_angle = e_angle
        self.action_anomaly_precedence_dict = constants.action_anomaly_precedence_dict
        self.action_precedence = list(self.action_anomaly_precedence_dict.values())
        self.action_precedence_names = list(self.action_anomaly_precedence_dict.keys())

    ## Estimating sitting order of the candidates
    def candidate_number_head(self, head, desk_bbox):
        keys = list(desk_bbox.keys())
        candidate_num = 0
        center_point = ((head[0] + head[2]) / 2, (head[1] + head[3]) / 2)
        for dsk in keys:
            if (
                desk_bbox[dsk]["xmax"] >= center_point[0] >= desk_bbox[dsk]["xmin"]
                and desk_bbox[dsk]["ymax"] >= center_point[1] >= desk_bbox[dsk]["ymin"]
            ):
                candidate_num = int(dsk)

        if candidate_num > 0:
            return candidate_num
        else:
            return print("No Candidate found in selected ROI")

    def remove_numbers(self, item):
        if isinstance(item, list):
            return [re.sub(r"\[\d+\.\d+\]", "", elem).strip() for elem in item]
        elif isinstance(item, str):
            return re.sub(r"\[\d+\.\d+\]", "", item).strip()
        return item

    def calculate_area(self, box):
        """Calculates the area of a bounding box."""
        x_min, y_min, x_max, y_max = box
        return max(0, x_max - x_min) * max(0, y_max - y_min)

    def calculate_intersection_area(self, box1, box2):
        """Calculates the intersection area between two bounding boxes."""
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        # Find overlap coordinates
        x_min_inter = max(x_min1, x_min2)
        y_min_inter = max(y_min1, y_min2)
        x_max_inter = min(x_max1, x_max2)
        y_max_inter = min(y_max1, y_max2)

        # Calculate intersection area
        return self.calculate_area((x_min_inter, y_min_inter, x_max_inter, y_max_inter))

    def calculate_percentage_coverage(self, ground_truth_box, predicted_box):
        """Calculates the percentage of the ground truth box area covered by the predicted box."""
        intersection_area = self.calculate_intersection_area(
            ground_truth_box, predicted_box
        )
        ground_truth_area = self.calculate_area(ground_truth_box)

        # Calculate percentage coverage
        if ground_truth_area == 0:
            return 0  # Avoid division by zero
        return (intersection_area / ground_truth_area) * 100

    @staticmethod
    def find_lenght_2_points(x1, y1, x2, y2):
        # Calculate distance
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def find_roi_with_max_iou(self, rois, stand_head, pose_result):
        overling = []
        for s_head in stand_head:
            for region_id, roi in rois.items():
                overlap = self.calculate_percentage_coverage(
                    (s_head[0], s_head[1], s_head[2], s_head[3]),
                    (roi["xmin"], roi["ymin"], roi["xmax"], roi["ymax"]),
                )
                if overlap > constants.overlap_threshold:
                    matched = None
                    for p_result in pose_result:
                        if (
                            s_head[0] < p_result["pose_midpoint"][0]
                            and s_head[2] > p_result["pose_midpoint"][0]
                        ):
                            matched = p_result["decision"]
                    if matched != 0:
                        distance_left = self.find_lenght_2_points(
                            s_head[0], s_head[1], roi["xmin"], roi["ymax"]
                        )
                        distance_right = self.find_lenght_2_points(
                            s_head[2], s_head[1], roi["xmax"], roi["ymax"]
                        )
                        ## left line (black)
                        # cv2.line(frame, (s_head[0],s_head[1]), (roi['xmin'],roi['ymax']), (0, 0, 0), 4)
                        ## right line (blue)
                        # cv2.line(frame, (s_head[2],s_head[1]), (roi['xmax'],roi['ymax']), (255, 0, 0), 4)
                        if distance_left > distance_right:
                            # print("invigilator standing at right edge of, ", region_id)
                            if matched == 1:
                                overling.append(str(int(region_id) + 1))
                            else:
                                overling.append(region_id)
                        else:
                            # print("invigilator standing at left edge of, ", region_id)
                            if matched == -1:
                                overling.append(str(int(region_id) - 1))
                            else:
                                overling.append(region_id)
                    else:
                        overling.append(region_id)
        return overling

    def visual_region(self, center_point, th_angle):
        length = 150  # Length of the lines

        # Convert degrees to radians
        angle_rad = th_angle * (3.142 / 180)

        # Calculate the endpoint for the 30-degree line
        left_end_point_1 = (
            int(center_point[0] - (length * math.sin(angle_rad))),
            int(center_point[1] - length * math.cos(angle_rad)),
        )

        # Calculate the endpoint for the -30-degree line
        right_end_point_2 = (
            int(center_point[0] + (length * math.sin(angle_rad))),
            int(center_point[1] - (length * math.cos(angle_rad))),
        )

        return left_end_point_1, right_end_point_2

    def get_angle(self, x1, y1, x2, y2):
        # Avoid division by zero (vertical line case)
        if x1 == x2:
            return 0  # Line coincides with vertical line, angle is 0

        # Calculate slope
        m = (y2 - y1) / (x2 - x1)

        # Calculate complementary angle using arctangent
        angle_rad = np.arctan2([y2 - y1], [x2 - x1]) * (180 / np.pi)

        # angle_rad += 90

        return angle_rad

    def compare_arrays(self, current):
        new_values = set(current) - set(self.before_abset)
        missing_values = set(self.before_abset) - set(current)
        return list(new_values), list(missing_values)

    @staticmethod
    def calculate_angle_between_vectors(p1, p2, p3, p4):
        """
        Calculate the signed angle between two lines using their directional vectors.
        The result will range from -180 to 180 degrees.

        Parameters:
            p1, p2: Tuple (x1, y1), (x2, y2) - Points on the first line.
            p3, p4: Tuple (x3, y3), (x4, y4) - Points on the second line.

        Returns:
            angle (float): Signed angle between the two lines in degrees.
        """
        # Define directional vectors for the two lines
        u = (p2[0] - p1[0], p2[1] - p1[1])  # Vector for line 1
        v = (p4[0] - p3[0], p4[1] - p3[1])  # Vector for line 2

        # Calculate dot product u . v
        dot_product = u[0] * v[0] + u[1] * v[1]

        # Calculate magnitudes of u and v
        magnitude_u = math.sqrt(u[0] ** 2 + u[1] ** 2)
        magnitude_v = math.sqrt(v[0] ** 2 + v[1] ** 2)

        # Ensure we don't divide by zero (check for zero-length vectors)
        if magnitude_u == 0 or magnitude_v == 0:
            raise ValueError("One of the lines has zero length.")

        # Calculate cosine of the angle
        cos_theta = dot_product / (magnitude_u * magnitude_v)

        # Clamp cos_theta to avoid math domain error due to floating-point precision
        cos_theta = max(-1, min(1, cos_theta))

        # Calculate the angle in radians
        angle_radians = math.acos(cos_theta)

        # Calculate the cross product u x v (in 2D, this is a scalar)
        cross_product = u[0] * v[1] - u[1] * v[0]

        # Determine the sign of the angle
        if cross_product < 0:
            angle_radians = -angle_radians

        # Convert angle to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    @staticmethod
    def find_endpoint(base_start, base_end, length):
        """
        Calculate the endpoint of the vertical line given the start, end of the base, and length.

        Parameters:
            base_start (tuple): Coordinates of the start of the base line (x1, y1).
            base_end (tuple): Coordinates of the end of the base line (x2, y2).
            length (float): Length of the vertical line.

        Returns:
            tuple: Coordinates of the endpoint (x, y).
        """
        # Coordinates of the base line
        x1, y1 = base_start
        x2, y2 = base_end

        # Calculate the angle of the base line
        delta_x = x2 - x1
        delta_y = y2 - y1
        angle = math.atan2(delta_y, delta_x)

        # Vertical line angle (90 degrees to base line)
        vertical_angle = angle + math.pi / 2  # Add 90 degrees in radians

        # Calculate the endpoint of the vertical line
        x_end = x2 + length * math.cos(vertical_angle)
        y_end = y2 + length * math.sin(vertical_angle)

        return (int(x_end), int(y_end))

    def analytical_computation(
        self,
        desk_roi,
        heads,
        stand_bbox,
        frame,
        stand_head,
        keypoints,
        phone_using,
        phone_bbox,
        hand_in_pocket_flag,
        grey_alert_flag
    ):

        frame_visual_area = frame
        candidate_order = []
        head_candidate_mapping = {}

        alert_notification = []
        action_final = []

        actions_classes = [
            "grey alert",
            "using mobile phone",
            "student interacting with invigilator",
            "looking around",
            "hand in pocket"
        ]
        # Inner list is for actions and outer list is of students
        action_matrix = [[0 for _ in range(len(actions_classes))] for _ in range(self.number_of_students)]



        absent_candidates = {
            "card_resetting": {i + 1: "False" for i in range(self.number_of_students)}
        }

        for gz in range(len(heads)):
            candidate_num = self.candidate_number_head(heads[gz], desk_roi)

            if candidate_num is None:
                continue  # if no head in any of the ROI then skip all computation

            if candidate_num not in candidate_order:
                candidate_order.append(candidate_num)
                head_candidate_mapping[candidate_num] = gz


        looking_around_list = [0] * self.number_of_students

        ## Changing the head box based flag list to candiate order based, mapping the flag to candiate number i:e desk roi ID
        # while len(candidate_order) < len(hand_flag):
        #     candidate_order.append(0)
        # hand_flag = np.array(hand_flag) * np.array(candidate_order)
        # hand_flag = hand_flag.tolist()

        pose_data = {}
        invigilator_pose = []
        ## aranging the key points detection based on the candiate order based on the head detection
        for data in keypoints:
            cv2.circle(frame, (data["Ax"], data["Ay"]), 2, (0, 255, 0), 3)
            cv2.circle(frame, (data["Bx"], data["By"]), 2, (0, 255, 0), 3)
            cv2.circle(frame, (data["Cx"], data["Cy"]), 2, (255, 255, 255), 3)
            # cv2.circle(frame, (data["Slx"], data["Sly"]), 2, (0, 255, 0), 3)
            # cv2.circle(frame, (data["Elx"], data["Ely"]), 2, (0, 255, 0), 3)
            # cv2.circle(frame, (data["Hlx"], data["Hly"]), 2, (0, 255, 0), 3)
            # cv2.circle(frame, (data["Srx"], data["Sry"]), 2, (0, 255, 0), 3)
            # cv2.circle(frame, (data["Erx"], data["Ery"]), 2, (0, 255, 0), 3)
            # cv2.circle(frame, (data["Hrx"], data["Hry"]), 2, (0, 255, 0), 3)
            matched = False
            for key, value in head_candidate_mapping.items():
                head_data = heads[value]
                if (head_data[0] < data["Cx"] < head_data[2]) and (
                    head_data[1] < data["Cy"] < head_data[3]
                ):
                    id = key
                    pose_data[id] = data
                    matched = True
                    break
            if not matched:
                for stand in stand_head:
                    if (stand[0] < data["Cx"] < stand[2]) and (
                        stand[1] < data["Cy"] < stand[3]
                    ):
                        invigilator_pose.append(data)
        
        ## Invigilator pose processing for the interaction class, computing where the invigilator is looking
        pose_result = []
        for pose in invigilator_pose:
            x_D = int((pose['Ax']+pose['Bx'])/2)
            y_D = int((pose['Ay']+pose['By'])/2)
            try:
                top_point = self.find_endpoint((pose['Ax'], pose['Ay']),(x_D,y_D),-160)
                angle = self.calculate_angle_between_vectors(p1=(x_D,0),p2=(x_D ,y_D),p3=top_point,p4=(x_D ,y_D))
                # decision {
                #     0: if looking strainght,
                #     -1: if looking left,
                #     1: if looking right 
                # }
                if abs(angle) < 15: 
                    decision = 0
                elif angle >= 0:
                    decision = 1
                else:
                    decision = -1
                
                pose_result.append({"pose_midpoint":(x_D,y_D),"decision":decision})
            except:
                print("Error at: ",self.cam_id, "Metedata = ", pose)
            

            # pose_result.append({"pose_midpoint": (x_D, y_D), "decision": decision})
        
        ## calculating the looking around from pose data.
        for la in head_candidate_mapping:
            if la in pose_data and pose_data[la] is not None:
                a,b = pose_data[la]['Ax'], pose_data[la]['Ay']
                e,d = int(pose_data[la]['Bx']), int(pose_data[la]['By'])
                
                if b and d > 0: #new logic to handle wrong visual field
                    angle = self.get_angle(a,b,e,d) 
                    print("Angle: ",angle)
                    
                    diff = (desk_roi[str(la)]['ymax'] - desk_roi[str(la)]['ymin']) / 2
                    ## New condition Code

                    # if a and b < left_point[1] and a and b < right_point[1]:
                    if d and b < desk_roi[str(la)]['left_y'] or d and b < desk_roi[str(la)]['right_y']:
                        frame_visual_area = cv2.line(frame, (desk_roi[str(la)]['left_x'],desk_roi[str(la)]['left_y']), (desk_roi[str(la)]['right_x'],desk_roi[str(la)]['right_y']), (0, 255, 0), 4)
                        allowed_angle = self.e_angle
                    else:
                        allowed_angle = self.i_angle
                        frame_visual_area = cv2.line(frame, (desk_roi[str(la)]['left_x'],desk_roi[str(la)]['left_y']), (desk_roi[str(la)]['right_x'],desk_roi[str(la)]['right_y']), (0, 255, 0), 4)

                    ## New condition code end
                    # LA_angle_threshold = allowed_angle - int(((d+b)/2 - diff) / (desk_roi[str(la)]['ymax'] - diff) * 15)
                    x = float(((d+b)/2 - diff) / (desk_roi[str(la)]['ymax'] - diff))
                    # LA_angle_threshold = allowed_angle - 107363*x**6 - 486809*x**5 + 907596**x**4 - 889781*x**3 + 483419*x**2 - 137968*x + 16195
                    LA_angle_threshold = 107363*x**6 - 486809*x**5 + 907596**x**4 - 889781*x**3 + 483419*x**2 - 137968*x + 16195
                    print("allowed_angle: ",allowed_angle)
                    print("x: ",x)
                    print("numerator",((d+b)/2 - diff))
                    print("denomenator",(desk_roi[str(la)]['ymax'] - diff))
                    print("LA_angle_threshold: ",LA_angle_threshold)
                    
                    center = (int((a+e)/2), int((b+d)/2))
                    left_point, right_point = self.visual_region(center, LA_angle_threshold)
                    frame_visual_area = cv2.line(frame, center, left_point, (255, 255, 255), 4)
                    frame_visual_area = cv2.line(frame, center, right_point, (255, 255, 255), 4) 
                    frame_visual_area = cv2.line(frame, (a,b), (e,d), (0, 255, 0), 4)
                    cv2.putText(frame, str(angle), (center[0], center[1] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    
                    if abs(angle) < LA_angle_threshold:
                        looking_around_list[la-1] = 0

                    elif abs(angle) > LA_angle_threshold:
                        looking_around_list[la-1] = 1
                        cv2.putText(frame, "Looking Around", (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        print("Looking Around")
                        # print("LA_list: ", looking_around_list)
                    # if constants.visual_field_drawing: 

        Stand_max_iou_roi = self.find_roi_with_max_iou(
            desk_roi, stand_head, pose_result
        )

        for k in range(len(action_matrix)):
            # Update Looking around
            action_matrix[k][actions_classes.index("looking around")] = looking_around_list[k]

            # Update for 'grey hand in pocket alert'
            action_matrix[k][actions_classes.index("grey alert")] = grey_alert_flag[k]
            # Update for 'using mobile phone'
            action_matrix[k][actions_classes.index("using mobile phone")] = phone_using[
                k
            ]
            # Update for 'hand in pocket'
            action_matrix[k][actions_classes.index("hand in pocket")] = hand_in_pocket_flag[k]

            # Update for 'student interacting with invigilator'
            action_matrix[k][
                actions_classes.index("student interacting with invigilator")
            ] = (
                1
                if str(k + 1) in Stand_max_iou_roi and (k + 1) in candidate_order
                else 0
            )

        ## To fill the buffer until it reaches its max limit
        if self.initial_appends:
            self.action_buffer.append(action_matrix)
            if len(self.action_buffer) == self.buffer_size:
                self.initial_appends = False

        ## As the buffer reaches its limit, its computation started with sliding window
        else:
            # Calculating the average of the actions in buffer for each candidate
            avg_action = np.mean(self.action_buffer, axis=0)

            # Identify actions where the average exceeds 0.8 for each person
            exceeds_threshold = (avg_action > constants.avg_action_threshold).astype(
                int
            )

            ## checking if the average of actions in buffer exist in the present new frame
            current_comparison = (exceeds_threshold & action_matrix).astype(int)
            current_comparison = current_comparison.tolist()

            # updating current comparison with strings based on precedence of actions if there were two or more actions found for same person.
            comparison_updated = []
            for hi, c in enumerate(current_comparison):

                c2 = sum(np.multiply(c, self.action_precedence))
                if c2 == 0:
                    comparison_updated.append("Normal")
                elif c2 >= 16:
                    comparison_updated.append(
                        self.action_precedence_names[self.action_precedence.index(16)]
                    )
                elif c2 >= 8:
                    comparison_updated.append(
                        self.action_precedence_names[self.action_precedence.index(8)]
                    )
                elif c2 >= 4:
                    comparison_updated.append(
                        self.action_precedence_names[self.action_precedence.index(4)]
                    )

                elif c2 >= 2:
                    comparison_updated.append(
                        self.action_precedence_names[self.action_precedence.index(2)]
                    )
                else:
                    comparison_updated.append(
                        self.action_precedence_names[self.action_precedence.index(1)]
                    )

            for index, (bef, curr) in enumerate(zip(self.before, comparison_updated)):
                if bef == curr:
                    action_final.append(curr)
                elif curr == "Normal":
                    action_final.append(curr)
                else:
                    # print("Updated desk roi",desk_roi)
                    alert_notification.append(
                        {
                            index
                            + 1: {
                                "id": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "desk": desk_roi[str(index + 1)]["desk"],
                                "alert_title": curr,
                            }
                        }
                    )

                    action_final.append(curr)

            self.before = action_final
            self.action_buffer.append(action_matrix)

        ## finding the abscent candidate to send to frontend for reset everything
        missing_candidates = list(set(self.candidate_list) - set(candidate_order))
        # Sort the missing candidate to maintain a consistent order
        missing_candidates.sort(key=lambda x: self.candidate_list.index(x))
        self.head_buffer.update_buffer(missing_candidates=missing_candidates)
        new, mising = self.compare_arrays(missing_candidates)
        message = None
        if new or mising:
            # print("NEW: ",new," missing: ",mising)
            message = {
                "cam_id": self.cam_id,
                "title": "message",
                "inactive": [desk_roi[str(n)]["desk"] for n in new],
                "activate": [desk_roi[str(m)]["desk"] for m in mising],
            }

        self.before_abset = missing_candidates

        ## Drawing Code ##
        if constants.desk_roi_drawing:
            for key, roi in desk_roi.items():
                cv2.rectangle(
                    frame_visual_area,
                    (roi["xmin"], roi["ymin"]),
                    (roi["xmax"], roi["ymax"]),
                    (255, 0, 255),
                    2,
                )
                if constants.desk_number_drawing:
                    cv2.putText(
                        frame,
                        str(roi["desk"]),
                        (roi["xmin"] + 60, roi["ymin"] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
        if constants.phone_bbox_drawing:
            for phone in phone_bbox:
                cv2.rectangle(
                    frame_visual_area,
                    (phone[0], phone[1]),
                    (phone[2], phone[3]),
                    (255, 0, 255),
                    2,
                )

        if constants.head_bbox_drawing:
            for head in heads:
                cv2.rectangle(
                    frame_visual_area,
                    (head[0], head[1]),
                    (head[2], head[3]),
                    (255, 0, 255),
                    2,
                )
        action_final_dict = {}
        for i,value in enumerate(action_final):
            action_final_dict[desk_roi[str(i+1)]['desk']] = value

        ## Uncomment to view the frame output ##
        cv2.imshow(self.cam_id, frame_visual_area)
        cv2.waitKey(1)
        for mis_cand in missing_candidates:
            if mis_cand in absent_candidates["card_resetting"]:
                absent_candidates["card_resetting"][mis_cand] = True
        
        absent_candidates_dict_mapping = {}
        for key,value in absent_candidates["card_resetting"].items():
            absent_candidates_dict_mapping[desk_roi[str(key)]['desk']] = value

        return (
            alert_notification,
            action_final_dict,
            absent_candidates_dict_mapping,
            frame_visual_area,
            message,
        )

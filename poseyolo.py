# Qiyas multi cam codes module changed for debugging can't be used for production as a backup saving here

from ultralytics import YOLO
import torch
import time
class ypose:
    def __init__(self, verbose=True):
    # Load the YOLO model
        # self.model = YOLO("models_1/best_v5-1.pt", task="pose")
        self.model = YOLO("models_1/bestv7-2.pt", task="pose")
        # self.model = YOLO("models_1/bestv7-2_half.engine", task="pose")
        self.verbose = verbose 
        empty_input = torch.empty(1, 3, 640, 640).to('cuda')  # Move to GPU if using CUDA
        with torch.no_grad():  # Disable gradient calculation for inference
            _ = self.model(empty_input)
            print("******** Pose Model Loaded Successfully ********")

    def pose(self, frames, batch_metadata):
        batch = len(frames)
        if frames:
            start_time=time.time()
            results = self.model.predict(frames,verbose = self.verbose, batch = batch)
            # Iterate over each detected person and print their keypoints
            for (result,metadata) in zip(results,batch_metadata):
                mylist = []
                if result.keypoints  is not None and result.keypoints.data.shape[1]>0: 
                    for person_idx, person_keypoints in enumerate(result.keypoints.data):
                        person_keypoints = person_keypoints.to('cpu').int().numpy()
                        # Extract keypoints all at once to reduce redundant indexing
                        Cx, Cy = person_keypoints[0, 0], person_keypoints[0, 1]
                        Ax, Ay = person_keypoints[2, 0], person_keypoints[2, 1]
                        Bx, By = person_keypoints[3, 0], person_keypoints[3, 1]
                        Slx, Sly = person_keypoints[4, 0], person_keypoints[4, 1]
                        Elx, Ely = person_keypoints[5, 0], person_keypoints[5, 1]
                        Hlx, Hly = person_keypoints[6, 0], person_keypoints[6, 1]
                        Srx, Sry = person_keypoints[7, 0], person_keypoints[7, 1]
                        Erx, Ery = person_keypoints[8, 0], person_keypoints[8, 1]
                        Hrx, Hry = person_keypoints[9, 0], person_keypoints[9, 1]

                        mylist.append({'Ax': Ax, 'Ay': Ay, 'Bx':Bx, 'By':By, 'Cx':Cx, 'Cy':Cy, 'Slx':Slx, 'Sly':Sly, 'Elx':Elx, 'Ely':Ely, 
                                       'Hlx':Hlx, 'Hly': Hly, 'Srx':Srx, 'Sry':Sry, 'Erx':Erx, 'Ery':Ery, 'Hrx':Hrx, 'Hry':Hry})
                batch_metadata[metadata]['pose_data'] = mylist
            
            # print(f"fps: {1/(time.time()-start_time)}")
        return frames,batch_metadata

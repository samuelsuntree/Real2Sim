import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils.helper1 import read_file
import cv2



start_ToI = 2*60 + 12 #1*60 + 35#13*60+10 #2*60+15 # 13*60+48 #
end_ToI = 2*60 + 48 #1*60 + 56#13*60+24 # 14*60 + 2 #
vid_name = "000000.mp4" 

path_to_vid = r"C:\Users\Thomas\Documents\stage\edge_consistency\input_files\000000.mp4"

vidcap = cv2.VideoCapture(path_to_vid)
if vidcap.isOpened() is False:
    print("Can't load the input video")
    quit()
    
vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ToI*1e3+28)     

os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")


# ======================================
# Individual steps vizualisation

yolo_skin = [0, 131, 255]
applied_skin = [255, 0, 230]
darkgreen_bgr = [0, 100, 0]
bright_red = [20, 20, 255]

os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\tmp")


def all_frames():
    for frame_idx in range((end_ToI-start_ToI)*25):
        ret, frame = vidcap.read()
        try:
            # Reading all steps
            os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\tmp")
            skin_points = read_file("yolo_edge", frame_idx)
            applied_shape = read_file("applied_shape", frame_idx)
            normalized_spine = read_file("normalized_spine", frame_idx)
            extended_spine = read_file("extended_spine", frame_idx)
            spine_estimation = read_file("spine_estimation", frame_idx)
            os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final\vizualisation")
            
            # Link with the video
            
            # Generating frame vizualisation
            
            # No process
            cv2.imwrite("frame{}.jpg".format(frame_idx), frame) 
            
            # YOLO
            cv2.polylines(frame, np.int32([skin_points]), True, yolo_skin, 2)
            cv2.imwrite("YOLO{}.jpg".format(frame_idx), frame) 

            # Spine
            tmp_frame = np.copy(frame)
            cv2.polylines(tmp_frame, np.int32([spine_estimation]), False, darkgreen_bgr, 2)
            """ for point in spine_estimation:
                cv2.circle(tmp_frame, np.int32([point[0], point[1]]), radius=2, color = darkgreen_bgr, thickness=-1) """
            cv2.imwrite("spine_estimation{}.jpg".format(frame_idx), tmp_frame) 

            # Extended spine
            tmp_frame = np.copy(frame)
            cv2.polylines(tmp_frame, np.int32([extended_spine]), False, darkgreen_bgr, 2)
            """ for point in extended_spine:
                cv2.circle(tmp_frame, np.int32([point[0], point[1]]), radius=2, color=(0, 0, 255), thickness=-1) """
            cv2.imwrite("spine_extended{}.jpg".format(frame_idx), tmp_frame)
            
            # Normalized spine + shape
            cv2.polylines(frame, np.int32([applied_shape]), True, applied_skin, 2)
            cv2.polylines(frame, np.int32([normalized_spine]), False, darkgreen_bgr, 2)
            """ for point in normalized_spine:
                cv2.circle(frame, np.int32([point[0], point[1]]), radius=2, color=(0, 0, 255), thickness=-1) """
            cv2.imwrite("applied_shape{}.jpg".format(frame_idx), frame)
            
        except Exception as e:
            #print(e)
            pass 

all_frames()

# ======================================
# Trajectory vizualisation

def trajectories():

    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")
    background = cv2.imread('background.jpg')

    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\tmp")


    spines = []
    for frame_idx in range((end_ToI-start_ToI)*25):
        try:
            # Reading all steps
            skin_points = read_file("yolo_edge", frame_idx)
            #applied_shape = read_file("applied_shape{}".format(frame_idx))
            normalized_spine = read_file("normalized_spine", frame_idx)
            if normalized_spine is not None:
                spines.append(normalized_spine)

        except Exception as e:
            pass
            #print(e)

    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], 
        [192, 192, 192], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128], [0, 128, 128], 
        [0, 0, 128], [255, 165, 0], [255, 105, 180], [75, 0, 130], [240, 230, 140], [173, 216, 230], 
        [255, 20, 147], [138, 43, 226], [144, 238, 144], [210, 105, 30], [70, 130, 180], [128, 128, 128], 
        [255, 182, 193], [139, 69, 19], [255, 222, 173], [0, 100, 0], [47, 79, 79], [135, 206, 250], 
        [0, 191, 255], [127, 255, 212], [216, 191, 216], [255, 160, 122], [255, 69, 0], [153, 50, 204], 
        [139, 0, 139], [255, 99, 71], [189, 183, 107], [60, 179, 113], [255, 248, 220], [176, 196, 222], 
        [220, 20, 60], [233, 150, 122], [218, 112, 214], [152, 251, 152], [244, 164, 96], [72, 61, 139], 
        [245, 222, 179], [199, 21, 133]
    ]

    spines = np.array(spines)
    for trajectory_index in range(50):
        if trajectory_index in [0, 9, 19, 29, 39, 49]: 
            cv2.polylines(background, np.int32([spines[:, trajectory_index, :]]), False, colors[trajectory_index], 2)

    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final\vizualisation")
    cv2.imwrite("Trajectories.jpg", background)

trajectories()

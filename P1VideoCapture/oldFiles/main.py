import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import time
import csv

import concurrent.futures


# ==================
from yolo import apply_yolo
from spine_estimation import get_final_spine
from spine_postprocessing import filtering_single_spine, extend_single_spine_to_skin, fit_single_spine_IoU
from trajectories_processing import filter_trajectories, interpolate_trajectories, vid_creation
from spine_2_lilypad_csv import shape_reconstruction
from utils.helper1 import *
from utils.helper2 import *
#from visualization import show_optimization_step, show_width, draw_velocity

"""
Project Structure:

edge_consistency/
│
├── main.py                 # The main code of the edge detection
├── yolo.py                 # Contains code related to the edge extraction from the video using yolo model
├── spine_estimation.py     # Contains code related to the spine estimation given the edge of the fish
├── spine_postprocessing.py # Contains code related to the spine post processing and width extraction and application on all frames
├── shape_processing.py     # Contains code related to the application of the shape to the spine
├── video_generation.py     # Contains code related to the visualization of the previous results
├── requirements.txt        # Contains package and their version needed
├── utils/
│   ├── __init__.py         # Initializes the utils package.
│   ├── helper1.py          # Contains utility functions, such as add_numbers.
│   └── helper2.py          # Contains additional utility functions.
├── input_files/  
│   ├── video.mp4           # video to process
│   └── trained_model.pt    # Yolo model trained on database
├── output_files/        
│   ├── final/        
|       ├── output{}.mp4    # Original video w/ added edge and spine
|       ├── frame{}.png     # yolo edge, corrected edge and spine for a given frame
|       └── edges{}.npy     # Edge for a given frame
│   └── tmp/
|       ├── yolo_edge{}.npy             # yolo edge for a given frame
|       ├── yolo_edge{}.png             # yolo edge for a given frame
|       ├── spine_estimation{}.npy      # spine estimation for a given frame
|       ├── spine_estimation{}.png      # spine estimation for a given frame
|       ├── filtered_spine{}.npy        # spine estimation for a given frame
|       ├── filtered_spine{}.png        # spine estimation for a given frame
|       ├── corrected_spine{}.npy       # spine corrected estimation for a given frame
|       ├── corrected_spine{}.png       # spine corrected estimation for a given frame
|       ├── applied_shape{}.npy         # spine corrected estimation with added width for a given frame
|       └── applied_shape{}.png         # spine corrected estimation with added width for a given frame
|        


Modules Overview:
- main.py: Orchestrates the application flow.
- module1.py: Defines greeting functionality and uses helper1 utilities.
- module2.py: Defines farewell functionality and uses helper1 utilities.
- utils/helper1.py: Provides utility functions like add_numbers.
"""

# ======================================================
# Path to modify for each PC 

# Project path
#os.chdir("C:/Users/Thomas Omarini/Documents/1 Stage_poisson/machine_perso/edge_consistency")
os.chdir("F:/edge_consistency_v1")

# YOLO model path
model = YOLO("input_files/best.pt")

# # # Video path
# # #path_to_vid = "input_files/000000.mp4"
# # path_to_vid = "input_files/1_1_0137_0154.mp4"


# # Generated data path
# temporary_files_path = r"F:/edge_consistency_v1/output_files/tmp/"
# final_output_path = r"F:/edge_consistency_v1/output_files/final/"


# # Time of interest in the video

# start_ToI = 0#2*60 + 12 #1*60 + 35#13*60+10 #2*60+15 # 13*60+48 #
# end_ToI = 3#2*60 + 48 #1*60 + 56#13*60+24 # 14*60 + 2 #


# 获取命令行传递的参数
if len(sys.argv) < 4:
    print("Please provide the video number, crop number, and video duration as arguments.")
    sys.exit(1)

video_number = sys.argv[1]  # 获取传入的数字参数（视频编号）
crop_number = sys.argv[2]   # 获取传入的数字参数（crop 编号）
video_duration = float(sys.argv[3])  # 获取视频的实际时长

# 参数化输入视频路径
video_folder = f"E:/fishlabel/videos(origin and converted)/output_clip/{video_number}"
video_file = f"{crop_number}.mp4"  # 根据传入的数字参数设置文件名
path_to_vid = os.path.join(video_folder, video_file)

# 生成数据路径
base_output_path = f"F:/edge_consistency_v1/output_files/{video_number}/{crop_number}"
temporary_files_path = os.path.join(base_output_path, "tmp/")
final_output_path = os.path.join(base_output_path, "final/")

# 创建必要的目录
os.makedirs(temporary_files_path, exist_ok=True)
os.makedirs(final_output_path, exist_ok=True)

# ======================================================
# Time of interest in the video

start_ToI = 0  # 开始时间
end_ToI = video_duration  # 结束时间

# # 读取视频时长以确定 end_ToI
# vidcap = cv2.VideoCapture(path_to_vid)
# if vidcap.isOpened() is False:
#     print(f"Can't load the input video {video_file}")
#     quit()

# # 获取视频的帧率和总帧数
# fps = vidcap.get(cv2.CAP_PROP_FPS)
# total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

# # 计算视频时长（秒）
# duration = total_frames / fps
# end_ToI = duration

# print(f"fps = {fps} and total_frames = {total_frames}.")
# print(f"Processing {video_file} from {start_ToI} to {end_ToI} seconds.")






# ======================================================
# Parameters 

# YOLO shape -- spline parameters
num_points = 500 # number of points per skin
s = 100 # smoothness parameter 

# Spine estimation and filtering
spine_seg_length = 10
threshold = 1

# ======================================================
# Parallelization functions 

def overhead_first_step(cpu_count, frames, display = False):
    print("Starting spine estimation")
    futures = []
    results = np.ones([len(frames), 2])*50
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        #print(os.getcwd())
        for i in range(cpu_count):
            chunk = frames[i::cpu_count]
            futures.extend([executor.submit(first_pipeline_step, val) for val in chunk])

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                frame, asymetric_dist, spineLength = future.result()
                results[frame] = asymetric_dist, spineLength
            except:
                pass
    
    if display:
        for frame in frames:
            try:
                skinPoints = read_file("yolo_edge", frame)
                spine = read_file("spine_estimation", frame)
                filtered_spine = read_file("extended_spine", frame)
                
                display_shape(skinPoints)
                display_spine(spine, unicolor = "darkgreen")
                display_spine(filtered_spine, unicolor = "red", marker = ".")
                save_fig("extended_spine{}".format(frame))
            except TypeError:
                plt.clf()
                pass
            
    print("Spine estimation done")
    
    return results

def overhead_third_step(cpu_count, frames, desired_length, shape_width, display = False):
    print("Starting morphology application")
    futures = []
    spines = - np.ones([len(frames), 50, 2])
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        for i in range(cpu_count):
            chunk = frames[i::cpu_count]
            futures.extend([executor.submit(third_pipeline_step, val, desired_length, shape_width) for val in chunk])
    
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                frame, data = future.result()
                spines[frame] = data
            except:
                pass
    
    # ====================
    # Shape + corrected shame vizualisation 
    
    if display:
        for frame in frames:
            try:
                skinPoints = read_file("yolo_edge", frame)
                applied_shape = read_file("applied_shape", frame)
                normalized_spine = read_file("normalized_spine", frame)
                
                display_shape(skinPoints)
                display_shape(applied_shape, color = "lightblue")
                display_spine(normalized_spine, unicolor = "red", marker = ".")
                save_fig("applied_shape{}".format(frame))
            except TypeError:
                plt.clf()
                pass
    print("Morphology application done")
    
    return spines

# ======================================================
# Pipeline steps functions 

def first_pipeline_step(frame):
    try:
        os.chdir(temporary_files_path)
        #print(os.getcwd())
        skinPoints = read_file("yolo_edge", frame)
        spine = get_final_spine(skinPoints, frame, spine_seg_length)
        spine = filtering_single_spine(skinPoints, spine, frame, threshold)
        spine = extend_single_spine_to_skin(skinPoints, spine, frame)
        spine_length = getSpineLength(spine)
        asymetric_dist = get_dif_dist_spine_skin(spine, skinPoints, nb_point_per_spine = 25)
        np.save("extended_spine{}.npy".format(frame), spine)
        return frame, asymetric_dist, spine_length
    
    except Exception as e:
        #print("couldn't generate frame {} | {}".format(frame, e))
        return frame, np.inf

def second_pipeline_step(results, ref_frame = False, display = True):
    print("Starting morphology estimation")
    if ref_frame == False:            
        spine_median_length = 210 #np.median(results[:, 0], axis=0)
        spines_length_dif = (results[:, 0] - spine_median_length)**2
        hyperparameter = 1000
        criteria =  results[:, 1]**2 + hyperparameter*spines_length_dif
        ref_frame = np.argmin(criteria)   
    
    skinPoints = read_file("yolo_edge", ref_frame)
    spine = read_file("extended_spine", ref_frame)

    if spine is None:
        print("faulty reference spine")
        return None, None  
    
    shape_width = getDistSpineSkin(spine, skinPoints)
    desired_length = getSpineLength(spine)
    
    np.save("ref_width{}.npy".format(ref_frame), np.asarray(shape_width))
    np.save("ref_length{}.npy".format(ref_frame), np.asarray(desired_length))
    
    if display:
        display_shape(skinPoints)
        display_spine(spine)
        save_fig("{}{}".format("ref_frame", ref_frame))    
    
    print("Morphology estimation has ended; reference frame is {}; spine length is {}".format(ref_frame, desired_length))
    
    return shape_width, desired_length

def third_pipeline_step(frame, desired_length, shape_width, debug = False):
    try:
        os.chdir(temporary_files_path)
        skinPoints = read_file("yolo_edge", frame)
        spine = read_file("extended_spine", frame)
        spine = fit_single_spine_IoU(skinPoints, spine, frame, desired_length, shape_width)
        if spine is not None:
            return frame, spine
        else:
            if debug is True:
                print("couldn't generate frame {}; {}".format(frame, e))
            else:
                pass
    
    except Exception as e:
        if debug is True:
            print("couldn't generate frame {}; {}".format(frame, e))
        else:
            pass

def fourth_pipeline_step(filename, spines):
    print("Starting spine export")
    # Save the NumPy array to a CSV file
    with open("{}.csv".format(filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Writing the 2D representation of the 3D array to the CSV file
        for frame_index, frame_data in enumerate(spines):
            # Filter out non-existing spines 
            if (frame_data[0][0] != -1):
                text = []
                for row in frame_data:
                    text += list(row)          
                writer.writerow([frame_index] + text)
    print("Spine export done")

def fifth_pipeline_step(frames, spines, spine_name, display_traj = False, display_vid = False, path_to_vid = None, shape_width = None, output_vid_name = None):
    print("Starting filtering/interpolating trajectories")
    # parameters 
    window_size = 5
    s = 5
    
    # filtering + interpolating
    filtered_trajectories, filtered_frames = filter_trajectories(spines, frames, window_size = window_size)
    resampled_trajectories = interpolate_trajectories(filtered_trajectories, filtered_frames, s = s)

    # Saving spine
    filtered_filename = "{}.csv".format(spine_name)
    with open(filtered_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, spine in enumerate(zip(resampled_trajectories)):
            text = []
            for row in spine:
                point = []
                for coordonate in row:
                    point += list(coordonate)      
                text += point        
            writer.writerow(["{}".format(i)] + text)

    if display_traj:
        # Raw trajectories 
        plt.plot(spines[:, -1, 0], spines[:, -1, 1], marker=".", linestyle="--", label="Head", color = "coral", ms = 10, linewidth=2)
        plt.plot(spines[:, 0, 0], spines[:, 0, 1], marker=".", linestyle="--", label = "Tail", color = "limegreen", ms = 10, linewidth=2)
        plt.plot(spines[:, -10, 0], spines[:, -10, 1], marker=".", linestyle="--", label="Body", color = "royalblue", ms = 10, linewidth=2)
        
        # Filtered trajectories
        plt.plot(filtered_trajectories[:, -1, 0], filtered_trajectories[:, -1, 1], marker=".", linestyle="dotted", label="Filtered head", color = "coral", ms = 10, linewidth=2)
        plt.plot(filtered_trajectories[:, 0, 0], filtered_trajectories[:, 0, 1], marker=".", linestyle="dotted", label = "Filtered tail", color = "limegreen", ms = 10, linewidth=2)
        plt.plot(filtered_trajectories[:, -10, 0], filtered_trajectories[:, -10, 1], marker=".", linestyle="dotted", label="Filtered body", color = "royalblue", ms = 10, linewidth=2)

        # interpolated trajectories
        plt.plot(resampled_trajectories[:, -1, 0], resampled_trajectories[:, -1, 1], marker=".", linestyle="-", label="Filtered + interpolated head", color = "red", ms = 10, linewidth=2)
        plt.plot(resampled_trajectories[:, 0, 0], resampled_trajectories[:, 0, 1], marker=".", linestyle="-", label = "Filtered + interpolated tail", color = "green", ms = 10, linewidth=2)
        plt.plot(resampled_trajectories[:, -10, 0], resampled_trajectories[:, -10, 1], marker=".", linestyle="-", label="Filtered + interpolated body", color = "blue", ms = 10, linewidth=2)
        plt.legend(fontsize=12)
        plt.savefig("trajectories{}".format(50))
        
    if display_vid:
        vid_creation(shape_width, path_to_vid, output_vid_name, resampled_trajectories)
    
    print("Filtering/interpolating trajectories done")

def sixth_pipeline_step(filename, shape_width):
    print("Starting export for lilypad")
    shape_reconstruction(filename, shape_width)
    print("Export for lilypad done")

# ======================================================
# Others

def delete_files_except_patterns(directory):
    # Define the regular expressions for the patterns
    yolo_edge_pattern = re.compile(r'yolo_edge\d+\.npy')
    faulty_polygon_pattern = re.compile(r'faulty_polygon\d+\.png')
    ref_spine = re.compile(r'ref_spine\d+\.npy')

    # List all files in the directory
    for filename in os.listdir(directory):
        # Full path to the file
        file_path = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if the filename matches any of the patterns
        if not (yolo_edge_pattern.match(filename) or faulty_polygon_pattern.match(filename) or ref_spine.match(filename)):
            try:
                # Delete the file
                os.remove(file_path)
                #print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

if __name__ == '__main__':
    
    
    
    
    # Reading video
    vidcap = cv2.VideoCapture(path_to_vid)
    if vidcap.isOpened() is False:
        print("Can't load the input video")
        quit()
    
    cpu_count = os.cpu_count()
    
    # Compute the yolo frames
    print("Starting video processing...")
    os.chdir(temporary_files_path)
    
    # =====================
    # Step 0 - YOLO application on video    
    apply_yolo(model, vidcap, start_ToI, end_ToI, s, num_points)    
    
    for test in range(1):
        # Clean the tmp folder to ensure each generation is independant
        delete_files_except_patterns(temporary_files_path)
        
        # =====================
        # First step - Spines generation + processing
        #print(os.getcwd())
        frames = get_relevant_frames("yolo_edge") # Get all yolo detection
        total_frame_processed = len(frames)       # Count them
        #print(os.getcwd())
        results = overhead_first_step(cpu_count, frames, display = False) # Process them
        
        # =====================
        # Second step - Morphology estimation   
        os.chdir(temporary_files_path)  
        shape_width, desired_length = second_pipeline_step(results)
        
        os.chdir(temporary_files_path)
        
        #shape_width, desired_length = read_file("ref_width", 706), read_file("ref_length", 706)
        
        cpt = 0
        while desired_length is None:
            total_frame_processed = len(frames)       # Count them
            results = overhead_first_step(cpu_count, frames, display = False) # Process them
            shape_width, desired_length = second_pipeline_step(results)
            print("Faulty morphology, regenerating spines")
            cpt += 1
            if cpt == 10:
                print("Repeted unsuccessful morphology")
                break
    
        # =====================
        # Third step - Morphology application       
        frames = get_relevant_frames("extended_spine") # Get all spines 
        spines = overhead_third_step(cpu_count, frames, desired_length, shape_width, display = False) # Process them
        
        # =====================
        # Fourth step - Data saving        
        os.chdir(final_output_path) # Go to final folder to save data
        fourth_pipeline_step("spines", spines) # Save data
        os.chdir(temporary_files_path) # Go back to temporary folder 
        
        # =====================
        # Fifth step - Data filter and interpolation
        os.chdir(final_output_path) # Go to final folder to save data
        frames, spines = load_frames_spines_from_csv("spines")
        fifth_pipeline_step(frames, spines, "spines_interpolated")
        
        # =====================
        # Sixth step - Generate csv for lilypad
        os.chdir(final_output_path)
        sixth_pipeline_step("spines_interpolated", shape_width)
        

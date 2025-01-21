import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from utils.helper1 import read_file, load_frames_spines_from_csv
from utils.helper2 import remove_duplicates
from shape_processing import get_estimated_shape
import cv2

def filter_trajectories(spines, frames, window_size=5):
    nb_vertebra = spines.shape[1]
    
    # Initialize a list to collect the filtered segments and corresponding frames
    filtered_segments = []
    filtered_frame_segments = []

    # Identify consecutive segments
    start_idx = 0
    for i in range(1, len(frames)):
        if frames[i] != frames[i-1] + 1:
            end_idx = i
            filtered_segments.append(spines[start_idx:end_idx])
            filtered_frame_segments.append(frames[start_idx:end_idx])
            start_idx = i
    # Add the last segment
    filtered_segments.append(spines[start_idx:])
    filtered_frame_segments.append(frames[start_idx:])
    
    # Initialize an empty list to store the filtered trajectories
    filtered_trajectories_list = []
    filtered_frames_list = []
    
    for segment_spines, segment_frames in zip(filtered_segments, filtered_frame_segments):
        if len(segment_frames) < window_size:
            # Append the unfiltered segment if it's too small
            filtered_trajectories_list.append(segment_spines)
            filtered_frames_list.append(segment_frames)
        else:
            segment_filtered = np.zeros([len(segment_spines) - window_size + 1, nb_vertebra, 2])
            
            for vertebra in range(nb_vertebra):
                sequence = segment_spines[:, vertebra, :]
                
                # Apply moving average with valid mode
                moving_avg_x = np.convolve(sequence[:, 0], np.ones(window_size) / window_size, mode='valid')
                moving_avg_y = np.convolve(sequence[:, 1], np.ones(window_size) / window_size, mode='valid')
                
                # Combine the averaged components back into 2D points
                segment_filtered[:, vertebra, 0] = moving_avg_x
                segment_filtered[:, vertebra, 1] = moving_avg_y
            
            # Append the filtered segment to the list of filtered trajectories
            filtered_trajectories_list.append(segment_filtered)
            filtered_frames_list.append(segment_frames[int(window_size/2):-int(window_size/2)])
    
    # Concatenate all filtered segments along the first axis
    filtered_trajectories = np.concatenate(filtered_trajectories_list, axis=0)
    filtered_frames = np.concatenate(filtered_frames_list, axis=0)
    
    return filtered_trajectories, filtered_frames

def interpolate_trajectories(filtered_trajectories, filtered_frames, s=0):
    nb_vertebra = filtered_trajectories.shape[1]
    
    resampled_trajectories = np.zeros([max(filtered_frames) - min(filtered_frames) + 1, nb_vertebra, 2])
    
    for vertebra in range(nb_vertebra):
        sequence = filtered_trajectories[:, vertebra, :]
        
        # Existing data points
        x = sequence[:, 0]
        y = sequence[:, 1]

        # Remove duplicates
        x, y = remove_duplicates(x, y)
        
        # Parameterize the existing data points
        tck, u = splprep([x, y], u=filtered_frames, s=s, per=False, k=3)

        resampled_points = []
        for frame in range(min(filtered_frames), max(filtered_frames) + 1):
            resampled_points.append(splev(frame, tck))
        
        resampled_trajectories[:, vertebra, :] = np.array(resampled_points)

    return resampled_trajectories

def vid_creation(shape_width, path_to_vid, output_vid_name, resampled_trajectories):
        print("generating video")
                
        vidcap = cv2.VideoCapture(path_to_vid)
        if vidcap.isOpened() is False:
            print("Can't load the input video")
            quit()
            
        vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ToI*1e3)     
        
        trajectory_data = resampled_trajectories
        num_frames = np.shape(resampled_trajectories)[0] 
        num_trajectories = np.shape(resampled_trajectories)[1]      
        frames = []
        
        
        #TODO: find the offset - ie the frame at which the fish start being processed succesfully
        offset = 28 + int(window_size/2)+1
        
        # run the video while the fish cannot be processed
        for _ in range(offset):
            _, frame = vidcap.read()
            height, width, _ = frame.shape
        
        
        # Process frames
        for frame_idx in range(num_frames):
            # Copy the background image
            #frame = background.copy()
            
            _, frame = vidcap.read()           
            
            # Add yolo
            try:
                #os.chdir(output_files\tmp)     
                skin_points = read_file("yolo_edge", frame_idx+28)         
                cv2.polylines(frame, np.int32([skin_points]), True, [0, 131, 255], 2)
                #display_shape(skin_shape)
                #os.chdir(output_files\final)     
            except:
                pass
            
            # Add raw morphology
            try:
                #os.chdir(output_files\tmp)          
                skin_points = read_file("applied_shape", frame_idx+28)         
                cv2.polylines(frame, np.int32([skin_points]), True, [255, 0, 230], 2)
                #display_shape(skin_shape)
                #os.chdir(output_files\final)
            except:
                pass
            
            # Add raw morphology spine  
            try:
                #os.chdir(output_files\tmp)          
                spine = np.array(read_file("normalized_spine", frame_idx+28))     
                #display_shape(skin_shape)
                #os.chdir(output_files\final))
                for point_index in range(len(spine)):
                    point = tuple(map(int, spine[point_index, :]))
                    color = [0, 100, 0] #colors[point_index]
                    if (point_index == 0):
                        cv2.circle(frame, point, radius=4, color=[0, 255, 0], thickness=-1)
                    elif (point_index == len(spine)-1):
                        cv2.circle(frame, point, radius=4, color=[0, 0, 255], thickness=-1)
                    
                    elif (point_index == len(spine)-20):
                        cv2.circle(frame, point, radius=4, color=[255, 0, 0], thickness=-1)
                    else:
                        cv2.circle(frame, point, radius=2, color=color, thickness=-1)
            except:
                pass
            
            # Add filtered + interpolated shape
            _, estimated_shape = get_estimated_shape(trajectory_data[frame_idx, :, :], shape_width)
            cv2.polylines(frame, np.int32([estimated_shape]), True, (255,255,0), 2)
            # Draw each trajectory point
            for traj_idx in range(num_trajectories):
                point = tuple(map(int, trajectory_data[frame_idx, traj_idx, :]))
                color = colors[traj_idx]
                color = [0, 100, 0]
                cv2.circle(frame, point, radius=2, color=color, thickness=-1)
            
            # Add the frame to the list
            frames.append(frame)
            
        vidcap.release()
        
        # Create video
        out = cv2.VideoWriter('{}.avi'.format(output_vid_name), cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        
        print("output video generated")

if __name__ == "__main__":
    
    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")

    start_ToI = 2*60 + 12 #1*60 + 35#13*60+10 #2*60+15 # 13*60+48 #
    end_ToI = 2*60 + 48 #1*60 + 56#13*60+24 # 14*60 + 2 #
    vid_name = "000000.mp4" + "{}".format(0)
    fps = 25
    frames, spines = load_frames_spines_from_csv(vid_name, start_ToI, end_ToI)
    
    window_size = 5
    
    # ==================================================
    
    plt.plot(spines[:, -1, 0], spines[:, -1, 1], marker=".", linestyle="--", label="Head", color = "coral", ms = 16, linewidth=4)
    plt.plot(spines[:, 0, 0], spines[:, 0, 1], marker=".", linestyle="--", label = "Tail", color = "limegreen", ms = 16, linewidth=4)
    plt.plot(spines[:, -10, 0], spines[:, -10, 1], marker=".", linestyle="--", label="Body", color = "royalblue", ms = 16, linewidth=2)

    # =================================
    nb_resampled_points = len(np.arange(frames[0], frames[-1] + 1))
    filtered_trajectories, filtered_frames  = filter_trajectories(spines, frames)
    resampled_trajectories = interpolate_trajectories(filtered_trajectories, filtered_frames, s = 5)
    
    # =================================
    
    """ plt.plot(filtered_trajectories[:, -1, 0], filtered_trajectories[:, -1, 1], marker=".", linestyle="dotted", label="Filtered head", color = "coral", ms = 10, linewidth=2)
    plt.plot(filtered_trajectories[:, 0, 0], filtered_trajectories[:, 0, 1], marker=".", linestyle="dotted", label = "Filtered tail", color = "limegreen", ms = 10, linewidth=2)
    plt.plot(filtered_trajectories[:, -10, 0], filtered_trajectories[:, -10, 1], marker=".", linestyle="dotted", label="Filtered body", color = "royalblue", ms = 10, linewidth=2) """
    
    # =================================
    
    #adjusted_resampled_trajectories = np.array(adjusted_resampled_trajectories)
    
    # Write the filtered spines to a new CSV file
    vid_name = "smoothed_{}".format(0)
    filtered_filename = "smoothed{}.csv".format(vid_name, start_ToI, end_ToI)
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
                

        plt.plot(resampled_trajectories[:, -1, 0], resampled_trajectories[:, -1, 1], marker=".", linestyle="-", label="Filtered + interpolated head", color = "red", ms = 16, linewidth=4)
        plt.plot(resampled_trajectories[:, 0, 0], resampled_trajectories[:, 0, 1], marker=".", linestyle="-", label = "Filtered + interpolated tail", color = "green", ms = 16, linewidth=4)
        plt.plot(resampled_trajectories[:, -10, 0], resampled_trajectories[:, -10, 1], marker=".", linestyle="-", label="Filtered + interpolated body", color = "blue", ms = 16, linewidth=4)
        
        filtered_frames = frames[int(5/2):-int(5/2)]
        missing_frames = []
        for i in range(1, len(filtered_frames)):        
            if (filtered_frames[i] != filtered_frames[i-1] + 1):
                missing_frames.append(i)
        
        """ plt.plot(resampled_trajectories[missing_frames, -1, 0], resampled_trajectories[missing_frames, -1, 1], marker="D", linestyle="", label="Filtered + interpolated head", color = "red", ms = 10, linewidth=2)
        plt.plot(resampled_trajectories[missing_frames, 0, 0], resampled_trajectories[missing_frames, 0, 1], marker="D", linestyle="", label = "Filtered + interpolated tail", color = "green", ms = 10, linewidth=2)
        plt.plot(resampled_trajectories[missing_frames, -10, 0], resampled_trajectories[missing_frames, -10, 1], marker="D", linestyle="", label="Filtered + interpolated body", color = "blue", ms = 10, linewidth=2) """
        
        """ plt.plot(filtered_trajectories[:, -1, 0], filtered_trajectories[:, -1, 1], marker="*", linestyle="-", label="Filtered + interpolated head", color = "red", ms = 10, linewidth=2)
        plt.plot(filtered_trajectories[:, 0, 0], filtered_trajectories[:, 0, 1], marker="*", linestyle="-", label = "Filtered + interpolated tail", color = "green", ms = 10, linewidth=2)
        plt.plot(filtered_trajectories[:, -10, 0], filtered_trajectories[:, -10, 1], marker="*", linestyle="-", label="Filtered + interpolated body", color = "blue", ms = 10, linewidth=2) """
        
        
        """ plt.figure()
        spine_lengths = []        
        for spine in resampled_trajectories[:, :, :]:
            #print(getSpineLength(spine))
            spine_lengths.append(getSpineLength(spine))
        
        plt.plot(np.arange(0, len(spine_lengths), 1), np.sort(spine_lengths), marker=".", linestyle="", color = "r", ms = 10, label = "Filtered + interpolated spine")
        
        spine_lengths = []   
        for spine in spines[:, :, :]:
            #print(getSpineLength(spine))
            spine_lengths.append(getSpineLength(spine))
        
        plt.plot(np.arange(0, len(spine_lengths), 1), np.sort(spine_lengths), marker=".", linestyle="", color = "darkgreen", ms = 10, label = "initial spine")
        plt.ylabel("Spine length [px]", fontsize=18)
        plt.tick_params(
            axis = "y",
            labelsize = 16)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.xlabel("Frames sorted by length", fontsize=18)
        plt.legend(fontsize=16)
        plt.show() """

        plt.legend(fontsize=16)
        plt.savefig("trajectories{}".format(50))
        plt.show()

        #quit()

        print("generating video")
        
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")
        dist_spine_skin = np.load("ref_width664.npy")
        
        # Load the background image
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")
        background = cv2.imread('background.jpg')
        height, width, _ = background.shape
        
        path_to_vid = r"C:\Users\Thomas\Documents\stage\edge_consistency\input_files\000000.mp4"
        
        vidcap = cv2.VideoCapture(path_to_vid)
        if vidcap.isOpened() is False:
            print("Can't load the input video")
            quit()
            
        vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ToI*1e3)     

        trajectory_data = resampled_trajectories
        num_frames = np.shape(resampled_trajectories)[0] 
        num_trajectories = np.shape(resampled_trajectories)[1]      

        # Assign a unique color to each trajectory
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_trajectories)]

        # Draw trajectory points on each frame
        frames = []
        missing_frame = 0
        offset = 28 + int(window_size/2)+1
        for _ in range(offset):
            ret, frame = vidcap.read()
        
        
        for frame_idx in range(num_frames):
            # Copy the background image
            #frame = background.copy()
            
            ret, frame = vidcap.read()
            
            def idk():
                try:
                    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\tmp")     
                    skin_points = read_file("yolo_edge", frame_idx+28)         
                    cv2.polylines(frame, np.int32([skin_points]), True, [0, 131, 255], 2)
                    #display_shape(skin_shape)
                    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")
                except:
                    pass
                try:
                    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\tmp")     
                    skin_points = read_file("applied_shape", frame_idx+28)         
                    cv2.polylines(frame, np.int32([skin_points]), True, [255, 0, 230], 2)
                    #display_shape(skin_shape)
                    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")
                except:
                    missing_frame += 1
                
                try:
                    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\tmp")     
                    spine = np.array(read_file("normalized_spine", frame_idx+28))     
                    #display_shape(skin_shape)
                    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")
                    for point_index in range(len(spine)):
                        point = tuple(map(int, spine[point_index, :]))
                        color = [0, 100, 0] #colors[point_index]
                        if (point_index == 0):
                            cv2.circle(frame, point, radius=4, color=[0, 255, 0], thickness=-1)
                        elif (point_index == len(spine)-1):
                            cv2.circle(frame, point, radius=4, color=[0, 0, 255], thickness=-1)
                        
                        elif (point_index == len(spine)-20):
                            cv2.circle(frame, point, radius=4, color=[255, 0, 0], thickness=-1)
                        else:
                            cv2.circle(frame, point, radius=2, color=color, thickness=-1)
                except:
                    pass
            
            _, estimated_shape = get_estimated_shape(trajectory_data[frame_idx, :, :], dist_spine_skin)
            cv2.polylines(frame, np.int32([estimated_shape]), True, (255,255,0), 2)
            # Draw each trajectory point
            for traj_idx in range(num_trajectories):
                point = tuple(map(int, trajectory_data[frame_idx, traj_idx, :]))
                color = colors[traj_idx]
                color = [0, 100, 0]
                cv2.circle(frame, point, radius=2, color=color, thickness=-1)
            
            try:
                os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\tmp")     
                skin_points = read_file("yolo_edge", frame_idx+offset)         
                cv2.polylines(frame, np.int32([skin_points]), True, [0, 131, 255], 2)
                #display_shape(skin_shape)
                os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\final")
            except:
                pass
                
            # Add the frame to the list
            frames.append(frame)


        vidcap.release()
        
        print(missing_frame)
        # Compile frames into a video
        out = cv2.VideoWriter('output{}.avi'.format(50), cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        
        plt.legend()
        plt.savefig("trajectories{}".format(50))
        plt.show()
    
    """ plt.plot(np.arange(0, len(spine_lengths), 1), np.sort(spine_lengths), "r*", ms = 10)
    plt.show() """
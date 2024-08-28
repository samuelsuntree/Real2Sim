import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import matplotlib as mpl
import statistics
import glob
from shapely.geometry import Polygon

#from frechetdist import frdist
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize

from spine_estimation import get_final_spine, direction_uniformization
from utils.helper1 import *
from utils.helper2 import * #findSpineIntersectionsWithSkin, get_intersection_rib_skin, get_estimated_skin_point, getIntersectionsWithSkin, orthogonalLine, findIntersectionsWithSkin,         distQuad, computeDirectoryCoefficient, computeLineIntersection, get_head, getSpineLength, getInterpolationU
from shape_processing import get_estimated_shape
from spine_estimation import get_final_spine


# ========================

def theta_filtering(spine, theta_threshold):
    spine_head = spine[int(2*len(spine)/3)-5:, :]

    deltas = spine_head[1:] - spine_head[:-1]
    thetas = np.arctan2(deltas[:, 1], deltas[:, 0])
    thetas_diffs = np.abs(np.diff(thetas))
    angle_diffs = np.minimum(thetas_diffs, 2 * np.pi - thetas_diffs)
    
    new_spine_head = [spine_head[0]]
    for i in range(1, len(angle_diffs)):
        #print(angle_diffs[i-1], theta_threshold)
        if angle_diffs[i-1] < theta_threshold:
            new_spine_head.append(spine_head[i].tolist())
            
    new_spine_head.append(spine_head[-1])     
    
    new_spine = np.vstack((spine[:int(2*len(spine)/3)-5], np.array(new_spine_head)))
    
    return new_spine


def filtering_single_spine(skinPoints, spine, frame, threshold):
    # Filter out spines that are too small
    if len(spine) < 20:
        return None
    # Delete unwanted initial points
    theta_threshold = np.pi/28
    spine = theta_filtering(spine, theta_threshold)   
    
    # Fit polynomial
    # Get the spine point around the tail
    x = spine[int(2*len(spine)/3):, 0]
    y = spine[int(2*len(spine)/3):, 1]
    
    degree = 2
    coeffs = np.polyfit(x, y, degree)

    x_fit = np.linspace(min(x), max(x), len(x))
    y_fit = np.polyval(coeffs, x_fit)
    
    # Error between the fit and initials spine points
    y_pred = np.polyval(coeffs, x)    
    mse = (y - y_pred)**2
    rmse = np.sqrt(mse)
    index_max_rmse = (np.argsort(rmse))[-1]
    biggest_error_index = int(2*len(spine)/3) + index_max_rmse
    
    if ((biggest_error_index == len(spine)-1) and (rmse[index_max_rmse] > threshold)):
        #plt.plot(spine[biggest_error_index][0], spine[biggest_error_index][1], "bD")
        spine = spine[:-2]
    elif ((biggest_error_index == len(spine)-2) and (rmse[index_max_rmse] > threshold)):
        #plt.plot(spine[biggest_error_index][0], spine[biggest_error_index][1], "bD")
        spine = spine[:-3]    
    
    
    # Initial spine filtering
    
    """  plt.figure()
    plt.plot(np.arange(int(2*len(spine)/3), len(spine)), rmse, color = "darkgreen", marker = "+") """
    
    """ plt.plot(x_fit, y_fit, color='red', marker = "+")
    display_shape(skinPoints)
    display_discret_spine(spine)
    plt.plot(spine[int(2*len(spine)/3)][0], spine[int(2*len(spine)/3)][1], "rD") """
    #save_fig("filtered_spine{}".format(frame))
    #np.save("filtered_spine{}.npy".format(frame), np.asarray(spine))
    
    return spine

def filtering_spine(threshold):
    print("starting filtering spines...")
    # Get all npy files
    npy_files = glob.glob('*.npy')
    
    # Filter only the relevant npy files
    pattern = re.compile(r'spine_estimation\d+\.npy')
    useful_npy_files = [filename for filename in npy_files if pattern.match(filename)]
    
    # Get the number associated to the frame
    frames = []
    for file_name in useful_npy_files:
        frames.append(int(os.path.splitext(file_name)[0].replace("spine_estimation", "")))

    # Process all frames
    nb_frames = len(frames)
    for i, frame in enumerate(frames):
        print("frame: {} | {:.2f} %".format(frame, i/nb_frames*100))
        filtering_single_spine(skinPoints, spine, frame, threshold)
        try:
            pass
        except Exception :
            print("problem frame {}".format(frame))
        
    print("done")

# ========================

def get_reference_spine():
    frames = get_relevant_frames("extended_spine")
    frames = np.sort(frames)
    asymetric_dist = np.zeros([len(frames)])
    for i, frame in enumerate(frames):
        skinPoints = read_file("yolo_edge", frame)
        spine = read_file("extended_spine", frame)
        """ display_shape(skinPoints)
        display_spine(spine)        
        save_fig("extended_spine{}".format(frame)) """
        asymetric_dist[i] = get_dif_dist_spine_skin(spine, skinPoints, nb_point_per_spine = 100)

    
    ref_frame = np.argmin(asymetric_dist)
    """ plt.plot(frames, asymetric_dist, marker = "+", color = "gold", linestyle = "dotted")
    plt.plot(frames[ref_frame], asymetric_dist[ref_frame], marker = "D", color = "red")
    
    plt.figure()
    skinPoints = read_file("yolo_edge", ref_frame)
    spine = read_file("extended_spine", ref_frame)
    display_shape(skinPoints)
    display_spine(spine)
    display_fig() """
    return ref_frame


# ========================

def correct_single_spine(frame, lengthAvg, spineLength):
    skinPoints = read_file("yolo_edge", frame)
    spine = read_file("spine_estimation", frame)
    length = getSpineLength(spine)
    
    # Regenerate spine if the estimation is too bad
    watchdog = 0
    while abs(length-lengthAvg) > 0.2*lengthAvg:
        if watchdog == 0:
            print("Regenerating frame {}".format(frame))
        watchdog += 1
        spine = get_final_spine(frame, spineLength)
        if spine is not None:
            length = getSpineLength(spine)
        else:
            length = np.inf
        if watchdog == 10:
            print("couldn't save frame {}".format(frame))
            return None
    
    tailIndex = get_head(spine, skinPoints)
    # Correct the spine length
    if tailIndex == -1:
        if length > lengthAvg:
            # Calculate excess length
            delta = length - lengthAvg
            i = 0
            previousDelta = distQuad(spine[0], spine[1])
            # Delete spine points starting from tail while the spine length is more than it should be
            while delta > previousDelta:
                previousDelta = distQuad(spine[i+1], spine[i])
                delta -= previousDelta
                i += 1
            # Filter last tail point
            spine = spine[:-2]
            # Filter out oscillating head        
            spine = spine[i:]            
            previousDelta = distQuad(spine[0], spine[1])
            spine[0] = extendSegment(spine[1], spine[0], previousDelta - delta)

        elif length < lengthAvg:
            # Calculate the length deficit and extend the first segment
            delta = lengthAvg - length
            previousDelta = distQuad(spine[0], spine[1])
            spine[0] = extendSegment(spine[1], spine[0], previousDelta + delta)
        
        # Filter last tail point
        spine = spine[1:] 
        # Filter out oscillating head        
        spine = spine[:-5].tolist()
        spine.append(extendSegment(spine[-2], spine[-1], spineLength*3).tolist())
        spine = np.array(spine)
    else:
        if length > lengthAvg:
            # Calculate excess length
            delta = length - lengthAvg
            i = 0
            previousDelta = distQuad(spine[-1], spine[-2])

            # Delete spine points starting from tail while the spine length is more than it should be
            while delta > previousDelta:
                previousDelta = distQuad(spine[-2-i], spine[-3-i])
                delta -= previousDelta
                i += 1

            # Update spine and extend the last segment
            spine = spine[:-i-1]
            previousDelta = distQuad(spine[-1], spine[-2])
            spine[-1] = extendSegment(spine[-2], spine[-1], previousDelta - delta)

        elif length < lengthAvg:
            # Calculate the length deficit and extend the last segment
            delta = lengthAvg - length
            previousDelta = distQuad(spine[-1], spine[-2])
            spine[-1] = extendSegment(spine[-2], spine[-1], previousDelta + delta)
            
        # Shorten the spine to filter out oscillating head
        spine = spine[4:].tolist()
        spine.reverse()
        spine.append(extendSegment(spine[-2], spine[-1], spineLength*3).tolist())
        spine.reverse()
        spine = np.array(spine)

    np.save("corrected_spine{}.npy".format(frame), np.asarray(spine))

    """ display_shape(skinPoints)
    display_discret_spine(spine)  """   
    save_fig("corrected_spine{}".format(frame))

def correcting_spine(frameAvg, spineLength):
    print("starting correcting spines...")
    # Get all npy files
    npy_files = glob.glob('*.npy')
    
    # Filter only the relevant npy files
    pattern = re.compile(r'spine_estimation\d+\.npy')
    useful_npy_files = [filename for filename in npy_files if pattern.match(filename)]
    
    # Get the number associated to the frame
    frames = []
    for file_name in useful_npy_files:
        frames.append(int(os.path.splitext(file_name)[0].replace("spine_estimation", "")))
    
    spine = read_file("spine_estimation", frameAvg)
    lengthAvg = getSpineLength(spine)

    nb_frames = len(frames)
    for i, frame in enumerate(frames):
        print("frame: {} | {:.2f} %".format(frame, i/nb_frames*100))
        correct_single_spine(frame, lengthAvg, spineLength)
        
    print("done")

# ========================

def overextend_single_spine_to_skin(spine, skinPoints, frame):

    spine_length = distQuad(spine[2], spine[3])
    # Ensure modifications don't compromise spine while computing on it
    new_spine = np.copy(spine)
    
    candidate = get_closest_intersection_spine_skin(spine[-2], spine[-1], skinPoints)
    candidate = extendSegment(spine[-2], candidate, distQuad(candidate, spine[-2]) + 20*spine_length)
    new_spine[-1] = candidate
    
    candidate = get_closest_intersection_spine_skin(spine[1], spine[0], skinPoints)  
    candidate = extendSegment(spine[1], candidate, distQuad(candidate, spine[1]) + 20*spine_length)
    new_spine[0] = candidate
    
    """ display_shape(skinPoints)
    display_discret_spine(new_spine, unicolor = "darkgreen")
    display_discret_spine(spine, unicolor = "red", marker=".")
    display_fig()
    save_fig("extended_spine{}".format(frame)) """
    #np.save("extended_spine{}.npy".format(frame), np.asarray(new_spine))
    
    return new_spine

def extend_single_spine_to_skin(skinPoints, spine, frame):

    # Ensure modifications don't compromise spine while computing on it
    new_spine = np.copy(spine)
    
    candidate = get_closest_intersection_spine_skin(spine[-2], spine[-1], skinPoints)
    new_spine[-1] = candidate
    
    candidate = get_closest_intersection_spine_skin(spine[1], spine[0], skinPoints)  
    new_spine[0] = candidate
    
    """ display_shape(skinPoints)
    display_discret_spine(new_spine)
    display_discret_spine(spine)
    display_fig() """
    #np.save("extended_spine{}.npy".format(frame), np.asarray(new_spine))
    
    return new_spine

def extend_spine_to_skin():
    print("starting expending spines...")
    # Get all npy files
    npy_files = glob.glob('*.npy')
    
    # Filter only the relevant npy files
    pattern = re.compile(r'filtered_spine\d+\.npy')
    useful_npy_files = [filename for filename in npy_files if pattern.match(filename)]
    
    # Get the number associated to the frame
    frames = []
    for file_name in useful_npy_files:
        frames.append(int(os.path.splitext(file_name)[0].replace("filtered_spine", "")))
    
    # Process all frames
    nb_frames = len(frames)
    for i, frame in enumerate(frames):
        print("frame: {} | {:.2f} %".format(frame, i/nb_frames*100))
        try :
            extend_single_spine_to_skin(frame)
        except Exception:
            print("problem frame {}".format(frame))
        
    print("done")

def calculate_iou(polygon1, polygon2, spine, frame):
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    try:
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
    except Exception as e:
        #print(e)
        fig, ax = plt.subplots()

        # Plot the polygons
        x, y = poly1.exterior.xy
        ax.plot(x, y, color="blue", alpha=0.7, linewidth=2, solid_capstyle='round', zorder=2)
        x, y = poly2.exterior.xy
        ax.plot(x, y, color="red", alpha=0.7, linewidth=2, solid_capstyle='round', zorder=2)
        
        # Set the aspect of the plot to be equal
        ax.set_aspect('equal', 'box')
        
        for polygon, color in zip([poly1, poly2], ['blue', 'green']):
            for coord in polygon.exterior.coords:
                ax.plot(*coord, 'o', color=color)
                
        # Display the plot
        display_spine(spine)
        save_fig("faulty_polygon{}".format(frame))
        
        return -np.inf
    
    return intersection_area / union_area

def resample_spine(spine, nb_points):
    u_new = np.linspace(0, 1, nb_points)
    tck = getInterpolationU(spine)
    
    spine = np.array([splev(u, tck) for u in u_new])
    
    return spine 

def get_spine_candidate(spine, start_index, desired_length):
    #plt.plot(spine[start_index][0], spine[start_index][1], "D", color = "darkgreen")
    length = getSpineLength(spine)
    spine_validity = False
    if desired_length > length:
        print("extented spine is too short -- probably faulty spine")
        return None
    else:
        dist = 0
        stop_index = start_index
        while ((dist < desired_length) and (stop_index+1 <= len(spine)-1)):
            dist_tmp = dist + distQuad(spine[stop_index], spine[stop_index+1])
            if dist_tmp < desired_length:
                stop_index += 1
                dist = dist_tmp
                spine_validity = False
            else:
                spine_validity = True
                spine_candidate = spine[start_index:stop_index]
                break
            
        remaining_dist = desired_length - dist
        
        #plt.plot(spine[stop_index][0], spine[stop_index][1], "D", color = "gold")
        
        if spine_validity:
            last_point = extendSegment(spine[stop_index], spine[stop_index+1], remaining_dist)
            spine_candidate = np.append(spine_candidate, [last_point], axis = 0)
            #print(start_index, stop_index)
        else:
            spine_candidate = None
        
    return spine_candidate

def fit_single_spine_IoU(skinPoints, spine, frame, desired_length, shape_distances):
    spine = np.array(direction_uniformization(spine, skinPoints))
    new_spine = overextend_single_spine_to_skin(spine, skinPoints, frame)
    resampled_spine = resample_spine(new_spine, 1000)

    """  display_shape(skinPoints)
    display_spine(resampled_spine, "*", unicolor = "red")
    display_fig() """
    cpt = 0
    cpt_maxi = -1
    maxi = 0
    while new_spine is not None:
        new_spine = get_spine_candidate(resampled_spine, cpt, desired_length)
        
        if new_spine is None:
            break
        
        else:
            # The method cannot produce valid spine anymore
            if (getSpineLength(new_spine) < desired_length*0.8):
                break
            
            
            #display_discret_spine(spine, "D", unicolor = colors[cpt])
            _, applied_shape = get_estimated_shape(new_spine, shape_distances)
            IoU = calculate_iou(applied_shape, skinPoints, new_spine, frame)
            #print(cpt, IoU)
            if (IoU > maxi):
                maxi = IoU
                cpt_maxi = cpt
            """ # Stopping condition is IoU max was already found and no chance to find it after
            if IoU < 0.1 and maxi > 0.1:
                break """
            if (IoU == -np.inf):
                return None
        cpt += 1
        
        """ if cpt == 28:
            _, applied_shape = get_estimated_shape(new_spine, shape_distances)
            display_shape(skinPoints)
            display_spine(resampled_spine, "*", unicolor = "red")
            display_shape(applied_shape, color = "lightblue")
            display_fig() """
    
    if cpt_maxi != -1:
        #plt.plot(resampled_spine[0][0], resampled_spine[0][1], "bD", ms = 14)
        new_spine = get_spine_candidate(resampled_spine, cpt_maxi, desired_length)
        #plt.plot(new_spine[0][0], new_spine[0][1], "gD", ms = 14)
        _, applied_shape = get_estimated_shape(new_spine, shape_distances)
        new_spine = resample_spine(new_spine, 50)        
        
        np.save("applied_shape{}.npy".format(frame), np.asarray(applied_shape))    
        np.save("normalized_spine{}.npy".format(frame), np.asarray(new_spine))
        
        return new_spine
    
    return spine


""" def fit_IoU(frameAvg):
    skinPoints = read_file("yolo_edge", frameAvg)
    spine = read_file("spine_estimation", frameAvg)
    spine[0] = get_closest_intersection_spine_skin(spine[2], spine[1], skinPoints)
    spine[-1] = get_closest_intersection_spine_skin(spine[-3], spine[-2], skinPoints)
    shape_distances = getDistSpineSkin(spine, skinPoints)
    #shape_distances =  shape_distances[int(len(shape_distances)/2):]
    desired_length = getSpineLength(spine)


    print("starting length correction...")
    frames = get_relevant_frames("filtered_spine")
    nb_frames = len(frames)
    for i, frame in enumerate(frames):
        try:
            overextend_single_spine_to_skin(frame)
            skinPoints = read_file("yolo_edge", frame)
            display_shape(skinPoints)
            spine = read_file("extended_spine", frame)
            current_length = getSpineLength(spine)        
            resampled_spine = resample_spine(spine, 1000)

            #display_discret_spine(spine, "*")
            cpt = 0
            cpt_maxi = -1
            maxi = 0
            while spine is not None:
                spine = get_spine_candidate(resampled_spine, cpt, desired_length)
                if spine is not None:
                    #display_discret_spine(spine, "D", unicolor = colors[cpt])
                    _, applied_shape = get_estimated_shape(spine, shape_distances)
                    IoU = calculate_iou(applied_shape, skinPoints, spine, frame)
                    #print(cpt, IoU)
                    if IoU > maxi:
                        maxi = IoU
                        cpt_maxi = cpt
                cpt += 1
                
            if cpt_maxi != -1:
                spine = get_spine_candidate(resampled_spine, cpt_maxi, desired_length)
                _, applied_shape = get_estimated_shape(spine, shape_distances)
                spine = resample_spine(spine, 50)
                np.save("normalized_spine{}.npy".format(frame), np.asarray(spine))
                
                display_discret_spine(spine)
                display_shape(applied_shape, color = "salmon")
            save_fig("normalized_spine{}".format(frame))
            plt.clf()            
            print("frame: {} | IoU : {} | {:.2f} %".format(frame, maxi, i/nb_frames*100))
        except Exception:
            print("error frame {}".format(frame))
"""
# ========================

def generate_reference_spine(frameAvg):
    skinPoints = read_file("yolo_edge", frameAvg)
    spine = read_file("spine_estimation", frameAvg)
    spine[0] = get_closest_intersection_spine_skin(spine[2], spine[1], skinPoints)
    spine[-1] = get_closest_intersection_spine_skin(spine[-3], spine[-2], skinPoints)
    
    display_shape(skinPoints)
    display_discret_spine(spine)
    np.save("reference_frame{}.npy".format(frameAvg), np.asarray(spine))
    save_fig("reference_frame{}".format(frameAvg))


if __name__ == "__main__":
    
    #os.chdir(r"C:\Users\Thomas Omarini\Documents\1 Stage_poisson\machine_perso\edge_consistency\output_files\tmp")
    os.chdir("C:/Users/Thomas/Documents/stage/edge_consistency/output_files/tmp")
    #filtering_single_spine(325, 1)
    #extend_single_spine_to_skin(325)   
    
    ref_frame = 150
    
    skinPoints = read_file("yolo_edge", ref_frame)
    spine = read_file("extended_spine", ref_frame)
    shape_distances = getDistSpineSkin(spine, skinPoints)
    desired_length = getSpineLength(spine)
    print(desired_length)
    
    frame = 164
    skinPoints = read_file("yolo_edge", frame)
    spine = read_file("filtered_spine", frame)
    print(getSpineLength(spine))
    spine = fit_single_spine_IoU(skinPoints, spine, frame, desired_length, shape_distances)
    #display_fig()
    print(getSpineLength(spine))
    
    display_shape(skinPoints)
    display_spine(spine, unicolor = "red")
    new_skinPoints = read_file("applied_shape", frame)
    display_shape(new_skinPoints, color = "lightblue")
    display_fig()
    
    
    """ frameAvg = 244
    skinPoints = read_file("yolo_edge", frameAvg)
    spine = read_file("spine_estimation", frameAvg)
    spine[0] = get_closest_intersection_spine_skin(spine[2], spine[1], skinPoints)
    spine[-1] = get_closest_intersection_spine_skin(spine[-3], spine[-2], skinPoints)
    
    distSpineAvgSkin = getDistSpineSkin(spine, skinPoints)    
    tailIndexAvg = get_head(spine, skinPoints)

    frame = 250
    skinPoints = read_file("yolo_edge", frame)
    spine = read_file("spine_estimation", frame)
    # Delete unwanted initial points
    spine = spine[1:-1]
    
    # Put the spine in the same direction than reference spine
    tailIndex = get_head(spine, skinPoints)    
    if tailIndex != tailIndexAvg:
        spine = np.flip(spine, axis=0)
    
    # Index 0 = tail
    spine = theta_filtering(spine, np.pi/128, 0.3)
    
    display_shape(skinPoints)
    display_discret_spine(spine)
    
    display_fig() """

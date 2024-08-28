import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import glob

#from frechetdist import frdist
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import splprep, splev

from utils.helper1 import *
from utils.helper2 import *


# ===================================

def get_estimated_shape(spine, distSpineSkin):
    
    # Generate spine function
    tck = getInterpolationU(spine)
    u_new = np.linspace(0, 1, len(distSpineSkin))
    
    estimatedSkinPointsWest = np.zeros([len(u_new), 2])
    estimatedSkinPointsEast = np.zeros([len(u_new), 2])
    
    # Determine iteration direction and range

    u_iter = enumerate(u_new)
    index_mapper = lambda j: j

    # Process spine points
    for j, u in u_iter:
        k = index_mapper(j)
        southPoint = splev(u - 1e-4, tck)
        northPoint = splev(u, tck)
        
        ribPointWest = orthogonalLine(northPoint, southPoint, 2, "West", "South")
        estimatedSkinPointsWest[k] = get_estimated_skin_point(northPoint, ribPointWest, distSpineSkin[k])
        
        ribPointEast = orthogonalLine(northPoint, southPoint, 2, "East", "South")
        estimatedSkinPointsEast[k] = get_estimated_skin_point(northPoint, ribPointEast, distSpineSkin[k])
        
    # Returns points as a closed polygon
    return tck, np.concatenate([estimatedSkinPointsWest, np.flip(estimatedSkinPointsEast, axis = 0)])

def applying_shape(frameAvg):
    
    print("starting applying shape to spines...")
    
    # =============================================
    # 
    # Filter out useful data
    # Get all npy files
    npy_files = glob.glob('*.npy')
    
    # Filter only the relevant npy files
    pattern = re.compile(r'normalized_spine\d+\.npy')
    useful_npy_files = [filename for filename in npy_files if pattern.match(filename)]
    
    # Get the number associated to the frame
    frames = []
    for file_name in useful_npy_files:
        frames.append(int(os.path.splitext(file_name)[0].replace("normalized_spine", "")))
    nb_frames = len(frames)
    
    # =============================================
    # 
    # Get relevant information using the reference spine
    skinPoints = read_file("yolo_edge", frameAvg)
    spine = read_file("spine_estimation", frameAvg)
    spine[0] = get_closest_intersection_spine_skin(spine[2], spine[1], skinPoints)
    spine[-1] = get_closest_intersection_spine_skin(spine[-3], spine[-2], skinPoints)
    distSpineSkin = getDistSpineSkin(spine, skinPoints)    
    
    # =============================================
    # 
    # Process each frame
    
    for i, frame in enumerate(frames):
        print("frame: {} | {:.2f} %".format(frame, i/nb_frames*100))
        skinPoints = read_file("yolo_edge", frame)
        spine = read_file("normalized_spine", frame) 
        tck, EstimatedSkinPoints = get_estimated_shape(spine, distSpineSkin)
        np.save("applied_shape{}.npy".format(frame), np.asarray(EstimatedSkinPoints))
        
        display_shape(skinPoints)
        display_continuous_fish(tck, EstimatedSkinPoints)
        save_fig("applied_shape{}".format(frame))

# ============================

if __name__ == "__main__":
    from spine_estimation import direction_uniformization
    
    os.chdir("C:/Users/Thomas/Documents/stage/edge_consistency/output_files/tmp")
    
    
    
    
    ref_frame = 150
    
    skinPoints = read_file("yolo_edge", ref_frame)
    spine = read_file("extended_spine", ref_frame)
    shape_distances = getDistSpineSkin(spine, skinPoints)
    desired_length = getSpineLength(spine)
    _, shape = get_estimated_shape(spine, shape_distances)
    display_shape(shape, color = "lightblue")
    display_shape(skinPoints)
    display_spine(spine)
    plt.plot(spine[0][0], spine[0][1], "rD", ms = 14)
    spine = direction_uniformization(spine, skinPoints)
    plt.plot(spine[0][0], spine[0][1], "bD", ms = 14)
    display_fig()
    
    
    frame = 186
    skinPoints = read_file("yolo_edge", frame)
    spine = read_file("filtered_spine", frame)
    _, shape = get_estimated_shape(spine, shape_distances)

    display_shape(shape, color = "lightblue")
    display_shape(skinPoints)
    display_spine(spine)
    plt.plot(spine[0][0], spine[0][1], "rD", ms = 14)
    display_fig()


    #save_fig("applied_shape{}".format(frame))
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import splev 
import os
import glob 
import re
import csv


# ================================
#
# Functions related to display or file managment
#
# ================================

colors = [
        "navy", "darkred", "darkgreen", "darkorange", "purple", "sienna", "darkcyan", "darkviolet",
        "darkblue", "maroon", "forestgreen", "chocolate", "indigo", "olive", "teal", "slateblue",
        "midnightblue", "firebrick", "darkolivegreen", "sienna", "darkslategray", "royalblue",
        "darkmagenta", "darkgoldenrod", "darkseagreen", "darkslateblue", "darkorchid", "saddlebrown",
        "darkgreen", "darkturquoise", "darkkhaki", "darkcyan", "indigo", "darkred", "darkorange",
        "darkblue", "darkolivegreen", "darkslategray", "maroon", "darkseagreen", "darkviolet",
        "darkslateblue", "darkgoldenrod", "darkcyan", "darkred", "darkolivegreen"
    ]

def display_spine(spine, marker = "*", unicolor = False):
    if (unicolor == False):
        for i in range(1, len(spine)):
            plt.plot([spine[i-1][0], spine[i][0]], [spine[i-1][1], spine[i][1]], color = colors[i%len(colors)], marker = marker, linestyle = "--", ms = 10, linewidth=2)
        plt.plot(spine[0][0], spine[0][1], color = colors[i%len(colors)], marker = marker, ms = 10, linewidth=2)
    else:
        for i in range(len(spine)-1):
            plt.plot([spine[i][0], spine[i+1][0]], [spine[i][1], spine[i+1][1]], linestyle = "-.", color = unicolor, marker = marker, ms = 10, linewidth=2)
        plt.plot(spine[0][0], spine[0][1], color = unicolor, marker = marker, ms = 10, linewidth=2)

def display_shape(skinPoints, color = "lightgray"):
    X = [skinPoint[0] for skinPoint in skinPoints]  
    Y = [skinPoint[1] for skinPoint in skinPoints]
    plt.plot(X, Y, color = color, marker = ".", linestyle = "-.")

def display_discret_spine(spine, marker = "*", unicolor = False):
    if unicolor == False:
        for i in range(len(spine)-1):
            plt.plot([spine[i][0], spine[i+1][0]], [spine[i][1], spine[i+1][1]], linestyle = "-.", color = colors[i%len(colors)], ms = 10, linewidth=2)
            plt.plot(spine[i][0], spine[i][1], color = colors[i%len(colors)], marker = marker, ms = 10, linewidth=2)
        plt.plot(spine[-1][0], spine[-1][1], color = colors[(i+1)%len(colors)], marker = marker, ms = 10, linewidth=2)
    else:
        for i in range(len(spine)-1):
            plt.plot([spine[i][0], spine[i+1][0]], [spine[i][1], spine[i+1][1]], linestyle = "-.", color = unicolor, ms = 10, linewidth=2)
            plt.plot(spine[i][0], spine[i][1], color = unicolor, marker = marker, ms = 10, linewidth=2)
        plt.plot(spine[-1][0], spine[-1][1], color = unicolor, marker = marker, ms = 10, linewidth=2)

def display_fig():
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def display_continuous_spine(tck, unicolor = False, nb_point_per_spine = 100):
    u_new = np.linspace(0, 1, nb_point_per_spine)
    spine_points = np.array([splev(u, tck) for u in u_new])
    color_indices = [i % len(colors) for i in range(len(u_new))]
    if unicolor == False :
        for l in range(len(u_new) - 1):
            plt.plot(spine_points[l:l+2, 0], spine_points[l:l+2, 1], linestyle='-.', color=colors[color_indices[l]], ms = 10, linewidth=2)
            plt.plot(spine_points[l, 0], spine_points[l, 1], color=colors[color_indices[l]], marker='*', ms = 10, linewidth=2)
        plt.plot(spine_points[-1, 0], spine_points[-1, 1], color=colors[color_indices[-1]], marker='*', ms = 10, linewidth=2)
    else:
        for l in range(len(u_new) - 1):
            plt.plot(spine_points[l:l+2, 0], spine_points[l:l+2, 1], linestyle='-.', color=unicolor, ms = 10, linewidth=2)
            plt.plot(spine_points[l, 0], spine_points[l, 1], color=unicolor, marker='*', ms = 10, linewidth=2)
        plt.plot(spine_points[-1, 0], spine_points[-1, 1], color=unicolor, marker='*', ms = 10, linewidth=2)

def display_continuous_fish(tck, EstimatedSkinPoints):
    plt.plot(EstimatedSkinPoints[:, 0], EstimatedSkinPoints[:, 1], color = "lightblue", marker = ".", linestyle = "-.", ms = 10, linewidth=2)
    u_new = np.linspace(0, 1, 50)
    for l in range(len(u_new)-1):
        spinePoint1 = splev(u_new[l], tck)
        spinePoint2 = splev(u_new[l+1], tck)        
        plt.plot([spinePoint1[0], spinePoint2[0]], [spinePoint1[1], spinePoint2[1]], linestyle = "-.", color = colors[l%len(colors)], ms = 10, linewidth=2)
        plt.plot(spinePoint1[0], spinePoint1[1], color = colors[l%len(colors)], marker = "*", ms = 10, linewidth=2)
    spinePoint3 = splev(u_new[-1], tck)
    plt.plot(spinePoint3[0], spinePoint3[1], color = colors[(l+1)%len(colors)], marker = "*", ms = 10, linewidth=2)

# ===============================

def get_relevant_frames(file):
    # Compile the regex pattern outside the list comprehension
    pattern = re.compile(r'{}(\d+)\.npy'.format(re.escape(file)))
    
    # List comprehension to filter and extract frame numbers in one go
    frames = [
        int(pattern.match(filename).group(1))
        for filename in glob.glob('*.npy')
        if pattern.match(filename)
    ]
    
    if frames == []:
        print("'{}' files not found".format(file))
    return frames

def save_fig(figname):
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("{}.png".format(figname))
    #plt.show()
    plt.clf()

def read_file(file_name, frame):
    try:
        data = np.load("{}{}.npy".format(file_name, frame))
        if np.size(data) == 0:
            return None
    except FileNotFoundError:
        data = None
    
    if file_name == "yolo_edge":
        X, Y = data[0], data[1]
        data = np.array([[X[i], Y[i]] for i in range(len(X))])
    
    return data

def load_frames_spines_from_csv(filename):
    print(os.getcwd())
    filename = "{}.csv".format(filename)
    spines = []
    frame = []

    # Read the CSV file and reconstruct the spines data
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            frame_index = int(row[0])
            spine_data = list(map(float, row[1:]))
            
            # Reconstruct the 2D array for each frame
            frame_data = np.array(spine_data).reshape(-1, 2)
            spines.append(frame_data)
            frame.append(frame_index)
            #print(getSpineLength(frame_data))
    
    return frame, np.array(spines)

if __name__ == "__main__":
    get_relevant_frames("oui")
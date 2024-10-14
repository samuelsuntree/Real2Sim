import os
import numpy as np 
import pandas as pd
import csv
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from utils.helper2 import remove_duplicates
from utils.helper1 import load_frames_spines_from_csv


from shape_processing import get_estimated_shape







os.chdir(r"F:/Real2Sim/output_files/final")


def initialize_csv_files(x_file, y_file):
    file1 = open(x_file, 'w', newline='')
    file2 = open(y_file, 'w', newline='')
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)
    return file1, file2, writer1, writer2

def estimate_and_interpolate_shapes(spines, shape_width, desired_points_count):
    estimated_skin_points = np.zeros([len(spines), desired_points_count, 2])
    for i, spine in enumerate(spines):
        _, estimated_skin_point = get_estimated_shape(spine, shape_width)
        x = estimated_skin_point[:, 0]
        y = estimated_skin_point[:, 1]
        x, y = remove_duplicates(x, y)
        tck, u = splprep([x, y], s=50)
        u_new = np.linspace(0, 1, desired_points_count)
        estimated_skin_point = np.array(splev(u_new, tck)).T
        estimated_skin_points[i] = estimated_skin_point
    return estimated_skin_points

def write_shape_to_csv(estimated_skin_points, writer1, writer2, desired_points_count, resX, resY, screenX, screenY):
    for trajectory in range(desired_points_count):
        columnX = estimated_skin_points[:, trajectory, 0] * resX / screenX
        columnY = estimated_skin_points[:, trajectory, 1] * resY / screenY
        writer1.writerow(columnX)
        writer2.writerow(columnY)

def calculate_derivatives(y_file, y_dot_file, desired_points_count):
    Y_data = pd.read_csv(y_file, header=None)
    y_dot = pd.DataFrame(index=range(len(Y_data[0])), columns=range(len(Y_data.columns)))
    sfd = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 11})
    T0 = np.array([0.5 + j / 2 for j in range(len(Y_data.columns))])
    
    for i in range(len(Y_data[0])):
        y0 = [Y_data[j][i] for j in range(len(Y_data.columns))]
        y_dot0 = sfd._differentiate(y0, T0)
        for j in range(len(Y_data.columns)):
            y_dot[j][i] = y_dot0[j]
    
    with open(y_dot_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(desired_points_count):
            columnFilteredY = [y_dot[j][i] for j in range(len(Y_data.columns))]
            writer.writerow(columnFilteredY)
    print(f"CSV file {y_dot_file} has been created.")

def shape_reconstruction(filename, shape_width):
    # Load spines and initialize parameters
    _, spines = load_frames_spines_from_csv(filename)
    desired_points_count = 256
    screenX, screenY = 2000, 1200
    resX, resY = 2**8, 2**7
    x_file = 'x.csv'
    y_file = 'y.csv'
    y_dot_file = 'y_dot.csv'
    
    # Initialize CSV writers
    file1, file2, writer1, writer2 = initialize_csv_files(x_file, y_file)
    
    # Estimate and interpolate shapes
    estimated_skin_points = estimate_and_interpolate_shapes(spines, shape_width, desired_points_count)
    
    # Write shapes to CSV files
    write_shape_to_csv(estimated_skin_points, writer1, writer2, desired_points_count, resX, resY, screenX, screenY)
    
    # Close CSV files
    file1.close()
    file2.close()
        
    # Calculate derivatives and write to CSV
    calculate_derivatives(y_file, y_dot_file, desired_points_count)



""" def shape_reconstruction(filename, shape_width):
    
    _, spines = load_frames_spines_from_csv(filename)
    
    # Output desired
    desired_points_count = 256          # power of 2
    screenX, screenY = 2000, 1200
    resX, resY = 2**8, 2**7

    x_file = 'x.csv'
    y_file = 'y.csv'
    y_dot_file = 'y_dot.csv'

    with open(x_file, 'w', newline='') as file1, open(y_file, 'w', newline='') as file2:
        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)
        
        estimated_skin_points = np.zeros([len(spines), desired_points_count, 2])
        for i, spine in enumerate(spines):
            _, estimated_skin_point = get_estimated_shape(spine, shape_width)
            
            x = estimated_skin_point[:, 0]
            y = estimated_skin_point[:, 1]
            x, y = remove_duplicates(x, y)
            
            tck, u = splprep(x = [x, y], s=50)
            u_new = np.linspace(0, 1, desired_points_count)
            estimated_skin_point = [splev(u, tck) for u in u_new]
            estimated_skin_points[i] = estimated_skin_point
            #plt.plot(x, y, "r+")
        
        for trajectory in range(desired_points_count):
            columnX = estimated_skin_points[:, trajectory, 0]*resX/screenX
            columnY = estimated_skin_points[:, trajectory, 1]*resY/screenY

            # Write the columns to the respective CSV files
            writer1.writerow(columnX)
            writer2.writerow(columnY)

    print(f"CSV files {x_file} and {y_file} have been created.")


    X_data = pd.read_csv("x.csv", header=None)
    Y_data = pd.read_csv("y.csv", header=None)

    y_dot = pd.DataFrame(index=range(len(Y_data[0])), columns=range(len(Y_data.columns)))

    sfd = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 11}) # pySINDy smooth finite differenciation
    T0 = np.array([0.5 + j/2 for j in range(len(Y_data.columns))]) # time vector for the derivation
    for i in range(len(Y_data[0])):
        y0 = [Y_data[j][i] for j in range(len(Y_data.columns))]
        y_dot0 = sfd._differentiate(y0, T0)
        for j in range(len(Y_data.columns)):
            y_dot[j][i] = y_dot0[j]

    # Open the CSV files for writing
    with open(y_dot_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Iterate through the main list
        for i in range(desired_points_count):
            columnFilteredY = []
            for j in range(len(Y_data.columns)):
                # Extract the filtered y for each tuple
                columnFilteredY.append(y_dot[j][i])

            # Write the columns to the respective CSV files
            writer.writerow(columnFilteredY)

    print(f"CSV files {y_dot_file} has been created.")


 """
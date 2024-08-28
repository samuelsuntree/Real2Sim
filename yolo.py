import os

import cv2
import numpy as np
from torchvision import models
from ultralytics import YOLO
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def crop_and_resize_image(model, input_image, target_size=(640, 640)):
    # Read the input image
    img = input_image

    # Get the dimensions of the input image
    img_height, img_width = img.shape[:2]

    # Get the bounding box of the detected shape
    results = model(img)
    r = results[0]
    boxes = r.boxes.xyxy.tolist()
    if boxes == []:
        return None

    xmin, ymin, xmax, ymax = boxes[0]
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)

    # Calculate the center of the bounding box
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    # Calculate the cropping coordinates
    crop_xmin = max(0, center_x - target_size[0] // 2)
    crop_ymin = max(0, center_y - target_size[1] // 2)
    crop_xmax = min(img_width, center_x + target_size[0] // 2)
    crop_ymax = min(img_height, center_y + target_size[1] // 2)

    # Check if the target region exceeds image boundaries
    if (crop_xmax - crop_xmin) < target_size[0]:
        if center_x - (target_size[0] // 2) < 0:
            crop_xmax = crop_xmin + target_size[0]
        elif center_x + (target_size[0] // 2) > img_width:
            crop_xmin = crop_xmax - target_size[0]

    if (crop_ymax - crop_ymin) < target_size[1]:
        if center_y - (target_size[1] // 2) < 0:
            crop_ymax = crop_ymin + target_size[1]
        elif center_y + (target_size[1] // 2) > img_height:
            crop_ymin = crop_ymax - target_size[1]

    # Crop and resize the image
    cropped_resized_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    return cropped_resized_img

def generalizeLabel(cropped_img, cropped_label, img):

    cropped_h, _ = cropped_img.shape[:2]
    img_h, _ = img.shape[:2]

    original_label_list = []
    label_list = cropped_label

    # If one of the images is the cropped version of the other
    if img_h > cropped_h:

        # Perform template matching
        result = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        top_left_coordinates = max_loc

        x_top, y_top = top_left_coordinates

        # Create new label for the original image
        for i in label_list:
            original_label_list.append([i[0] + x_top, i[1] + y_top])

        return original_label_list

def apply_yolo(model, vidcap, start_ToI, end_ToI, s, num_points):

    print("Starting yolo processing")

    #====================================
    # Video parameters
    nb_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = vidcap.get(cv2.CAP_PROP_FPS) 
    frame_width, frame_height = int(vidcap.get(3)), int(vidcap.get(4))
    size = (frame_width, frame_height)

    # Convert times of interests to frame
    start_FoI = start_ToI*fps   # FoI : Frame of Interest
    end_FoI = end_ToI*fps
    nb_FoI = end_FoI - start_FoI

    # Start processing the video at the starting time
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ToI*1e3) 

    # Contrast 
    alpha = 1  # Contrast control (1.0 for no change)
    beta = 0     # Brightness control (0 for no change)

    # ===========================================
    new_u = np.linspace(0, 1, num_points) # spline definition domain
    # Process each frame
    count = 0
    
    while (vidcap.isOpened() and (count < nb_FoI)):
        # Give feedback on process
        #print("{:.2f} %". format(100-(nb_FoI-count)/nb_FoI*100))
        
        # Get frame and change its shape
        _, frame = vidcap.read()

        # Perform YOLO inference    
        frame_cropped = crop_and_resize_image(model, frame, (640,640))
        if frame_cropped is not None:
            frame_cropped_contrasted = cv2.convertScaleAbs(frame_cropped, alpha=alpha, beta=beta)
            results = model(frame_cropped_contrasted, verbose=False)

            # Test on the results
            for r in results:
                if r.masks == None:
                    break
                mask = r.masks.xy
                xys = mask[0]
                # Detect the edge
                uncropped_xys = generalizeLabel(frame_cropped, xys, frame)
                # If there is a fish
                if uncropped_xys is not None:    
                    # Get the coordonates of the edge        
                    XY = np.int32(uncropped_xys)
                    # filter out same points
                    for i in range(len(XY)-1):
                        if ((XY[i][0] == XY[i-1][0]) and (XY[i][1] == XY[i-1][1])):
                            XY[i-1] = np.array([-1, -1])
                            
                    X, Y = [], []
                    for xy in XY:
                        if (xy[0] != -1):
                            X.append(xy[0])
                            Y.append(xy[1])
                    
                    # Calculate the spline representation
                    try:
                        tck, u = splprep([X, Y], s=s, per = len(X))
                    except Exception:
                        #print("couldn't perform fit")
                        plt.plot(X, Y, color="red", marker = "x", linestyle = "")
                        ax = plt.gca()
                        ax.set_aspect('equal', adjustable='box')
                        #plt.show()
                        plt.savefig("couldnt_fit_{}.png".format(count))
                        np.save("couldnt_fit{}.npy".format(count), np.array([X, Y]))
                        plt.clf()
                        break
                    
                    interpolated_points = splev(new_u, tck)
                                        
                    plt.plot(interpolated_points[0], interpolated_points[1], color="lightgray", marker = ".", linestyle = "dotted")
                    plt.plot(X, Y, color="red", marker = "x", linestyle = "")
                    ax = plt.gca()
                    ax.set_aspect('equal', adjustable='box')
                    plt.savefig("yolo_edge{}.png".format(count))
                    plt.clf()
                    
                    np.save("yolo_edge{}.npy".format(count), interpolated_points)
                
                else:
                    ax = plt.gca()
                    ax.set_aspect('equal', adjustable='box')
                    #plt.imsave("problem{}.png".format(count), frame)
                    plt.clf()
        else:
            minutes = (start_FoI+count)/fps
            #print("no fish detected on time {}:{}".format(int(minutes//60), int(minutes%60)))
            plt.clf()

        count += 1

    print("Yolo processing done")
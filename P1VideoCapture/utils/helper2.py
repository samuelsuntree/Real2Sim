import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev

# ================================
#
# Functions related to display or file managment
#
# ================================

def distQuad(point0, point1):
    return np.sqrt((point1[0]-point0[0])**2 + (point1[1]-point0[1])**2)

def orthogonalLine(northPoint, southPoint, orthoStep, side, origin):
    # Compute the direction vector
    dx = northPoint[0] - southPoint[0]
    dy = northPoint[1] - southPoint[1]
    
    # Normalize the direction vector
    length = np.sqrt(dx**2 + dy**2)
    dx /= length
    dy /= length
    
    # Compute the orthogonal vector by rotating 90 degrees
    ortho_dx = -dy
    ortho_dy = dx
    
    # Scale the orthogonal vector by orthoStep
    ortho_dx *= orthoStep
    ortho_dy *= orthoStep
    
    # Determine the side (East or West)
    if side == "West":
        ortho_dx = -ortho_dx
        ortho_dy = -ortho_dy
    
    # Determine the origin (North or South)
    if origin == 'North':
        originPoint = northPoint
    else:
        originPoint = southPoint
    
    # Compute the orthogonal point
    xI = originPoint[0] + ortho_dx
    yI = originPoint[1] + ortho_dy
    
    return np.array([xI, yI])

def findIntersectionsWithSkin(point0, point1, skinPoints):
    """ Both points MUST NOT be skinPoints, otherwise code breaks"""
    # Computes cross product for all points
    vectors_to_skin = point0 - skinPoints
    vectors_to_rib = point0 - point1
    crossProduct = np.cross(vectors_to_skin, vectors_to_rib)
    
    # Find the index where the crossproduct changes
    tmpIdx = np.where(np.sign(crossProduct[:-1]) != np.sign(crossProduct[1:]))[0]

    return tmpIdx

def getIntersectionsWithSkin(tmpIdx, ribPoint, originSpinePoint, skinPoints):
    
    coefRib = computeDirectoryCoefficient(ribPoint, originSpinePoint)
    nbSkinPoints = len(skinPoints)
    
    # If no intersection found
    if tmpIdx.size == 0:
        return None
    
    intersectionPoints = np.zeros((tmpIdx.size, 2)) if tmpIdx.size > 1 else None
    
    # Process intersections
    for i, idx in enumerate(tmpIdx):
        coefSkin = computeDirectoryCoefficient(skinPoints[idx], skinPoints[(idx+1) % nbSkinPoints])
        intersection = computeLineIntersection(skinPoints[idx], coefSkin, ribPoint, coefRib)
        
        if tmpIdx.size > 1:
            intersectionPoints[i] = intersection
        else:
            intersectionPoints = intersection
    
    return intersectionPoints

def computeDirectoryCoefficient(point1, point2, inverted = False):
    """ From point2 to point1"""
    dx = (point1[0] - point2[0])
    dy = (point1[1] - point2[1])       

    # Numerical error
    epsi = 1e-11

    # If you want the coef orthogonal to the line
    if inverted == True:
        dx, dy = -dy, dx
        
    # Compute the directory coefficient of the perpendicular line w/ handling of specific cases
    if -epsi < dx < epsi:  
        directoryCoef = None
    elif -epsi < dy < epsi:  
        directoryCoef = 0
    else:
        directoryCoef = dy/dx

    return directoryCoef

def get_estimated_skin_point(northPoint, ribPoint, avgDist):
    estimatedSkinPoint = np.zeros([2])
    
    # Compute the direction vector
    dx = northPoint[0] - ribPoint[0]
    dy = northPoint[1] - ribPoint[1]
    
    # Normalize the direction vector
    length = np.sqrt(dx**2 + dy**2)
    dx /= length
    dy /= length
    
    # Scale the orthogonal vector by avgDist
    dx *= avgDist
    dy *= avgDist
    
    # Compute the orthogonal point
    estimatedSkinPoint[0] = northPoint[0] + dx
    estimatedSkinPoint[1] = northPoint[1] + dy
    
    return estimatedSkinPoint

def get_intersection_rib_skin(spine_segment, skinPoints):
    northPoint, southPoint = spine_segment
    ribPoint = orthogonalLine(northPoint, southPoint, 0.2, "West", "North")
    if not point_inside_polygon(ribPoint, skinPoints):
        ribPoint = orthogonalLine(northPoint, southPoint, 0.2, "East", "North")
        
    #plt.plot([ribPoint[0], southPoint[0]], [ribPoint[1], southPoint[1]], "r+", linestyle = "-", ms = 14)
    #tmpIdx = findIntersectionsWithSkin(southPoint, ribPoint, skinPoints)
    tmpIdx = findIntersectionsWithSkin(northPoint, ribPoint, skinPoints)
    intersection_rib_skin = getIntersectionsWithSkin(tmpIdx, ribPoint, northPoint, skinPoints)
    # Filter out multiples intersections 
    """  if intersection_rib_skin is None :
        return None
    elif len(intersection_rib_skin) > 2:
        dist_tmp = np.zeros([len(intersection_rib_skin)])
        for i, intersection_point in enumerate(intersection_rib_skin):
            dist_tmp[i] = distQuad(southPoint, intersection_point)
        intersection_rib_skin = intersection_rib_skin[np.argsort(dist_tmp[:2])] """
        
    return intersection_rib_skin

def get_closest_intersection_spine_skin(spine_point, closest_spine_point, skinPoints):
    tmpIdx = findIntersectionsWithSkin(spine_point, closest_spine_point, skinPoints)
    intersection_spine_skin = getIntersectionsWithSkin(tmpIdx, spine_point, closest_spine_point, skinPoints)
    # Filter out multiples intersections
    dist_tmp = np.zeros([len(intersection_spine_skin)])
    for i, intersection_point in enumerate(intersection_spine_skin):
        dist_tmp[i] = distQuad(closest_spine_point, intersection_point)
    intersection_spine_skin = intersection_spine_skin[np.argmin(dist_tmp)]
        
    return intersection_spine_skin

def computeLineIntersection(point1, coef1, point2, coef2):
    
    # Handling all cases
    if coef1 == None and coef2 != 0:
        xI = point1[0]
        yI = coef2 * (xI - point2[0]) + point2[1]
        
    elif coef1 == None and coef2 == 0:
        xI = point1[0]
        yI = point2[1]
        
    elif coef1 != 0 and coef2 == None:
        xI = point2[0]
        yI = coef1 * (xI - point1[0]) + point1[1]
        
    elif coef1 == 0 and coef2 == None:
        xI = point2[0]
        yI = point1[1]
    
    else :
        xI = (point1[0]*coef1 - point2[0]*coef2 + point2[1] - point1[1])/(coef1 - coef2)
        yI = coef1 * (xI - point1[0]) + point1[1]
    
    return np.array([xI, yI])

def point_inside_polygon(point, polygon):
    """
    Check if a point (x, y) is inside a polygon defined by a list of points.
    """
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    x, y = point
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_head(spine, skinPoints):
    # ===============================
    # find who is head and who is tail
    dist = np.zeros([2])
    
    # Compute the average distance to the skin for the extremity point
    # Need to compute both distance to handle case where the point is close to the skin
    spine_segment = [spine[1], spine[2]]
    intersection_rib_skin = get_intersection_rib_skin(spine_segment, skinPoints)
    # Get only the 2 closest intersection point
    if len(intersection_rib_skin) > 2:
        dist_tmp = np.zeros([len(intersection_rib_skin)])
        for i, intersection_point in enumerate(intersection_rib_skin):
            dist_tmp[i] = distQuad(spine[1], intersection_point)
        intersection_rib_skin = intersection_rib_skin[np.argsort(dist_tmp[:2])]
    
    """ for inter in intersection_rib_skin:
        plt.plot(inter[0], inter[1], "r*", ms = 14) """
    
    
    dist[0] = (distQuad(spine[1], intersection_rib_skin[0]) + distQuad(spine[1], intersection_rib_skin[1]))/2
    
    spine_segment = [spine[-2], spine[-3]]
    intersection_rib_skin = get_intersection_rib_skin(spine_segment, skinPoints) 
    # Get only the 2 closest intersection point
    if len(intersection_rib_skin) > 2:
        dist_tmp = np.zeros([len(intersection_rib_skin)])
        for i, intersection_point in enumerate(intersection_rib_skin):
            dist_tmp[i] = distQuad(spine[-2], intersection_point)        
        intersection_rib_skin = intersection_rib_skin[np.argsort(dist_tmp)[:2]]
    
    """ for inter in intersection_rib_skin:
        plt.plot(inter[0], inter[1], "b*", ms = 14) """
    
    dist[1] = (distQuad(spine[-2], intersection_rib_skin[0]) + distQuad(spine[-2], intersection_rib_skin[1]))/2

    #print(dist)
    if dist[1] > dist[0]:
        tailIndex = 0
    elif dist[1] < dist[0]:
        tailIndex = 1
    else:
        print("problem tail index")
        tailIndex = 0
    
    return tailIndex

def getSpineLength(spine):
    diffs = np.diff(spine, axis=0)
    squared_distances = np.sum(diffs**2, axis=1)
    distances = np.sqrt(squared_distances)
    length = np.sum(distances)
    
    return length

def getInterpolationU(spine):
    # Separate the points into x and y coordinates
    x = spine[:, 0]
    y = spine[:, 1]
    tck, u = splprep(x = [x, y], s=0)
    
    return tck

def getDistSpineSkin(spine, skinPoints, nb_point_per_spine = 100):
    
    # Generate spine function
    tck = getInterpolationU(spine)
    u_new = np.linspace(0, 1, nb_point_per_spine)

    # Get the points on the spline
    northPoints = np.array([splev(u, tck) for u in u_new])
    southPoints = np.array([splev(u-1e-5, tck) for u in u_new])

    # Initialize distances array
    dist = np.zeros((len(u_new), 2))
    colors = ["r", "b", "darkgreen", "salmon"]
    for i, (northPoint, southPoint) in enumerate(zip(northPoints, southPoints)):
        spine_segment = [northPoint, southPoint]
        intersection_rib_skin = get_intersection_rib_skin(spine_segment, skinPoints)
        # Get only the 2 closest intersection point
        if (point_inside_polygon(northPoint, skinPoints) is True):
            if intersection_rib_skin is None :
                continue
            elif len(intersection_rib_skin) > 2:
                dist_tmp = np.zeros([len(intersection_rib_skin)])
                for i, intersection_point in enumerate(intersection_rib_skin):
                    dist_tmp[i] = distQuad(spine_segment[0], intersection_point)
                    #plt.plot(intersection_point[0], intersection_point[1], "+", color = colors[i])
                intersection_rib_skin = intersection_rib_skin[np.argsort(dist_tmp)[:2]]

            dist[i, 0] = distQuad(northPoint, intersection_rib_skin[0])
            dist[i, 1] = distQuad(northPoint, intersection_rib_skin[1])
            
            #plt.plot(intersection_rib_skin[1][0], intersection_rib_skin[1][1], "b+")
            #plt.plot(intersection_rib_skin[0][0], intersection_rib_skin[0][1], "r+")
            
    return np.mean(dist, axis=1)

def get_dif_dist_spine_skin(spine, skinPoints, nb_point_per_spine = 100):
    
    # Generate spine function
    tck = getInterpolationU(spine)
    u_new = np.linspace(0, 1, nb_point_per_spine)

    # Get the points on the spline
    northPoints = np.array([splev(u, tck) for u in u_new])
    southPoints = np.array([splev(u-1e-5, tck) for u in u_new])

    # Initialize distances array
    dist = np.zeros((len(u_new), 2))
    for i, (northPoint, southPoint) in enumerate(zip(northPoints, southPoints)):
        if (i == 0):
            southPoint = splev(1e-5, tck)
        spine_segment = [northPoint, southPoint]
        intersection_rib_skin = get_intersection_rib_skin(spine_segment, skinPoints)
        # Get only the 2 closest intersection point
        #if (point_inside_polygon(northPoint, skinPoints) is True):
        if intersection_rib_skin is None :
            continue
        elif len(intersection_rib_skin) > 2:
            dist_tmp = np.zeros([len(intersection_rib_skin)])
            for i, intersection_point in enumerate(intersection_rib_skin):
                dist_tmp[i] = distQuad(spine_segment[0], intersection_point)
                #plt.plot(intersection_point[0], intersection_point[1], "+", color = colors[i])
            intersection_rib_skin = intersection_rib_skin[np.argsort(dist_tmp)[:2]]
            
        dist[i, 0] = distQuad(northPoint, intersection_rib_skin[0])
        dist[i, 1] = distQuad(northPoint, intersection_rib_skin[1])
    
    return np.sum(np.abs(dist[:, 0]-dist[:, 1]))**2

def extendSegment(fixedPoint, pointToMove, dist):
    if (np.all(fixedPoint == pointToMove)):
        return None
    dx = pointToMove[0] - fixedPoint[0]
    dy = pointToMove[1] - fixedPoint[1] 
    current_dist = distQuad(fixedPoint, pointToMove)

    dx *= dist/current_dist
    dy *= dist/current_dist

    return np.array([fixedPoint[0] + dx, fixedPoint[1] + dy])

def remove_duplicates(array1, array2):
    combined_array = list(zip(array1, array2))
    seen = {}
    unique1 = []
    unique2 = []

    for item in combined_array:
        key = tuple(item)
        if key not in seen:
            seen[key] = True
            unique1.append(item[0])
            unique2.append(item[1])
    
    return unique1, unique2


if __name__ == "__main__":
    
    from helper1 import read_file, display_spine, display_shape, display_fig
    #os.chdir("C:/Users/Thomas Omarini/Documents/1 Stage_poisson/machine_perso/edge_consistency/output_files/tmp/")
    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency\output_files\tmp")
    frame = 180

    spine = read_file("extended_spine", frame)
    
    plt.plot(spine[0][0], spine[0][1], "rD", ms = 12)
    skinPoints = read_file("yolo_edge", frame)
    display_shape(skinPoints)
    display_spine(spine)
    display_fig()

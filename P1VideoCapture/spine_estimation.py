import os
import re
import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.optimize import minimize_scalar

from utils.helper2 import *
from utils.helper1 import *



def findProjection(point1, point2, pointToProject):

    invertedCoef = computeDirectoryCoefficient(point1, point2, inverted = True)
    coef = computeDirectoryCoefficient(point1, point2, inverted = False)
    projectedPoint = computeLineIntersection(pointToProject, invertedCoef, point1, coef)

    return projectedPoint

def findInflectionPoints(originSpinePoint, ribPoint, skinPoints):
    """ Returns index of skin points that are orthogonal to the spine segment """
    # Computes cross product for all points
    crossProduct = np.zeros(len(skinPoints))
    for i in range(len(skinPoints)):
        crossProduct[i] = np.cross(np.subtract(skinPoints[i],skinPoints[i-1]), originSpinePoint - ribPoint)
    
    # Find the index where the crossproduct changes
    tmpIdx = np.where(np.sign(crossProduct[:-1]) != np.sign(crossProduct[1:]))[0]
    
    return tmpIdx

def segregateInflectionPoints(currentSpinePoint, idxListInflectionPoints, ribPointNorth, skinPoints):
    # Segregate inflection points to east and west 
    idxInflectionWest, idxInflectionEast = [], []
    for idx in idxListInflectionPoints:
        if np.dot(currentSpinePoint - np.array(skinPoints[idx]), currentSpinePoint - ribPointNorth) >= 0:
            idxInflectionEast.append(idx)
        else:
            idxInflectionWest.append(idx)
    return idxInflectionWest, idxInflectionEast

def polygon_area(vertices):
    # Ensure the vertices are in clockwise order
    vertices = np.array(vertices)
    if np.cross(vertices[-1] - vertices[0], vertices[1] - vertices[0]) < 0:
        vertices = vertices[::-1]

    # Use the Shoelace formula to calculate the area
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def merge_and_sort_arrays_with_indices(arrays):
    merged_list = []
    for i, arr in enumerate(arrays):
        for j, value in np.ndenumerate(arr):
            merged_list.append((value, i, j[0]))
    merged_list.sort(key=lambda x: x[0])  # Sort based on the distance point/rib
    return merged_list

def findClosestIntersectionPointToRib(intersectionPoints, originPoint, idx):
    dist = np.zeros([len(intersectionPoints), 2])
    for i, intersectionPoint in enumerate(intersectionPoints):
        dist[i] = distQuad(originPoint, intersectionPoint), idx[i] # List distance to the rib, idx of the current intersection point
    if len(intersectionPoints) == 0:        
        return None, None
        
    idxClosest = int(dist[np.argmin(dist[:, 0])][1]) # get the index of the intersection point based on the minimal distance to the rib
    
    return idxClosest, intersectionPoints[np.argmin(dist[:, 0])]

def intermediaireGetIntersectionOrdering(listIdx, spinePoint, skinPoints):

        dist = np.zeros([len(listIdx), 2])
        for i, idx in enumerate(listIdx):
            dist[i] = distQuad(skinPoints[idx], spinePoint), idx    

        # Order intersection points by distance to the rib
        if len(listIdx) > 1:
            searchOrder = np.argsort(dist[:, 0])
        else:
            searchOrder = listIdx

        return searchOrder

def getIntersectionOrdering(listIdxNorth, listIdxSouth, spine, skinPoints):
        currentSpinePoint, previousSpinePoint = spine
        searchOrderNorth = intermediaireGetIntersectionOrdering(listIdxNorth, currentSpinePoint, skinPoints)
        searchOrderSouth = intermediaireGetIntersectionOrdering(listIdxSouth, previousSpinePoint, skinPoints)
            
        return searchOrderNorth, searchOrderSouth

def getPointsOfInterest(listIdxNorth, listIdxSouth, listIdxInflection):
    pointsOfInterests = np.array(merge_and_sort_arrays_with_indices([listIdxNorth, listIdxSouth, listIdxInflection]), dtype = "int64")
    return pointsOfInterests

def updateVerticesIncreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints):
    indexPt = (point[0]+1)%nbSkinPoints
    if previousPoint[0] < indexPt:                            
        for skinIdx in range(previousPoint[0], indexPt):
            vertices.append(skinPoints[skinIdx])
    elif previousPoint[0] > indexPt:
        for skinIdx in range(previousPoint[0], indexPt+nbSkinPoints):
            vertices.append(skinPoints[skinIdx%nbSkinPoints])
    
    return vertices

def updateVerticesDecreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints):
    
    indexPt = (point[0]-1)%nbSkinPoints
    if indexPt < previousPoint[0]:                            
        for skinIdx in range(previousPoint[0], indexPt, -1):
            vertices.append(skinPoints[skinIdx])
            #print(skinIdx)
    elif indexPt > (previousPoint[0]+1)%nbSkinPoints:
        for skinIdx in range((previousPoint[0]+1)%nbSkinPoints, indexPt):#+nbSkinPoints):
            vertices.append(skinPoints[skinIdx%nbSkinPoints])
            #print(skinIdx%nbSkinPoints)
    return vertices

def getVerticles(pointsOfInterest, idxToSearch, idxAtEndSearch, searchOrders, intersectionPoints, otherIntersectionLink, spine, variation, skinPoints):
    
    nbSkinPoints = len(skinPoints)
    spineLength = distQuad(spine[0], spine[1])
    vertices = []      
    currentSpinePoint, previousSpinePoint = spine[0], spine[1]      

    # ============================
    # check the ordering of intersection points
    
    # Quantity used to get the real intersection points
    nbStartingPointType = 1 # always start with one but increment it after the first iteration on the points of interest
    nbOtherPointType = -1
    
    # Get the closest intersection point from which the treatement of points of interest will begind     
    searchStartingIndex = np.where(pointsOfInterest == idxToSearch)[0][0]    
    startingPoint = pointsOfInterest[searchStartingIndex]
    
    # Get the closest intersection point on the other rib that whill serve as the stopping point
    searchStopIndex = np.where(pointsOfInterest == idxAtEndSearch)[0][0]
    searchStopIndexSkinPoint = pointsOfInterest[searchStopIndex][0]
    
    # Test if last inflection point is inside rib
    ribCheck = True

    # Add first spine Point
    if startingPoint[1] == 0:
        vertices.append(currentSpinePoint)
    else:
        vertices.append(previousSpinePoint)

    #print(startingPoint)
    # Used to stop adding unwanted points
    validityCheck = True
    
    

    # If we decrement the indexes
    if variation == -1:
        # =================================
        # New method
        
        # Test all points of interest starting from the closest smallest intersection between skin and rib
        nbPointOfInterest = len(pointsOfInterest[:, 0])
        for i in range(1, nbPointOfInterest+1):       
            point = pointsOfInterest[(searchStartingIndex-i)%(nbPointOfInterest)]
            previousPoint = pointsOfInterest[(searchStartingIndex -i+1)%(nbPointOfInterest)]
            #print(point, previousPoint)
            if (validityCheck == True):
                # Inflection type
                if (previousPoint[1] == 2):
                    projectedPointOntoSpine = findProjection(previousSpinePoint, currentSpinePoint, skinPoints[previousPoint[0]])
                    # Filtering out cases where the skin is outside out rib
                    condition1 = distQuad(previousSpinePoint, projectedPointOntoSpine) < spineLength
                    condition2 = distQuad(currentSpinePoint, projectedPointOntoSpine) < spineLength
                    if condition1 == True and condition2 == True:
                        # Adding all skin points between points of interest
                        #vertices = updateVerticesDecreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints)
                        ribCheck = True
                    else:
                        ribCheck = False                
                
                # If inflection type
                if (point[1] == 2):
                    projectedPointOntoSpine = findProjection(previousSpinePoint, currentSpinePoint, skinPoints[point[0]])
                    # Filtering out cases where the skin is outside out rib
                    condition1 = distQuad(previousSpinePoint, projectedPointOntoSpine) < spineLength
                    condition2 = distQuad(currentSpinePoint, projectedPointOntoSpine) < spineLength
                    if condition1 == True and condition2 == True:
                        ribCheck = True
                    else:
                        ribCheck = False 
                        
                    if (ribCheck == True):                        
                        # If last point is also a starting point, correct the indexation
                        if ((previousPoint[1] == startingPoint[1]) and (len(intersectionPoints) > previousPoint[1])):
                            vertices.append(intersectionPoints[previousPoint[1]][nbStartingPointType-1])
                            vertices = updateVerticesDecreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints)
                        # If last point other intersection point, correct the indexation
                        elif (previousPoint[1] == (startingPoint[1]+1)%2):
                            intersectionPoint = [sublist[0] for sublist in otherIntersectionLink if sublist[1] == previousPoint[0]]
                            if len(intersectionPoint) != 0:
                                intersectionPoint = intersectionPoint[0]
                                vertices.append(intersectionPoint)
                                vertices = updateVerticesDecreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints)

                        # If last point inflection point
                        else:
                            vertices = updateVerticesDecreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints)
                    else:
                        pass
                
                
                # Other intersection type
                elif (point[1] == (startingPoint[1]+1)%2):  
                    intersectionPoint = [sublist[0] for sublist in otherIntersectionLink if sublist[1] == point[0]]
                    if (len(intersectionPoint) != 0) and (len(intersectionPoints) > previousPoint[1]):
                        intersectionPoint = intersectionPoint[0]
                        if (ribCheck == True):                            
                        # If last point is also an intersection point, correct the indexation       
                            if (previousPoint[1] == startingPoint[1]):
                                vertices.append(intersectionPoints[previousPoint[1]][nbStartingPointType-1])
                                correctedIndexPoint = np.copy(point)
                                correctedIndexPoint[0] = correctedIndexPoint[0]+1
                                vertices = updateVerticesDecreasing(correctedIndexPoint, previousPoint, vertices, skinPoints, nbSkinPoints)
                                vertices.append(intersectionPoint)
                            else:
                                correctedIndexPoint = np.copy(point)
                                correctedIndexPoint[0] = correctedIndexPoint[0]+1
                                vertices = updateVerticesDecreasing(correctedIndexPoint, previousPoint, vertices, skinPoints, nbSkinPoints)
                                vertices.append(intersectionPoint)
        
                            # If the point is the closest to the rib, then stop going through the points of interest
                            if point[0] == searchStopIndexSkinPoint:
                                break
                            
                        elif (ribCheck == False):
                            vertices.append(intersectionPoint)
                            # If the point is the closest to the rib, then stop going through the points of interest
                            if point[0] == searchStopIndexSkinPoint:
                                break                         
                        else:
                            pass
                        nbOtherPointType -= 1 
                    
                else:
                    pass
                    
            # StartingPoint type
            if (point[1] == startingPoint[1]):                
                if (point[2] != nbStartingPointType):
                    validityCheck = False
                else:
                    validityCheck = True
                    if len(intersectionPoints) > point[1]:
                        # If we need to consider the vertices until the current point 
                        if (ribCheck == True):
                            correctedIndexPoint = np.copy(point)
                            correctedIndexPoint[0] = correctedIndexPoint[0]+1
                            vertices = updateVerticesDecreasing(correctedIndexPoint, previousPoint, vertices, skinPoints, nbSkinPoints)
                            vertices.append(intersectionPoints[point[1]][nbStartingPointType])
                            nbStartingPointType += 1
                        elif (ribCheck == False):
                            vertices.append(intersectionPoints[point[1]][nbStartingPointType])
                            nbStartingPointType += 1
                        else:
                            pass
                        ribCheck = True

                
                
    # If we incecrement the indexes
    
    if variation == 1:
        # =================================
        # New method
        
        # Test all points of interest starting from the closest smallest intersection between skin and rib
        nbPointOfInterest = len(pointsOfInterest[:, 0])
        for i in range(1, nbPointOfInterest+1):       
            point = pointsOfInterest[(searchStartingIndex+i)%(nbPointOfInterest)]
            previousPoint = pointsOfInterest[(searchStartingIndex +i-1)%(nbPointOfInterest)]
            #print(point, previousPoint)
            #print(intersectionPoints)
            if (validityCheck == True):
                # Inflection type
                if (previousPoint[1] == 2):
                    projectedPointOntoSpine = findProjection(previousSpinePoint, currentSpinePoint, skinPoints[previousPoint[0]])
                    # Filtering out cases where the skin is outside out rib
                    condition1 = distQuad(previousSpinePoint, projectedPointOntoSpine) < spineLength
                    condition2 = distQuad(currentSpinePoint, projectedPointOntoSpine) < spineLength
                    if condition1 == True and condition2 == True:
                        # Adding all skin points between points of interest
                        #vertices = updateVerticesDecreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints)
                        ribCheck = True
                    else:
                        ribCheck = False                
                
                # If inflection type
                if (point[1] == 2):
                    projectedPointOntoSpine = findProjection(previousSpinePoint, currentSpinePoint, skinPoints[point[0]])
                    # Filtering out cases where the skin is outside out rib
                    condition1 = distQuad(previousSpinePoint, projectedPointOntoSpine) < spineLength
                    condition2 = distQuad(currentSpinePoint, projectedPointOntoSpine) < spineLength
                    if condition1 == True and condition2 == True:
                        ribCheck = True
                    else:
                        ribCheck = False 
                        
                    if (ribCheck == True):
                        
                        # If last point is also a starting point, correct the indexation
                        if ((previousPoint[1] == startingPoint[1]) and (len(intersectionPoints) > previousPoint[1])):
                            vertices.append(intersectionPoints[previousPoint[1]][nbStartingPointType-1])
                            correctedIndexPreviousPoint = np.copy(previousPoint)
                            correctedIndexPreviousPoint[0] = correctedIndexPreviousPoint[0]+1
                            vertices = updateVerticesIncreasing(point, correctedIndexPreviousPoint, vertices, skinPoints, nbSkinPoints)
                            
                        # If last point other intersection point, correct the indexation
                        elif (previousPoint[1] == (startingPoint[1]+1)%2):
                            intersectionPoint = [sublist[0] for sublist in otherIntersectionLink if sublist[1] == previousPoint[0]]
                            if len(intersectionPoint) != 0:
                                intersectionPoint = intersectionPoint[0]
                                vertices.append(intersectionPoint)
                                vertices = updateVerticesIncreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints)

                        # If last point inflection point
                        else:
                            vertices = updateVerticesIncreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints)
                    else:
                        pass
                
                
                # Other intersection type
                elif (point[1] == (startingPoint[1]+1)%2):  
                    intersectionPoint = [sublist[0] for sublist in otherIntersectionLink if sublist[1] == point[0]]
                    if len(intersectionPoint) != 0:
                        intersectionPoint = intersectionPoint[0]
                        if (ribCheck == True):
                        # If last point is also an intersection point, correct the indexation       
                            if ((previousPoint[1] == startingPoint[1]) and (len(intersectionPoints) > previousPoint[1])):
                                vertices.append(intersectionPoints[previousPoint[1]][nbStartingPointType-1])
                                correctedIndexPreviousPoint = np.copy(previousPoint)
                                correctedIndexPreviousPoint[0] = correctedIndexPreviousPoint[0]+1
                                vertices = updateVerticesIncreasing(point, correctedIndexPreviousPoint, vertices, skinPoints, nbSkinPoints)
                                vertices.append(intersectionPoint)
                            else:
                                vertices = updateVerticesIncreasing(point, previousPoint, vertices, skinPoints, nbSkinPoints)
                                vertices.append(intersectionPoint)
                            # If the point is the closest to the rib, then stop going through the points of interest
                            if point[0] == searchStopIndexSkinPoint:
                                break
                            
                        elif (ribCheck == False):  
                            vertices.append(intersectionPoint)
                            # If the point is the closest to the rib, then stop going through the points of interest
                            if point[0] == searchStopIndexSkinPoint:
                                break                           
                        else:
                            pass
                        nbOtherPointType -= 1 
                else:
                    pass
                    
            # StartingPoint type
            if (point[1] == startingPoint[1]) :                
                if (point[2] != nbStartingPointType):
                    validityCheck = False
                else:
                    validityCheck = True
                    # If we need to consider the vertices after the current point 
                    if len(intersectionPoints) > point[1]:
                        if (ribCheck == True):
                            correctedIndexPreviousPoint = np.copy(previousPoint)
                            correctedIndexPreviousPoint[0] = correctedIndexPreviousPoint[0]+1
                            vertices = updateVerticesIncreasing(point, correctedIndexPreviousPoint, vertices, skinPoints, nbSkinPoints)
                            vertices.append(intersectionPoints[point[1]][nbStartingPointType])
                            nbStartingPointType += 1
                        elif (ribCheck == False):                        
                            vertices.append(intersectionPoints[point[1]][nbStartingPointType])
                            nbStartingPointType += 1
                        else:
                            pass
                        ribCheck = True
    
    # Add last spine Point
    if startingPoint[1] == 0:
        vertices.append(previousSpinePoint)
    else:
        vertices.append(currentSpinePoint)
        
    return vertices

def getVariation(oppositeSpineIdx, idxOfInterest, idxAtEndSearch, rib, spine, skinPoints):
    nbSKinPoints = len(skinPoints)
    
    spineLength = distQuad(spine[0], spine[1])
    
    # Project skinPoints onto this spine for the index adjacent to the closest rib/skin intersection
    projAroundIdxOfInterest = findProjection(spine[(oppositeSpineIdx+1)%2], spine[oppositeSpineIdx], skinPoints[(idxOfInterest+1)%nbSKinPoints])
    
    # Check how the projection of the skin is relative to the spine
    dist = np.zeros(2)
    dist[0] = distQuad(projAroundIdxOfInterest, spine[(oppositeSpineIdx+1)%2])
    dist[1] = distQuad(projAroundIdxOfInterest, spine[oppositeSpineIdx])
    #plt.plot(projAroundIdxOfInterest[0], projAroundIdxOfInterest[1], marker = "+")
    
    # If the projection of the next index point is on the spine
    if ((dist[0] < spineLength) and (dist[1] < spineLength)):
        variation = 1
    else:
        variation = -1

    return variation

def reorderListIntersection(intersections, idx, spinePoint):
        dist = np.zeros([len(intersections)])
        # Create list of distances
        for i, intersectionPoint in enumerate(intersections):
            dist[i] = distQuad(spinePoint, intersectionPoint) # List distance to the rib, idx of the current intersection point
        sortedDistance = np.argsort(dist)
        
        # Create sorted list
        sortedIntersections = np.zeros([len(intersections), 2])
        sortedIdx = np.empty([len(intersections)], dtype='int')
        for i, j in enumerate(sortedDistance):
            sortedIntersections[i] = intersections[j]
            sortedIdx[i] = idx[j]
        
        return sortedIntersections, sortedIdx

def getIntersectionLink(secondClosestIdxSpine, intersections, idx):
        """ Returns the intersections points and its index """
        link = []
        for i, pt in enumerate(intersections[secondClosestIdxSpine]):
            link.append([pt, idx[secondClosestIdxSpine][i]])
        return link 

def getBothAreas(previousSpinePoint, currentSpinePoint, skinPoints, display = False, displayDebug = False):
    
    # Number of skinpoints
    nbSKinPoints = len(skinPoints)
    
    # Compute the orthogonal line
    ribPointSouth = orthogonalLine(currentSpinePoint, previousSpinePoint, 0.02, 'West', 'South')
    ribPointNorth = orthogonalLine(currentSpinePoint, previousSpinePoint, 0.02, 'West', 'North') 
    
    # Compute the intersection w/ the skin
    idxListNorth = findIntersectionsWithSkin(currentSpinePoint, ribPointNorth, skinPoints)
    intersectionPointsNorth = getIntersectionsWithSkin(idxListNorth, ribPointNorth, currentSpinePoint, skinPoints)
    
    idxListSouth = findIntersectionsWithSkin(previousSpinePoint, ribPointSouth, skinPoints)
    intersectionPointsSouth = getIntersectionsWithSkin(idxListSouth, ribPointSouth, previousSpinePoint, skinPoints)
    
    """ for idx in idxListNorth:
        plt.plot(skinPoints[idx][0], skinPoints[idx][1], "r.")    
    for idx in idxListSouth:
        plt.plot(skinPoints[idx][0], skinPoints[idx][1], "k.")
        
    for pt in intersectionPointsNorth:
        plt.plot(pt[0], pt[1], "r+")    
    for pt in intersectionPointsSouth:
        plt.plot(pt[0], pt[1], "k+") """
    
    """ ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show() """

    # Segregate north and south index into west and east
    idxSouthEast, idxSouthWest = [], []
    idxNorthEast, idxNorthWest = [], []      

    intersectionsSouthEast, intersectionsSouthWest = [], []
    intersectionsNorthEast, intersectionsNorthWest = [], []        
    for i, idx in enumerate(idxListNorth):
        if np.dot(currentSpinePoint - np.array(intersectionPointsNorth[i]), currentSpinePoint - ribPointNorth) >= 0:
            idxNorthEast.append(idx)
            intersectionsNorthEast.append(intersectionPointsNorth[i])
        else:
            idxNorthWest.append(idx)    
            intersectionsNorthWest.append(intersectionPointsNorth[i])    
    for i, idx in enumerate(idxListSouth):
        if np.dot(previousSpinePoint - np.array(intersectionPointsSouth[i]), previousSpinePoint - ribPointSouth) >= 0:
            idxSouthEast.append(idx)
            intersectionsSouthEast.append(intersectionPointsSouth[i])
        else:
            idxSouthWest.append(idx)
            intersectionsSouthWest.append(intersectionPointsSouth[i])
    
    # Reordering the intersection list by closeness to the spine
    
    intersectionsNorthWest, idxNorthWest = reorderListIntersection(intersectionsNorthWest, idxNorthWest, currentSpinePoint)
    intersectionsNorthEast, idxNorthEast = reorderListIntersection(intersectionsNorthEast, idxNorthEast, currentSpinePoint)
    intersectionsSouthWest, idxSouthWest = reorderListIntersection(intersectionsSouthWest, idxSouthWest, previousSpinePoint)
    intersectionsSouthEast, idxSouthEast = reorderListIntersection(intersectionsSouthEast, idxSouthEast, previousSpinePoint)
    
    
    """ print("intersectionsNorthWest: ", intersectionsNorthWest)
    print("intersectionsNorthEast: ", intersectionsNorthEast)
    print("intersectionsSouthEast: ", intersectionsSouthEast)
    print("intersectionsSouthWest: ", intersectionsSouthWest) """
    # Display intersection points NW, NE, SW, SE 
    if (displayDebug == True):
        for pt in intersectionsNorthWest:
            plt.plot(pt[0], pt[1], marker = "*", color = "darkgreen")
        for pt in intersectionsNorthEast:
            plt.plot(pt[0], pt[1], marker = "*", color = "brown")
        for pt in intersectionsSouthWest:
            plt.plot(pt[0], pt[1], marker = "*", color = "midnightblue")
        for pt in intersectionsSouthEast:
            plt.plot(pt[0], pt[1], marker = "*", color = "purple")
            
    #==================================================================
    # Compute the orthogonal skin points w/ spine
    idxListInflectionPoints = findInflectionPoints(currentSpinePoint, ribPointNorth, skinPoints)

    idxInflectionWest, idxInflectionEast = segregateInflectionPoints(currentSpinePoint, idxListInflectionPoints, ribPointNorth, skinPoints)

    # Display inflexion points
    if (displayDebug == True): 
        for idx in idxInflectionWest:
            plt.plot(skinPoints[idx][0], skinPoints[idx][1], color = "orange", marker = "o")
        for idx in idxInflectionEast:
            plt.plot(skinPoints[idx][0], skinPoints[idx][1], color = "orange", marker = "+")
    
    """ print("idxInflectionWest: ", idxInflectionWest)
    print("idxInflectionEast: ", idxInflectionEast) """
    
    #==================================================================
    # Find the closest intersection point to the rib and the corresponding skinPoint index
        
    idxNW, intersectionNorthWest = findClosestIntersectionPointToRib(intersectionsNorthWest, currentSpinePoint, idxNorthWest)
    idxNE, intersectionNorthEast = findClosestIntersectionPointToRib(intersectionsNorthEast, currentSpinePoint, idxNorthEast)
    idxSW, intersectionSouthWest = findClosestIntersectionPointToRib(intersectionsSouthWest, previousSpinePoint, idxSouthWest)
    idxSE, intersectionSouthEast = findClosestIntersectionPointToRib(intersectionsSouthEast, previousSpinePoint, idxSouthEast)
    
    # If there are no intersection
    if ((idxNW == None) or (idxNE == None) or (idxSW == None) or (idxSE == None)):
        return None, None
    
    
    """ plt.plot([intersectionNorthWest[0], currentSpinePoint[0]], [intersectionNorthWest[1], currentSpinePoint[1]], linestyle = "-", color = "gold")
    plt.plot([intersectionSouthWest[0], previousSpinePoint[0]], [intersectionSouthWest[1], previousSpinePoint[1]], linestyle = "-", color = "red")
    
    plt.plot([skinPoints[idxNW][0], currentSpinePoint[0]], [skinPoints[idxNW][1], currentSpinePoint[1]], linestyle = "-", color = "orange")
    plt.plot([skinPoints[idxSW][0], previousSpinePoint[0]], [skinPoints[idxSW][1], previousSpinePoint[1]], linestyle = "-", color = "gray")
    
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show() """
    
    # Get the skinPoint index corresponding to the point just before the intersection point
    
    """ plt.plot(skinPoints[idxNW][0], skinPoints[idxNW][1], "r^")
    plt.plot(skinPoints[idxSW][0], skinPoints[idxSW][1], "rv")
    plt.plot(skinPoints[idxNE][0], skinPoints[idxNE][1], "k^")
    plt.plot(skinPoints[idxSE][0], skinPoints[idxSE][1], "kv")  """   
    
    #==================================================================
    # Creation of the vertices 
    # Spine segment length
    spine = [previousSpinePoint, currentSpinePoint]
    rib = [ribPointNorth, ribPointSouth]
    
    #==================================================================
    # Searching the closest intersection point to the rib 
    # West 
    
    matCompaDistWest = np.array([[distQuad(intersectionNorthWest, currentSpinePoint), idxNW], [distQuad(intersectionSouthWest, previousSpinePoint), idxSW]])     
    
    closestIdxSpineWest = np.argmin(matCompaDistWest[:, 0])  # Represents the side of the rib at which the intersection is the closest
    idxToSearchWest = int(matCompaDistWest[closestIdxSpineWest][1])  # Idx of the closest intersection

    secondClosestIdxSpineWest = (closestIdxSpineWest+1)%2
    idxAtEndSearchWest = int(matCompaDistWest[secondClosestIdxSpineWest][1])
    
    intersectionsWest = intersectionsNorthWest, intersectionsSouthWest
    idxWest = [idxNorthWest, idxSouthWest]
    otherIntersectionLinkWest = getIntersectionLink(secondClosestIdxSpineWest, intersectionsWest, idxWest)
    
    # East          
    matCompaDistEast = np.array([[distQuad(intersectionNorthEast, currentSpinePoint), idxNE], [distQuad(intersectionSouthEast, previousSpinePoint), idxSE]])     
    closestIdxSpineEast = np.argmin(matCompaDistEast[:, 0])
    idxToSearchEast = int(matCompaDistEast[closestIdxSpineEast][1])

    secondClosestIdxSpineEast = (closestIdxSpineEast+1)%2
    idxAtEndSearchEast = int(matCompaDistEast[secondClosestIdxSpineEast][1])
    
    intersectionsEast = intersectionsNorthEast, intersectionsSouthEast
    idxEast = [idxNorthEast, idxSouthEast]
    otherIntersectionLinkEast = getIntersectionLink(secondClosestIdxSpineEast, intersectionsEast, idxEast)
    
    """ plt.plot(skinPoints[idxToSearchEast][0], skinPoints[idxToSearchEast][1], "k^")
    plt.plot(skinPoints[idxAtEndSearchEast][0], skinPoints[idxAtEndSearchEast][1], "kv") """
    
    """ print("idxAtEndSearchEast: ", idxAtEndSearchEast)
    print("idxToSearchEast: ", idxToSearchEast) """
    
    # Get the spine point that is closest to the closest intersection    
    """ plt.plot(spine[closestIdxSpineWest][0], spine[closestIdxSpineWest][1], "r*")
    plt.plot(spine[closestIdxSpineEast][0], spine[closestIdxSpineEast][1], "k*") """
    
    spine = [currentSpinePoint, previousSpinePoint]  

    #==================================================================

    # West      
    # Get the sens of index variation 
    variationWest = getVariation(secondClosestIdxSpineWest, idxToSearchWest, idxAtEndSearchWest, rib, spine, skinPoints)
    #print("West : ", variationWest)   
    pointsOfInterest = getPointsOfInterest(idxNorthWest, idxSouthWest, idxInflectionWest)      
    searchOrders = getIntersectionOrdering(idxNorthWest, idxSouthWest, spine, skinPoints)
    intersectionPointsWest = [intersectionsNorthWest, intersectionsSouthWest]
    verticesWest = getVerticles(pointsOfInterest, idxToSearchWest, idxAtEndSearchWest, searchOrders, intersectionPointsWest, otherIntersectionLinkWest, spine, variationWest, skinPoints)
    areaWest = polygon_area(verticesWest)
    
    #==================================================================
    # East        
    # Get the sens of index variation 
    variationEast = getVariation(secondClosestIdxSpineEast, idxToSearchEast, idxAtEndSearchEast, rib, spine, skinPoints)
    #print("East : ", variationEast)
    pointsOfInterest = getPointsOfInterest(idxNorthEast, idxSouthEast, idxInflectionEast)      
    searchOrders = getIntersectionOrdering(idxNorthEast, idxSouthEast, spine, skinPoints)
    intersectionPointsEast = [intersectionsNorthEast, intersectionsSouthEast]
    verticesEast = getVerticles(pointsOfInterest, idxToSearchEast, idxAtEndSearchEast, searchOrders, intersectionPointsEast, otherIntersectionLinkEast, spine, variationEast, skinPoints)
    areaEast = polygon_area(verticesEast)
    
    
    # ==================
    # Display
    
    if display == True:
        X = [verticle[0] for verticle in verticesWest]
        Y = [verticle[1] for verticle in verticesWest]            
        plt.plot(X, Y, color = "green")
    
        X = [verticle[0] for verticle in verticesEast]
        Y = [verticle[1] for verticle in verticesEast]            
        plt.plot(X, Y, color = "midnightblue")
    
    #return 0, areaEast
    return areaWest, areaEast
    #return areaWest, 1

def criteriaFunction(theta, spineLength, previousSpinePoint, skinPoints):
    x = previousSpinePoint[0] + spineLength * np.cos(theta)
    y = previousSpinePoint[1] + spineLength * np.sin(theta)
    currentSpinePoint = np.array([x, y])
    #plt.plot(currentSpinePoint[0], currentSpinePoint[1], marker = "*")
    areaWest, areaEast = getBothAreas(previousSpinePoint, currentSpinePoint, skinPoints)
    
    if (areaWest != None):        
        criteria = 1*(areaWest - areaEast)**2 #+ 0.0001*(areaWest + areaEast)
    else:
        criteria = 1e10 # better than np.inf bc of solver used 
    return criteria

def getAngleBetweenLines(pt, circle):
    O = circle[0]
    radius = circle[1]
    
    projectionI = np.array([pt[0], O[1]])
    projectionQ = np.array([O[0], pt[1]])
    
    I = np.array([O[0]+radius, O[1]])
    Q = np.array([O[0], O[1]+radius])

    distI = np.sqrt(distQuad(projectionI, I))
    distQ = np.sqrt(distQuad(projectionQ, Q))
    """ plt.plot(I[0], I[1], "r+")
    plt.plot(Q[0], Q[1], "c+") """
    """ plt.plot(projectionI[0], projectionI[1], "c.")
    plt.plot(projectionQ[0], projectionQ[1], "c.") """
    
    #angle = np.arctan((pt[1]-O[1])/(pt[0]-O[0]))
    angle = np.arctan2(pt[1]-O[1], pt[0]-O[0])
    
    return angle

def getIntersectionCircleLine(circle_center, radius, point1, point2):

    h, k = circle_center
    x1, y1 = point1
    x2, y2 = point2
    
    # Coefficients for the quadratic equation
    dx = x2 - x1
    dy = y2 - y1
    A = dx**2 + dy**2
    B = 2 * (dx * (x1 - h) + dy * (y1 - k))
    C = (x1 - h)**2 + (y1 - k)**2 - radius**2
    
    # Discriminant
    D = B**2 - 4 * A * C
    
    if D < 0:
        # No intersection
        return []
    
    # Find the solutions for t
    D = B**2 - 4*A*C  # Discriminant
    t1 = (-B + np.sqrt(D)) / (2 * A)
    t2 = (-B - np.sqrt(D)) / (2 * A)
    
    intersections = []
    
    if 0 <= t1 <= 1:
        intersection = [x1 + t1 * dx, y1 + t1 * dy]
        intersections.append(intersection)
        
    if 0 <= t2 <= 1:
        intersection = [x1 + t2 * dx, y1 + t2 * dy]
        intersections.append(intersection)
    
    return intersections

def getIndexBounds(spineLength, previousSpinePoint, skinPoints, length_modifier):
    
    dist = np.zeros(len(skinPoints))
    for i, skinPoint in enumerate(skinPoints):
        dist[i] = distQuad(skinPoint, previousSpinePoint)
    
    index_dist_min = np.argmin(dist)  
    dist_min = dist[index_dist_min] 
    condition = False
    cpt = 0
    while condition == False:
        cpt += 1
        dist_target = dist - dist_min - length_modifier*spineLength
        signs = np.sign(dist_target)
        sign_changes = np.diff(signs)
        indexes = np.where(sign_changes != 0)[0]
        # Count the number of bounds that can be identified
        
        bool_array = np.zeros(4, dtype = 'bool')        
        dist = np.zeros([len(skinPoints), 0])
        distCompa = np.zeros(4)
        if len(indexes) == 4:            
            for j in range(4):
                # First method : find if intersection point is outside of fish
                intermediary_point = (skinPoints[indexes[j]] + skinPoints[indexes[j-1]])/2
                bool_array[j] = point_inside_polygon(intermediary_point, skinPoints)
                
                """ # Second method : exclude the bounds where the middle point is closest to the skin
                for i, skinPoint in enumerate(skinPoints):
                    dist[i] = distQuad(skinPoint, previousSpinePoint)
                index_dist_min = np.argmin(dist)  
                distCompa[j] = dist[index_dist_min] """
            num_identifiable = np.count_nonzero(~bool_array) # counts number of false
            #print(bool_array)
            
            if num_identifiable == 1:
                condition = True
            
        if cpt == 100:
            print(bool_array)
            print("idk something")
            for theta in range(0, 360, 2):
                plt.plot(previousSpinePoint[0] + length_modifier*spineLength*np.cos(theta*np.pi/180), previousSpinePoint[1] + length_modifier*spineLength*np.sin(theta*np.pi/180), color = "lightblue", marker = ".")
            display_fig()
            
        length_modifier += 0.01
    #print(length_modifier)
    # Go through the list of bounds and affect them if the middle point is outside of fish
    for i, test in enumerate(bool_array):
        if test == False:
            front = np.array([indexes[i-2], indexes[i-1]])
            back = np.array([indexes[(i)%4], indexes[(i+1)%4]])
            break
    
    for i, index in enumerate(indexes):
        plt.plot(skinPoints[index][0], skinPoints[index][1], "*", color = colors[i%len(colors)])

    return (front, back), length_modifier

def getSolutionSpan(spineLength, previousSpinePoint, skinPoints):
    length_modifier = 1
    bounds, length_modifier = getIndexBounds(spineLength, previousSpinePoint, skinPoints, length_modifier)
    
    while ((bounds == None) or (len(bounds) != 2)):
        bounds, length_modifier = getIndexBounds(length_modifier*spineLength, previousSpinePoint, skinPoints, length_modifier)
        #print(bounds)
        
        # watchdog
        if (length_modifier > 3):
            return None, length_modifier*spineLength

    return bounds, length_modifier

def generate_point(bounds):
    random_array = np.zeros([2])
    
    for i in range(2):
        lower_bound, upper_bound = bounds[i]
        random_array[i] = np.random.uniform(lower_bound, upper_bound)
        
    return random_array

def generate_segment(spine_length, skinPoints):
    min_x = np.min(skinPoints[:, 0])
    max_x = np.max(skinPoints[:, 0])
    min_y = np.min(skinPoints[:, 1])
    max_y = np.max(skinPoints[:, 1])
    point0 = generate_point([[min_x, max_x], [min_y, max_y]])
    while (point_inside_polygon(point0, skinPoints) is not True):
        point0 = generate_point([[min_x, max_x], [min_y, max_y]])
    
    angle = np.random.uniform(0, 2*np.pi)
    point1 = np.array([point0[0] + spine_length*np.cos(angle), point0[1] + spine_length*np.sin(angle)])
    while (point_inside_polygon(point1, skinPoints) is not True):
        angle = np.random.uniform(0, 2*np.pi)
        point1 = np.array([point0[0] + spine_length*np.cos(angle), point0[1] + spine_length*np.sin(angle)])

    return [point0, point1]

def get_angle(pt0, pt1):
    return np.arctan2(pt1[1]-pt0[1], pt1[0]-pt0[0])

# ===============================

def get_dist_mini(spine_segment, skinPoints):
    # Extend the spine segment in the wanted direction
    tmpIdx = findIntersectionsWithSkin(spine_segment[0], spine_segment[1], skinPoints)
    intersection_points = getIntersectionsWithSkin(tmpIdx, spine_segment[0], spine_segment[1], skinPoints)
    
    # Find the distance between the point that starts the spine estimation and skin
    # Get the right intersection
    dist_spine_point0 = np.zeros([len(intersection_points)])
    dist_spine_point1 = np.zeros([len(intersection_points)])
    for i, intersection_point in enumerate(intersection_points):
        dist_spine_point0[i] = distQuad(intersection_point, spine_segment[0])
        dist_spine_point1[i] = distQuad(intersection_point, spine_segment[1])
    
    # Get the distance to use it as estimation stop criteria
    dist_mini = np.inf
    for i, _ in enumerate(intersection_points):
        if dist_spine_point1[i] > dist_spine_point0[i]:
            dist = dist_spine_point0[i]                    
            if dist < dist_mini:
                dist_mini = dist
                index_mini = i
            
    #plt.plot(intersection_points[index_mini][0], intersection_points[index_mini][1], color = "gold", marker = "D")
    return dist_mini

def generatePointsCircle(center, angles, radius):
    # Calculate the x and y coordinates of the points on the circle
    points = []
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append(np.array([x, y]))
    return points

def get_spine(spine_segment, spineLength, skinPoints):
    
    spine = []
    spine.append(spine_segment[1])
    spine.append(spine_segment[0])
    
    """ print(spine_segment)
    plt.plot([spine_segment[0][0], spine_segment[1][0]], [spine_segment[0][1], spine_segment[1][1]], marker = "", linestyle = "-", color = "black")
    plt.plot([spine_segment[0][0]], [spine_segment[0][1]], "rX", linestyle = "-")
    plt.plot([ spine_segment[1][0]], [spine_segment[1][1]], "kX", linestyle = "-") """
    

    # Method
    k = 0

    dist_mini = get_dist_mini(spine_segment, skinPoints)
    dist_limit = spineLength/2
    
    while dist_mini >= min(dist_limit, spineLength):
        
        #print(k)
        #plt.plot([spine_segment[0][0], spine_segment[1][0]], [spine_segment[0][1], spine_segment[1][1]], marker = "*", color = colors[k%len(colors)], linestyle = "--")
        
        # ===========================
        # Find the intersection points (rib/skin) by construction a rib point
        rib_point = orthogonalLine(spine_segment[0], spine_segment[1], 0.2, "West", "North")    
        tmpIdx = findIntersectionsWithSkin(rib_point, spine_segment[0], skinPoints)
        intersection_points = getIntersectionsWithSkin(tmpIdx, rib_point, spine_segment[0], skinPoints)
        
        dist = np.zeros(len(intersection_points))
        for i, intersection_point in enumerate(intersection_points):
            #plt.plot(intersection_point[0], intersection_point[1], "r.")
            dist[i] = distQuad(intersection_point, spine_segment[1])
        
        dist_max = dist[np.argmax(dist)]
        
        # Update spine segment creation condition 
        dist_limit = dist_max
        
        # ===========================
        # Create intersection circle/skin based on intersection rib/skin   
        
        intersection_points = []
        j = 0
        for index in range(1, len(skinPoints)):
            tmp = getIntersectionCircleLine(spine_segment[0], dist_max, skinPoints[index], skinPoints[index-1])
            
            if (np.size(tmp) != 0):  
                if np.size(tmp) == 2:
                    intersection_points.append(tmp)
                else:
                    for intersection_point in tmp:
                        intersection_points.append([intersection_point])
                        #print("idk something")
        
        if np.size(intersection_points) < 2:
            #print("problem, no intersection points")
            break
        intersection_points = np.concatenate(intersection_points)
        
        
        # ===========================
        # Sort all intersection candidates and keep only the candidates furthest from previous spine point
        
        """ # Create virtual spine point far behind the spine segment to ensure robustness to fish shape
        virtual_spine_point = extendSegment(spine_segment[0], spine_segment[1], 1)
        plt.plot(virtual_spine_point[0], virtual_spine_point[1], marker = "p", color = "gold")
        
        # Check if virtual point is still inside the fish
        if point_inside_polygon(virtual_spine_point, skinPoints) == False:
            print("virtual point outside feesh")
            break """
        
        # Sort intersection points
        if len(intersection_points) > 2:
            dist = np.zeros([len(intersection_points)])
            for i, intersection_point in enumerate(intersection_points):
                #dist[i] = distQuad(virtual_spine_point, intersection_point)
                dist[i] = distQuad(spine_segment[1], intersection_point)
            index_biggest_dist = np.argpartition(dist, 2)[-2:]
        elif len(intersection_point) == 2:
            index_biggest_dist = [0, 1]
        else:
            print("no interfection found")
        
        """ plt.plot(intersection_points[index_biggest_dist[0]][0], intersection_points[index_biggest_dist[0]][1], "g*")
        plt.plot(intersection_points[index_biggest_dist[1]][0], intersection_points[index_biggest_dist[1]][1], "b*") """

        # ===========================
        # Changing points to angle
        
        angle_bounds = np.array([get_angle(spine_segment[0], intersection_points[index_biggest_dist[0]])%(2*np.pi), 
                        get_angle(spine_segment[0], intersection_points[index_biggest_dist[1]])%(2*np.pi)])
        angle_spine = np.array(get_angle(spine_segment[0], spine_segment[1]))%(2*np.pi)

        min_angle, max_angle = min(angle_bounds), max(angle_bounds)
        
        # Ensure the bounds are sorted
        if min_angle < angle_spine < max_angle :
            min_angle += 2*np.pi
            min_angle, max_angle = max_angle, min_angle
            #print("modif angle")
        
        bounds = [min_angle, max_angle]
        previousSpinePoint = spine_segment[0]
        res = minimize_scalar(criteriaFunction, args=(spineLength, previousSpinePoint, skinPoints), bounds=bounds)
        thetaOpti = res.x
        
        x = previousSpinePoint[0] + spineLength * np.cos(thetaOpti)
        y = previousSpinePoint[1] + spineLength * np.sin(thetaOpti)
        currentSpinePoint = (np.array([x, y]) + previousSpinePoint)/2
        
        """ 
        # Visualization for bounding space search iteration per iteration
        display_shape(skinPoints)
        
        points_circle = generatePointsCircle(spine_segment[0], np.linspace(min_angle, max_angle), dist_max)
        for point_circle in points_circle:
            plt.plot(point_circle[0], point_circle[1], ".", color =  "lightblue")
        
        
        # # All points that can be the intersection
        for intersection_point in intersection_points:
            plt.plot(intersection_point[0], intersection_point[1], "+", color = "darkgreen")
            
        points_circle = generatePointsCircle(spine_segment[0], angle_bounds, dist_max)
        for i, point_circle in enumerate(points_circle):
            plt.plot(point_circle[0], point_circle[1], color = ['red', 'red'][i], marker = "*")
        
        plt.plot(spine_segment[0][0], spine_segment[0][1], color = "gray", marker="*")
        plt.plot(spine_segment[1][0], spine_segment[1][1], "k*")
        plt.plot([spine_segment[0][0], spine_segment[1][0]], [spine_segment[0][1], spine_segment[1][1]], linestyle = "-", color = "gray")
        display_fig() """
        
        # Update spine segment
        spine_segment[1] = spine_segment[0]
        spine_segment[0] = currentSpinePoint
        spine.append(currentSpinePoint)
        # Update spine creation condition
        dist_mini = get_dist_mini(spine_segment, skinPoints)

        k += 1
    return spine 

def spine_core_estimation(spineLength, skinPoints):
    spine = [0, 0]
    while len(spine) <= 5:
        init_spine_segment = generate_segment(5, skinPoints)
        """ display_shape(skinPoints)
        plt.plot([init_spine_segment[0][0], init_spine_segment[1][0]], [init_spine_segment[0][1], init_spine_segment[1][1]], "k*", linestyle = "-")
        display_fig() """
        compa_spine = []
        # Go in both direction starting from the initial spine segment
        for direction in range(0, 2):
            spine_segment = [init_spine_segment[direction], init_spine_segment[(direction+1)%2]]
            spine = get_spine(spine_segment, spineLength, skinPoints)
            compa_spine.append(spine)
        """ display_shape(skinPoints)
        plt.plot([init_spine_segment[0][0], init_spine_segment[1][0]], [init_spine_segment[0][1], init_spine_segment[1][1]], "k*", linestyle = "-")
        display_discret_spine(compa_spine[0], unicolor="red")
        plt.plot([init_spine_segment[0][0], init_spine_segment[1][0]], [init_spine_segment[0][1], init_spine_segment[1][1]], "k*", linestyle = "-")
        display_fig()
        display_shape(skinPoints)
        plt.plot([init_spine_segment[0][0], init_spine_segment[1][0]], [init_spine_segment[0][1], init_spine_segment[1][1]], "k*", linestyle = "-")
        display_discret_spine(compa_spine[0], unicolor="red")
        display_discret_spine(compa_spine[1], unicolor="gold")
        plt.plot([init_spine_segment[0][0], init_spine_segment[1][0]], [init_spine_segment[0][1], init_spine_segment[1][1]], "k*", linestyle = "-")
        display_fig() """
        # Get the lengthiest spine 
        if len(compa_spine[0]) > len(compa_spine[1]):
            spine = compa_spine[0]
        else:
            spine = compa_spine[1]
            
    return spine

def direction_uniformization(spine, skinPoints):
    """ Tail is always index 0"""
    tailIndex = get_head(spine, skinPoints)    
    
    if tailIndex != 0:
        spine = np.flip(spine, axis=0)
    
    return spine

def get_final_spine(skinPoints, frame, spineLength):
    
    # Generate spine segment as long as the spine estimation fails
    spine = spine_core_estimation(spineLength, skinPoints)
    spine = get_spine([spine[-2], spine[-1]], spineLength, skinPoints)  
    watchdog = 0  
    while len(spine) <= 5:
        spine = spine_core_estimation(spineLength, skinPoints)
        spine = get_spine([spine[-2], spine[-1]], spineLength, skinPoints) 
        watchdog += 1
        if watchdog == 12:
            print("problem with frame {}".format(frame))
            break
    if watchdog > 1:
        print("regenerated frame {} {} times".format(frame, watchdog))
    # Always filter out last points
    spine = spine[1:-2]
    # Flip the spine to ensure the spine always start at tail
    spine = np.array(direction_uniformization(spine, skinPoints))
    
    """ display_shape(skinPoints)
    display_spine(spine, marker = "*")
    display_fig()
    #plt.plot(spine[0][0], spine[0][1], "rD")
    save_fig("spine_estimation{}".format(frame)) """
    np.save("spine_estimation{}.npy".format(frame), np.asarray(spine))
    
    return spine

def spine_generation(spineLength):
    print("spine generation starting...")
    frames = get_relevant_frames("yolo_edge")
    
    # Process each frame
    nb_frames = len(frames)
    for i, frame in enumerate(frames):
        if (i/nb_frames*100)//25 == 0:
            print("{}%".format(frame, i/nb_frames*100))
        try:
            # Use previous spine to generate a new one that is independant to initials conditions
            get_final_spine(frame, spineLength)
        except Exception:
            print("problem frame {}".format(frame))
            plt.clf()
    
    print("spine generation done")


if __name__ == "__main__":
    #os.chdir("C:/Users/Thomas Omarini/Documents/1 Stage_poisson/machine_perso/edge_consistency/output_files/tmp")
    os.chdir("C:/Users/Thomas/Documents/stage/edge_consistency/output_files/tmp")
    frame = 156
    spineLength = 10
    
    # Use previous spine to generate a new one that is independant to initials conditions
    get_final_spine(frame, spineLength)
    
    """ skinPoints = read_file("yolo_edge", frame)
    previousSpinePoint = np.array([1400, 617])
    currentSpinePoint = np.array([1400, 626])
    
    display_shape(skinPoints)
    getBothAreas(previousSpinePoint, currentSpinePoint, skinPoints, display = True, displayDebug = True)
    display_fig() """

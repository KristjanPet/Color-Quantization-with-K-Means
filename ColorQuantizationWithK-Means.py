import cv2
import numpy as np
import math
from numba import jit
import random


@jit(nopython=True)
def isMin(distance, i):
    if distance[i] == min(distance):
        return True
    return False

@jit(nopython=True)
def UpdateClusterCenters(k, img, dataPoints):
    height = img.shape[0]
    width = img.shape[1]

    newClusterCenters = np.zeros((k, 3))  # New cluster center positions
    numberOfColors = np.zeros(k)

    for x in range(width):
        for y in range(height):  # Iterate through the entire image
            colors = img[y, x]
            distance = np.zeros(k)

            for i in range(k):  # Calculate distance from each cluster to pixel colors
                for j in range(3):  # For all three color channels
                    distance[i] += pow((dataPoints[i][j] - colors[j]), 2)  # Square of the difference, Euclidean distance

                distance[i] = math.sqrt(distance[i])  # Absolute distance

            for i in range(k):  # Determine which cluster the pixel colors are closest to
                if isMin(distance, i):
                    newClusterCenters[i] += colors  # Sum the colors
                    numberOfColors[i] += 1  # Count of colors belonging to the cluster

    for i in range(k):  # Recalculate cluster values
        newClusterCenters[i] = newClusterCenters[i] / numberOfColors[i]

    return newClusterCenters

@jit(nopython=True)
def ColorizeImage(k, clusterCenters, image):
    height = image.shape[0]
    width = image.shape[1]

    coloredImage = np.zeros(image.shape, dtype=np.uint8)

    for x in range(width):
        for y in range(height):  # Iterate through the entire image
            colors = image[y, x]
            distance = np.zeros(k)

            for i in range(k):  # Calculate distance from each cluster to pixel colors
                for j in range(3):  # For all three color channels
                    distance[i] += pow((clusterCenters[i][j] - colors[j]), 2)  # Square of the difference, Euclidean distance

                distance[i] = math.sqrt(distance[i])  # Absolute distance

            for i in range(k):  # Determine which cluster the pixel colors are closest to
                if isMin(distance, i):
                    coloredImage[y, x] = clusterCenters[i]

    return coloredImage

# --------------MAIN-------------------------------------

k = 8  # Number of centers
numIterations = 40

img = cv2.imread('beach.jpg')

clusterCenters = np.zeros((k, 3))  # Cluster center locations

for y in range(k):
    for x in range(3):
        clusterCenters[y][x] = random.randint(1, 255)  # Set random cluster center locations

for x in range(numIterations):
    clusterCenters = UpdateClusterCenters(k, img, clusterCenters)

ColorizedImage = ColorizeImage(k, clusterCenters, img)

cv2.imshow("Original Image", img)
cv2.imshow("Colorized Image", ColorizedImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
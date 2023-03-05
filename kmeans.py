###############################################################################
#                                                                             #
# Script: This script runs the kmeans clustering algorithm on images for      #
#         image segmentation.                                                 #
#                                                                             #
#         I wrote this script some time ago (2021 to my best belief) as       #
#         a challenge to implement the k-means clustering algorithm for       #
#         image segmentation. This is certainly not the most optimal          #
#         solution but it's something I was/am quite proud of.                #
#                                                                             #
# Author: Khamis Buol (2023)                                                  #
#                                                                             #
###############################################################################

# %%
# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import color, io


# %%
# Initialising kmeans and associated functions
# def get_centroids_coordinates(centroids_dict):
#     """
#     The get_centroids_coordinates function simply returns the coordinates of
#     each centroid contained in a dictionary of coordinates.

#     :param centroids_dict: dictionary of centroids with centroid as key and
#     :return coords: array of pixel tuples for each coordinate
#     """
#     # Convert the dictionary of centroids into a list of keys
#     centroids = list(centroids_dict.keys())

#     # Extract only the pixel coordinates of each centroid
#     coords = [(centroid[-2], centroid[-1]) for centroid in centroids]
#     return coords


def convergence_is_reached(previous_centroids, current_centroids):
    """
    This function checks if we have reached convergence, which is when the
    new centroid location equals the previous centroid location

    :param previous_centroids: array of previous centroids
    :param current_centroids: array of current centroids
    :return: true if previous and current centroids are equal otherwise false
    """
    print("previous centroids:", previous_centroids)
    print("current_centroids:", current_centroids)
    return previous_centroids == current_centroids


# Randomly selects k centroids in an image
def initialise_centroids(I, k):
    """
    The initialise_centroids function is used to randomly select centroid
    positions on a given image. 

    :param I: input image
    :param k: number of centroids we want to initialise
    :return
    """

    # Get image dimensions
    y_, x_, _ = I.shape

    # Generate random pixel coordinates given image height and width
    xs = random.sample(range(0, x_), k)
    ys = random.sample(range(0, y_), k)

    centroids = dict()
    centroid_coordinates = list()
    for i in range(len(xs)):
        # Width and height values
        x = xs[i]
        y = ys[i]

        # Lab components of each pixel at y, x
        # L is the first component of each pixel (light)
        # a is the second component of each pixel (red/green value)
        # b is the third component of each pixel (blue/yellow value)
        L = I[y, x, 0]
        a = I[y, x, 1]
        b = I[y, x, 2]

        # Setting key as the pixel coordinate and
        # assigning list where neighbours will be placed
        key = (L, a, b)
        centroids[key] = list()
        centroid_coordinates.append(key)
    return centroids, centroid_coordinates


def nearest_centroid(pixel, centroids):
    """
    The nearest_centroid function finds and assigns the nearest centroid
    for a given pixel

    :param pixel: pixel location
    :param centroids: dictionary of centroids to compare
    :return nearest: the closest centroid to the pixel
    """
    centroid_keys = list(centroids.keys())
    nearest = centroid_keys[0]
    for centroid in centroid_keys:
        distance = euclidean_distance(pixel, centroid)
        if distance <= euclidean_distance(pixel, nearest):
            nearest = centroid
    return nearest


def euclidean_distance(pixel, centroid):
    """
    This function is used to calculate the Euclidean distance between a pixel
    and a given centroid

    :param pixel: a pixel value
    :param centroid: centroid value
    :return distance: the distance betwen the pixel and given centroid
    """
    pixel = np.array(pixel)
    centroid = np.array(centroid)

    distance = np.linalg.norm(pixel-centroid)
    return distance


def update_centroids(centroids):
    """
    This function is used to update the Lab values of centroids with the 
    average value. 

    :param centroids: dictionary of centroids to update
    :return output_dict: updated dictionary of centroids
    """
    output_dict = dict()
    for k, v in centroids.items():
        if len(v) > 0:
            average = list(np.mean(v, axis=0))
            L = float(average[0])
            a = float(average[1])
            b = float(average[2])
            new_key = (L, a, b)
            k = new_key
        output_dict[k] = v
    return output_dict


def kmeans(I, k):
    """
    This is the main implementation of the kmeans algorithm which segments
    an given image into k clusters. It uses Lab colour space for images. 

    :param I: image to segment
    :param l: number of clusters to segment the image into
    :return centroids: k centroids with which each pixel is assigned to 
    """

    # Get shape of input image
    y_, x_, _ = I.shape

    # Generate initial centroids
    centroids, previous_centroids = initialise_centroids(I, k)
    current_centroids_keys = list()
    counter = 0
    # while not convergence_is_reached(
    #         current_centroids_keys, previous_centroids):
    while counter != 5:
        for i in range(y_):
            for j in range(x_):
                L = I[i, j, 0]
                a = I[i, j, 1]
                b = I[i, j, 2]
                pixel = np.array([L, a, b])
                nearest = nearest_centroid(pixel, centroids)
                centroids[nearest].append(pixel)
        previous_centroids = current_centroids_keys
        centroids = update_centroids(centroids)
        current_centroids_keys = list(centroids.keys())
        print("counter", counter)
        counter += 1
    return centroids


# Function paints pixels of an image with average of each given centroid
def paint_image(I, centroids):
    """
    This function is used to re-draw an image with which the kmeans algorith
    has been executed over.

    :param I: input image
    :param centroids: centroids with which each pixel has been segmented into
    :return I: segmented image
    """
    y_, x_, _ = I.shape
    for i in range(y_):
        for j in range(x_):
            L = I[i, j, 0]
            a = I[i, j, 1]
            b = I[i, j, 2]
            pixel = np.array([L, a, b])
            nearest = nearest_centroid(pixel, centroids)
            new_L = nearest[0]
            new_a = nearest[1]
            new_b = nearest[2]
            I[i, j, 0] = new_L
            I[i, j, 1] = new_a
            I[i, j, 2] = new_b
    return I


# %%
# Select image
image = 'art_inspos/A567456A-32F1-4E71-98B4-27AFEEFCDDC3.jpg'

# %%
# Running kmeans algorithm
I1 = io.imread(image)
I1 = color.rgb2lab(I1)
k1 = 5
means = kmeans(I1, k1)

# %%
# Drawing resulting image
new_image = paint_image(I1, means)
new_image = color.lab2rgb(new_image)

# %%
plt.imshow(new_image)

# %%

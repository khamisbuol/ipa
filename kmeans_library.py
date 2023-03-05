###############################################################################
#                                                                             #
# Script: This script runs the kmeans clustering algorithm on images for      #
#         image segmentation using library functions.                         #
#                                                                             #
#         Most (if not all) of the code is from geeks for geeks:              #
# https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/  #
#                                                                             #
# Author: Khamis Buol (2023)                                                  #
#                                                                             #
###############################################################################

# %%
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %%
# Importing and reading image
path = 'art_inspos/DSCF1016.JPG'
image = cv2.imread(path)

# Changing colour space to RGB from BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %%
# Displaying imported image
plt.imshow(image)


# %%
# Reshaping image into a 2D array of pixels and 3 color values (RGB)
pixels = image.reshape((-1, 3))

# Converting to float type
pixels = np.float32(pixels)

# %%
# Running the kmeans algorithm

# Criteria for which the algorithm stops running - i.e., convergence is reached
# after 1000 iterations or epsilon becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# Performing k-means clustering with k randomly selected clusters
k = 8
retval, labels, centers = cv2.kmeans(
    pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Converting data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))

plt.imshow(segmented_image)

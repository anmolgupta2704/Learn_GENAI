#importing libraries
import numpy as np
import pandas as pd
import cv2 as cv
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from numpy import array
from sys import getsizeof

#Fetching the url and showing the image
urls = ["https://iiif.lib.ncsu.edu/iiif/0052574/full/800,/0/default.jpg"]

for url in urls:
    image = io.imread(url)

    # Show image using matplotlib (BEST for VS Code / Jupyter)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Image")
    plt.show()

    print('\n')

#Getting the multi-dimensional array from the image
array1 = array(image)

#Memory occupied by the multi-dimensional array
size1 = getsizeof(array1)
print("Multidimensional Array:\n", array1)

#Flatten the array (convert to 1D)
array2 = array1.flatten()

#Memory occupied by flattened array
size2 = getsizeof(array2)

#displaying the 1-D array
print("\nFlattened Array:\n", array2)

#Print sizes
print(f"\nSize of Multidimensional Image : {size1}")
print(f"Size of Flattened Image : {size2}")

difference = size1 - size2

#Print difference
print(f"Difference in memory: {difference}")
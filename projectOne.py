import cv2 as cv
from matplotlib import scale
import matplotlib.pyplot as plt
import numpy as np

import helpers as h

image = cv.imread('nebula-tempered.png')

colors = ('b', 'g', 'r')

new_img = image.copy()

# Build 0..255 lookup that wraps: (i - shift) % 256
table = np.array([(i - 200) % 256 for i in range(256)], dtype=np.uint8)

for c in (0, 2):  # 0=Blue, 2=Red (BGR)
    new_img[:, :, c] = cv.LUT(new_img[:, :, c], table)

new_img = h.luminanceDifferenceForGreen(new_img)

#new_img = h.rescaleFrame(new_img, scale=0.05)
#cv.imshow('Corrected Image', new_img)
#cv.waitKey(0)
#cv.destroyAllWindows()

for i, color in enumerate(colors):
    histogram = cv.calcHist([new_img], [i], None, [256], [0, 256])
    plt.plot(histogram, color=color)
    plt.xlim([0, 256])

plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.ylim([0, 1e7])
plt.show() 

equalized = h.histogram_equalization(new_img)

binary_img = h.treshholding(equalized, tresh=150)
#binary_img = h.rescaleFrame(binary_img, scale=0.05)

non_equalized_binary = h.treshholding(new_img, tresh=150)
"""non_equalized_binary = h.rescaleFrame(non_equalized_binary, scale=0.05)

new_img = h.rescaleFrame(new_img, scale=0.05)

# Display the results

cv.imshow('Equalized Image', new_img)
cv.imshow('Binary Image', binary_img)
cv.imshow('Non-Equalized Binary Image', non_equalized_binary)
cv.waitKey(0)
cv.destroyAllWindows()"""

# Show Count of Stars
# Question 2 final answer
print("Count of Stars (Equalized):", h.countStars(binary_img))
print("Count of Stars (Non-Equalized):", h.countStars(non_equalized_binary))

print("New Count of Stars (Equalized):", h.newCountStars(binary_img))
print("New Count of Stars (Non-Equalized):", h.newCountStars(non_equalized_binary))
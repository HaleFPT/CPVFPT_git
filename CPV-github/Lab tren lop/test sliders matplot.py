import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# Load image using OpenCV
image = cv2.imread('tattoo.jpg', cv2.IMREAD_GRAYSCALE)

# Create plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Make room for slider

# Display image
im = ax.imshow(image, cmap='binary', vmin=0, vmax=255)

# Create slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
threshold_slider = Slider(ax=ax_slider,
                           label='Threshold',
                           valmin=0,
                           valmax=255,
                           valinit=127)

# Define function to apply threshold to create binary image
def apply_threshold(threshold):
    binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    im.set_data(binary_image)
    fig.canvas.draw_idle()

# Register callback with slider
threshold_slider.on_changed(apply_threshold)

# Show plot
plt.show()

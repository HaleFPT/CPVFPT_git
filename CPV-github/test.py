import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def apply_segment(threshold, im, image, gray, min_markers, max_markers, fig):
    # Threshold image
    ret, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    # Generate markers using connected components
    num_markers, markers = cv.connectedComponents(thresh)

    # Set markers outside range to 0
    markers[(markers < min_markers) | (markers > max_markers)] = 0

    # Apply watershed segmentation
    markers = cv.watershed(image, markers)

    # Apply color map to markers for visualization
    markers_display = cv.applyColorMap(np.uint8(markers), cv.COLORMAP_TWILIGHT_SHIFTED)

    # Display segmented image
    im.set_data(markers_display)
    fig.canvas.draw_idle()


def segment_sliders():
    # Load image using OpenCV
    image = cv.imread('images/tattoo flowers.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (640, 480))
    image = cv.medianBlur(image, 3)

    # Convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Create plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)  # Make room for sliders

    # Display image
    im = ax.imshow(image)

    # Create sliders
    ax_threshold = plt.axes([0.25, 0.15, 0.65, 0.03])  # [left, bottom, width, height]
    threshold_slider = Slider(ax=ax_threshold,
                              label='Threshold',
                              valmin=0,
                              valmax=255,
                              valinit=127)

    ax_min_markers = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    min_markers_slider = Slider(ax=ax_min_markers,
                                label='Min markers',
                                valmin=0,
                                valmax=255,
                                valinit=100)

    ax_max_markers = plt.axes([0.25, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    max_markers_slider = Slider(ax=ax_max_markers,
                                label='Max markers',
                                valmin=0,
                                valmax=255,
                                valinit=200)

    # Register callbacks with sliders
    threshold_slider.on_changed(
        lambda threshold: apply_segment(threshold, im, image, gray, min_markers_slider.val, max_markers_slider.val,
                                        fig))
    min_markers_slider.on_changed(
        lambda min_markers: apply_segment(im, image, gray, int(threshold_slider.val), min_markers,
                                          max_markers_slider.val,
                                          fig))
    max_markers_slider.on_changed(
        lambda max_markers: apply_segment(im, image, gray, int(threshold_slider.val), min_markers_slider.val,
                                          max_markers,
                                          fig))

    # Show plot
    plt.show()


if __name__ == '__main__':
    segment_sliders()

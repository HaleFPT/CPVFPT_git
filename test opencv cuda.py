import cv2

count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)
'''
# Load image from file
img = cv2.imread('CPV-github/Faces/DangNM.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
# Create GpuMat object from image data
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img)

# Perform thresholding on GPU
_, gpu_thresh = cv2.cuda.threshold(gpu_img, 120, 255, cv2.THRESH_BINARY)

# Download result from GPU memory
thresh = gpu_thresh.download()

# Display result
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
'''


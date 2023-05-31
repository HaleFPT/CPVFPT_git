'''
Function 1: color balance, to perform this function, the user needs to enter the necessary parameters to perform color balance. (can use the slider to represent it visually)
Function 2: Show histogram and enter the necessary information to perform histogram equalization.
Function 3: implement the median filter to remove noise in the image(salt and pepper noise)
Function 4: implement the Mean filter to remove noise in image (salt and pepper noise)
Function 5. implement Fourier transform to create high frequent noise in imgpath and use low pass filter (LPF) to remove noise.
'''

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
from scipy import signal
from scipy.fft import fft2, ifft2

img_path = '../CPVFPT_git/CPV-github/images/Save = Follow Amyio.jpg'
img = cv.imread(img_path)



# Function 1: color balance, to perform this function, the user needs to enter the necessary parameters to perform color balance. (can use the slider to represent it visually)
def nothing(x):
    pass  # --- window to have all the controls


def color_balance(img):
    name = "Controlers"
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, l = cv.split(img_hsv)
    cv.namedWindow(name)
    cv.createTrackbar("H", name, 0, 255, nothing)
    # warning: overflow saturation at 255
    cv.createTrackbar("S", name, 0, 255, nothing)
    cv.createTrackbar("L", name, 0, 255, nothing)
    out = img.copy()

    while True:
        h = cv.getTrackbarPos("H", name)
        s = cv.getTrackbarPos("S", name)
        l = cv.getTrackbarPos("L", name)

        if h > 0:
            lim = 255 - h
            out[:, :, 0][img_hsv[:, :, 0] > lim] = 255
            out[:, :, 0][img_hsv[:, :, 0] <= lim] = (h + img_hsv[:, :, 0][img_hsv[:, :, 0] <= lim]).astype(
                img_hsv.dtype)
        else:
            lim = 0 - h
            out[:, :, 0][img_hsv[:, :, 0] < lim] = 0
            out[:, :, 0][img_hsv[:, :, 0] >= lim] = (h + img_hsv[:, :, 0][img_hsv[:, :, 0] >= lim]).astype(
                img_hsv.dtype)

        if s > 0:
            lim = 255 - s
            out[:, :, 1][img_hsv[:, :, 1] > lim] = 255
            out[:, :, 1][img_hsv[:, :, 1] <= lim] = (s + img_hsv[:, :, 1][img_hsv[:, :, 1] <= lim]).astype(
                img_hsv.dtype)
        else:
            lim = 0 - s
            out[:, :, 1][img_hsv[:, :, 1] < lim] = 0
            out[:, :, 1][img_hsv[:, :, 1] >= lim] = (s + img_hsv[:, :, 1][img_hsv[:, :, 1] >= lim]).astype(
                img_hsv.dtype)

        if l > 0:
            lim = 255 - l
            out[:, :, 2][img_hsv[:, :, 2] > lim] = 255
            out[:, :, 2][img_hsv[:, :, 2] <= lim] = (l + img_hsv[:, :, 2][img_hsv[:, :, 2] <= lim]).astype(
                img_hsv.dtype)
        else:
            lim = 0 - l
            out[:, :, 2][img_hsv[:, :, 2] < lim] = 0
            out[:, :, 2][img_hsv[:, :, 2] >= lim] = (l + img_hsv[:, :, 2][img_hsv[:, :, 2] >= lim]).astype(
                img_hsv.dtype)

        out = cv.cvtColor(out, cv.COLOR_HSV2BGR)
        cv.imshow("Original", img)
        cv.imshow("Color Balance", out)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


# Clips all channels identically.This preserves the overall color relationship while making highlights appear lighter and shadows appear darker.The Auto Contrast command uses this algorithm.
def Enhance_Monochromatic_Contrast(img_path):
    lab= cv.cvtColor(img_path, cv.COLOR_BGR2LAB)
    l_chanel, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_chanel)
    limg = cv.merge((cl, a, b))
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return final


# Maximizes the tonal range in each channel to produce a more dramatic correction. Because each channel is adjusted individually, Enhance Per Channel Contrast may remove or introduce color casts. The Auto Tone command uses this algorithm.
def Enhance_Per_Channel_Contrast(image):
    img = image.copy()
    b, g, r = cv.split(img)  # split the image into its RGB channels
    r = cv.equalizeHist(r)  # enhance contrast in the red channel
    g = cv.equalizeHist(g)  # enhance contrast in the green channel
    b = cv.equalizeHist(b)  # enhance contrast in the blue channel
    img = cv.merge((b, g, r))  # merge the enhanced channels back into the image
    return img


# Finds the average lightest and darkest pixels in an image and uses them to maximize contrast while minimizing clipping. The Auto Color command uses this algorithm.
def Find_Dark_and_Light_Colors(image):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Find the darkest and lightest pixels
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    avg_dark = minVal
    avg_light = maxVal

    # Calculate the scale factor
    scale_factor = 255 / (avg_light - avg_dark)

    # Shift the image values so that the darkest pixel is at 0 and the lightest pixel is at 255
    image = (image - avg_dark) * scale_factor

    # Clip any values that are above 255 or below 0
    image = np.clip(image, 0, 255)

    # Convert the image back to 8-bit unsigned integers
    image = image.astype("uint8")

    return image


# Function 2: Show histogram and enter the necessary information to perform histogram equalization.
# CLAHE (Contrast Limited Adaptive Histogram Equalization) is a method of histogram equalization that improves the contrast of an image by performing histogram equalization on small regions of the image. This method is useful for imgpath with high contrast and low contrast areas.
def histogram_equalizor(img):
    # Convert the image to grayscale
    image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create a CLAHE object
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply the CLAHE algorithm to the grayscale image
    equalized = clahe.apply(gray)
    # show histogram of original image vs equalized image
    fig, axs = plt.subplots(1, 4, figsize=(10, 10))
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[1].imshow(gray, cmap='gray')
    axs[1].set_title('Grayscale Image')
    axs[2].hist(gray.ravel(), 256, [0, 256])
    axs[2].set_title('Histogram of Grayscale Image')
    axs[3].hist(equalized.ravel(), 256, [0, 256])
    axs[3].set_title('Histogram of Equalized Image')
    plt.show()
    return equalized



# salt and pepper noise
def salt_pepper_noise(img):
    noise_img = random_noise(img, mode='s&p', amount=0.3)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    cv.imshow('salt and pepper noise', noise_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return noise_img

# Function 3: implement the median filter to remove noise in the image(salt and pepper noise
def median_filter(img):
    median = cv.medianBlur(img, 5)
    cv.imshow('median', median)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function 4: implement the Mean filter to remove noise in image (salt and pepper noise)
def mean_filter(img):
    mean = cv.blur(img, (5, 5))
    cv.imshow('mean', mean)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function 5. implement Fourier transform to create high frequent noise in imgpath and use low pass filter (LPF) to remove noise
def show_magnitude_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum.astype(np.float32)

def low_pass_filter(image, cutoff_frequency=50):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = 1
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

'''def addSinunoidNoise(img):
    """
    add sinunoid noise demo
    :return:
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fft_shift = fshift
    rows,cols = fft_shift.shape
    fft_mag = 10*np.log(np.abs(fft_shift))
    #process freq spectrogram, create high response at 4 high freqs
    max_val = np.max(fft_shift)
    #print(max_val)
    delta = 5
    fft_shift[rows//4-delta:rows//4+delta,cols//4-delta:cols//4+delta] =max_val
    fft_shift[rows *3//4 - delta:rows*3//4 + delta, cols // 4 - delta:cols // 4 + delta] = max_val
    fft_shift[rows *3//4 - delta:rows*3//4 + delta, cols *3// 4 - delta:cols *3// 4 + delta] = max_val
    fft_shift[rows //4 - delta:rows//4 + delta, cols *3// 4 - delta:cols *3// 4 + delta] = max_val
    fft_noise_mag = 10 * np.log(np.abs(fft_shift))
    fft_ishift = np.fft.ifftshift(fft_shift)
    img_back = np.fft.ifft2(fft_ishift)
    img_back = np.abs(img_back).astype(np.uint8)
    cv.imshow('src',img)
    cv.imshow('src freq',fft_mag.astype(np.uint8))
    cv.imshow('noise img',img_back)
    cv.imshow('noise img freq',fft_noise_mag.astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img_back'''


def addSinunoidNoise(img, bShow=False):
    height, width = img.shape[:2]
    sin_map = np.sin(np.arange(height * width) * 2 * np.pi / height).reshape(height, width)
    noise = (sin_map * 255).astype(np.uint8)
    noisy_img = cv.add(img, noise)

    if bShow:
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(noisy_img, cmap='gray')
        axs[1].set_title('Noisy Image')
        plt.show()

    return noisy_img


def reconstruct_image(image, cutoff_frequency=50):
    filtered_image = low_pass_filter(image, cutoff_frequency)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Reconstructed Image')
    plt.show()




def fourier_transform():
    image = cv.imread(img_path, 0)
    image = addSinunoidNoise(image)
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Noisy Image')
    axs[1].imshow(show_magnitude_spectrum(image), cmap='gray')
    axs[1].set_title('Magnitude Spectrum')
    axs[2].imshow(low_pass_filter(image), cmap='gray')
    axs[2].set_title('Reconstructed Image')
    plt.show()

'''    color_balance(img)
    EMC = Enhance_Monochromatic_Contrast(img)
    EPCC = Enhance_Per_Channel_Contrast(img)
    FDLC = Find_Dark_and_Light_Colors(img)

    histogram_equalizor(img)
    median_filter(img)
    mean_filter(img)
    fourier_transform(img)
    
    cv.imshow('Enhance_Monochromatic_Contrast', EMC)
    cv.imshow('Enhance_Per_Channel_Contrast', EPCC)
    cv.imshow('Find_Dark_and_Light_Colors', FDLC)

    cv.waitKey(0)
    cv.destroyAllWindows()'''



def menu():
    print("1. Color Balance")
    print("2. Histogram Equalization")
    print("3. Median Filter")
    print("4. Mean Filter")
    print("5. Fourier Transform")
    print("6. Exit")
    print("|==================================================|")
    print("|===================  For fun:  ===================|")
    print("|==================================================|")
    print("|==============| 7. EMC, EPCC, FDLC |==============|")
    print("|===========| 8. Salt and pepper nois |============|")
    print("|==================================================|")
    print("Please select a function: ")

    choice = int(input())
    if 1 > choice or choice > 8:
        print("Invalid choice")
        menu()
    else:
        return choice
    return choice


def main():
    global noise_img, img
    choice = menu()

    if choice == 1:
        color_balance(img)
        main()
    elif choice == 2:
        histogram_equalizor(img)
        main()
    elif choice == 3:
        median_filter(noise_img)
        main()
    elif choice == 4:
        mean_filter(noise_img)
        main()
    elif choice == 5:
        fourier_transform()
        main()
    elif choice == 6:
        exit()
    elif choice == 7:
        #for fun
        EMC = Enhance_Monochromatic_Contrast(img)
        EPCC = Enhance_Per_Channel_Contrast(img)
        FDLC = Find_Dark_and_Light_Colors(img)
        cv.imshow('Enhance_Monochromatic_Contrast', EMC)
        cv.imshow('Enhance_Per_Channel_Contrast', EPCC)
        cv.imshow('Find_Dark_and_Light_Colors', FDLC)
        cv.waitKey(0)
        cv.destroyAllWindows()
        main()
    elif choice == 8:
        noise_img = salt_pepper_noise(img)
        main()
    else:
        print("Invalid choice")
        main()


if __name__ == '__main__':
    main()

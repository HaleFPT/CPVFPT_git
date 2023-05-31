import cv2 as cv
import numpy as np
"""
Learn more from here:
https://docs.opencv.org/3.4/de/dbc/tutorial_py_fourier_transform.html
"""
def fourierTranform(img,bShow=False):
    """
    show fourier transform of img
    :param img: img to be analysed
    :param bShow: whether to show fourier transform
    :return: fourier tranform of img
    """
    #img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
    #assert img is not None, "file could not be read, check with os.path.exists()"
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    if bShow:
        cv.imshow('src',img)
        cv.imshow('fft',magnitude_spectrum)
        cv.waitKey(0)
    return fshift

def fourierTransform_Demo():
    """
    Demo fourier transform
    :return:
    """
    img = cv.imread('Save = Follow Amyio.jpg',cv.IMREAD_GRAYSCALE)
    fft = fourierTranform(img)
    cv.imshow('src',img)
    magnitude_spectrum = 10*np.log(np.abs(fft))
    cv.imshow('fft',magnitude_spectrum.astype(np.uint8))
    cv.waitKey(0)

def addSinunoidNoise(img,bShow=False):
    """
    add sinunoid high freq noise into image
    :param img: img to be added
    :param type: high = add high freq, low = add low freq, mid=mid freq
    :return: noised added
    """
    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2
    # create a mask first, center square is 1, remaining all zeros
    #mask[int(rows/4), int(cols/4)] = 255
    #mask[int(rows* 3 / 4), int(cols * 3/ 4)] = 255
    #mask[int(rows  / 4), int(cols * 3 / 4)] = 255
    #mask[int(rows * 3 / 4), int(cols  / 4)] = 255
    # apply mask and inverse DFT
    dft_shift = fourierTranform(img)
    #dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    #dft_shift = np.fft.fftshift(dft)
    max_val = np.max(dft_shift)
    print(max_val)
    fshift = dft_shift
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    if bShow:
        cv.imshow('src',img)
        cv.imshow('noise',img_back)
        cv.waitKey(0)
    return img_back.astype(np.uint8)


def addSinunoidNoise_Demo():
    """
    add sinunoid noise demo
    :return:
    """
    img = cv.imread('Save = Follow Amyio.jpg',cv.IMREAD_GRAYSCALE)
    fft_shift = fourierTranform(img)
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



def imageReconstructionDemo():
    """
    Demo for image reconstruction in slide
    :return:
    """
    img = cv.imread('Save = Follow Amyio.jpg', cv.IMREAD_GRAYSCALE)
    fft_shift = fourierTranform(img)
    rows, cols = fft_shift.shape
    fft_mag = 10 * np.log(np.abs(fft_shift))
    # process freq spectrogram, create high response at 4 high freqs
    max_val = np.max(fft_shift)

    #draw stars in freq domain to create the noises image
    r = 3 * np.minimum(rows,cols)/8
    delta = 2
    for rad in np.linspace(0,2*np.pi,6):
        col = int(r*np.cos(rad)+cols/2)
        row = int(r*np.sin(rad)+rows/2)
        fft_shift[row-delta:row+delta ,col] = max_val
        fft_shift[row, col-delta:col+delta] = max_val


    fft_noise_mag = 10 * np.log(np.abs(fft_shift))
    fft_ishift = np.fft.ifftshift(fft_shift)
    img_back = np.fft.ifft2(fft_ishift)
    img_back = np.abs(img_back).astype(np.uint8)
    cv.imshow('src', img)
    cv.imshow('src freq', fft_mag.astype(np.uint8))
    cv.imshow('noise img', img_back)
    cv.imshow('noise img freq', fft_noise_mag.astype(np.uint8))
    cv.waitKey(0)
    # use band pass filter to filter out this
    filter_kernel = np.ones(img.shape,np.uint8)

    def createCircleMask(img, distanceFromCenter, circleValue=1):
        """
        helper function for this function. create circle mask inside img
        :param img: img to be masked
        :param distanceFromCenter: radius of the circle
        :param circleValue: value of the circle ( either 0 or 1)
        :return: image with circle mask
        """
        mask = np.zeros_like(img)
        rows,cols = img.shape
        for row in range(rows):
            row_2_center = row-rows/2
            for col in range(cols):
                col_2_center = col - cols/2
                dist = np.sqrt(row_2_center**2 + col_2_center**2)
                if dist<distanceFromCenter:
                    mask[row,col] = circleValue
                else:
                    mask[row, col] = 1 - circleValue
        return mask

    delta = 4
    small_circle_mask = createCircleMask(img,r-delta,1)
    big_cicle_mask = createCircleMask(img,r+delta,0)
    mask =small_circle_mask+big_cicle_mask
    cv.imshow('mask',mask*255)
    cv.waitKey(0)
    fft_shift*=mask
    fft_filter_noise_mag = 10 * np.log(np.abs(fft_shift))
    cv.imshow('filtered noise img freq', fft_filter_noise_mag.astype(np.uint8))

    fft_ishift = np.fft.ifftshift(fft_shift)
    img_back = np.fft.ifft2(fft_ishift)
    img_back = np.abs(img_back).astype(np.uint8)

    cv.imshow('filtered img', img_back)
    cv.waitKey(0)


def main():
    fourierTransform_Demo()
    #addSinunoidNoise_Demo()
    #imageReconstructionDemo()

main()
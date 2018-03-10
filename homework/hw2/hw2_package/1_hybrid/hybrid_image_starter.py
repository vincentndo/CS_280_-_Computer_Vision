import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from align_image_code import align_images
import numpy as np
from scipy.signal import convolve2d
from skimage import color


def standard_gaussian(X, Y, sigma):
    return 1 / (2 * np.pi * sigma**2) ** 0.5 * np.exp(-(X**2. + Y**2.) / (2 * sigma**2))


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2):
    filter_factor = 15
    threshold_1 = sigma1
    threshold_2= sigma2
    std_1 = 5
    std_2 = 5
    x = np.arange(-filter_factor, filter_factor + 1)
    y = np.arange(-filter_factor, filter_factor + 1)
    X, Y = np.meshgrid(x, y)

    low_pass_filter = standard_gaussian(X, Y, std_1)
    low_pass_filter /= np.sum(low_pass_filter)

    impulse_filter = np.zeros((2 * filter_factor + 1, 2 * filter_factor + 1))
    impulse_filter[filter_factor, filter_factor] = 1
    high_pass_filter = standard_gaussian(X, Y, std_2)
    high_pass_filter /= np.sum(high_pass_filter)
    high_pass_filter = impulse_filter - high_pass_filter

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, low_pass_filter, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("gaussian_low_pass_filter.jpg")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, high_pass_filter, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("gaussian_high_pass_filter.jpg")

    im1_filtered = np.zeros(im1_aligned.shape)
    im2_filtered = np.zeros(im2_aligned.shape)

    for i in range(3):
        im1_filtered[:, :, i] = convolve2d(im1_aligned[:, :, i], low_pass_filter, mode="same")
        im1_filtered_fft = np.fft.fft2(im1_filtered[:, :, i])
        # im1_filtered_fft[(threshold_1 + 1):(-threshold_1), (threshold_1 + 1):(-threshold_1)] = 0
        im1_filtered_fft[(threshold_1 + 1):(-threshold_1), :] = 0
        im1_filtered_fft[:, (threshold_1 + 1):(-threshold_1)] = 0
        im1_filtered[:, :, i] = np.real(np.fft.ifft2(im1_filtered_fft))

        im2_filtered[:, :, i] = convolve2d(im2_aligned[:, :, i], high_pass_filter, mode="same")
        im2_filtered_fft = np.fft.fft2(im2_filtered[:, :, i])
        # im2_filtered_fft[1:(threshold_2+1), 1:(threshold_2+1)] = 0
        # im2_filtered_fft[-threshold_2:, -threshold_2:] = 0
        im2_filtered_fft[1:(threshold_2+1), :] = 0
        im2_filtered_fft[-threshold_2:, :] = 0
        im2_filtered_fft[:, 1:(threshold_2+1)] = 0
        im2_filtered_fft[:, -threshold_2:] = 0
        im2_filtered[:, :, i] = np.real(np.fft.ifft2(im2_filtered_fft))

    im1_filtered = normalize(im1_filtered)
    plt.imsave(common_name + "im1_filtered.jpg", im1_filtered, format="jpg")
    plt.figure()
    plt.imshow(im1_filtered)
    plt.show()
    im1_filtered_fft_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(im1_filtered)))))
    plt.imsave(common_name + "im1_filtered_fft_log.jpg", im1_filtered_fft_log, format="jpg")
    plt.imshow(im1_filtered_fft_log)
    plt.show()
    im2_filtered = normalize(im2_filtered)
    plt.imsave(common_name + "im2_filtered.jpg", im2_filtered, format="jpg")
    plt.imshow(im2_filtered)
    plt.show()
    im2_filtered_fft_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(im2_filtered)))))
    plt.imsave(common_name + "im2_filtered_fft_log.jpg", im2_filtered_fft_log, format="jpg")
    plt.imshow(im2_filtered_fft_log)
    plt.show()
    hybrid = 0.5 * im1_filtered + 0.5 * im2_filtered
    return normalize(hybrid)


if __name__ == "__main__":

    image_pairs = [("malik.png", "papadimitriou.png"), ("cat.png", "dog.png"), ("boy.png", "girl.png")]

    for im1_name, im2_name in image_pairs:

        common_name = im1_name.split('.')[0] + '+' + im2_name.split('.')[0] + '_'

        # First load images

        # high sf
        im1 = plt.imread(im1_name)
        im1 = im1[:,:,:3]
        im1_input_fft_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(im1)))))
        plt.imsave(common_name + "im1_input_fft_log.jpg", im1_input_fft_log, format="jpg")
        plt.imshow(im1_input_fft_log)
        plt.show()

        # low sf
        im2 = plt.imread(im2_name)
        im2 = im2[:,:,:3]
        im2_input_fft_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(im2)))))
        plt.imsave(common_name + "im2_input_fft_log.jpg", im2_input_fft_log, format="jpg")
        plt.imshow(im2_input_fft_log)
        plt.show()

        # Next align images (this code is provided, but may be improved)
        im1_aligned, im2_aligned = align_images(im1, im2)

        ## You will provide the code below. Sigma1 and sigma2 are arbitrary 
        ## cutoff values for the high and low frequencies

        sigma1 = 10
        sigma2 = 5
        hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

        hybrid_fft_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(hybrid)))))
        plt.imsave(common_name + "hybrid_fft_log.jpg", hybrid_fft_log, format="jpg")
        plt.imshow(hybrid_fft_log)
        plt.show()
        plt.imsave(common_name + "hybrid.jpg", hybrid, format="jpg")
        plt.imshow(hybrid)
        plt.show()

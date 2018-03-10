import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.signal import convolve2d


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def standard_gaussian(X, Y, a, b, c, d, sigma_1, sigma_2):
    return 1 / (np.pi * (sigma_1**2 + sigma_2**2)) ** 0.5 * np.exp( -(a * X + b * Y) ** 2. / (2 * sigma_1**2) - (c * X + d * Y) ** 2. / (2 * sigma_2**2))


def difference_filter(I):
    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    I_filtered_Dx = np.zeros(I.shape)
    I_filtered_Dy = np.zeros(I.shape)
    for i in range(3):
        I_filtered_Dx[:, :, i] = convolve2d(I[:, :, i], Dx, mode="same")
        I_filtered_Dy[:, :, i] = convolve2d(I[:, :, i], Dy, mode="same")

    return I_filtered_Dx, I_filtered_Dy


def derivative_gaussian_filter(I,sigma):
    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    filter_factor = 15
    x = np.arange(-filter_factor, filter_factor + 1)
    y = np.arange(-filter_factor, filter_factor + 1)
    X, Y = np.meshgrid(x, y)

    gaussian_filter = standard_gaussian(X, Y, 1, 0, 0, 1, sigma, sigma)
    gaussian_filter_Dx = convolve2d(gaussian_filter, Dx, mode="same")
    gaussian_filter_Dy = convolve2d(gaussian_filter, Dy, mode="same")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_Dx, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("derivative_gaussian_filter_Dx.jpg")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_Dy, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("derivative_gaussian_filter_Dy.jpg")

    I_filtered_Dx = np.zeros(I.shape)
    I_filtered_Dy = np.zeros(I.shape)
    for i in range(3):
        I_filtered_Dx[:, :, i] = convolve2d(I[:, :, i], gaussian_filter_Dx, mode="same")
        I_filtered_Dy[:, :, i] = convolve2d(I[:, :, i], gaussian_filter_Dy, mode="same")

    return I_filtered_Dx, I_filtered_Dy


def oriented_filter(I):
    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])
    Dxy = np.array([[0, 1], [-1, 0]])
    Dyx = np.array([[1, 0], [0, -1]])

    filter_factor = 15
    x = np.arange(-filter_factor, filter_factor + 1)
    y = np.arange(-filter_factor, filter_factor + 1)
    X, Y = np.meshgrid(x, y)

    gaussian_filter_x = standard_gaussian(X, Y, 1, 0, 0, 1, 2, 6)
    gaussian_filter_Dx = convolve2d(gaussian_filter_x, Dx, mode="same")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_x, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("oriented_gaussian_filter_Dx.jpg")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_Dx, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("derivative_oriented_gaussian_filter_Dx.jpg")

    gaussian_filter_y = standard_gaussian(X, Y, 1, 0, 0, 1, 6, 2)
    gaussian_filter_Dy = convolve2d(gaussian_filter_y, Dy, mode="same")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_y, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("oriented_gaussian_filter_Dy.jpg")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_Dy, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("derivative_oriented_gaussian_filter_Dy.jpg")

    gaussian_filter_xy = standard_gaussian(X, Y, 1, -1, 1, 1, 2, 6)
    gaussian_filter_Dxy = convolve2d(gaussian_filter_xy, Dxy, mode="same")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_xy, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("oriented_gaussian_filter_Dxy.jpg")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_Dxy, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("derivative_oriented_gaussian_filter_Dxy.jpg")

    gaussian_filter_yx = standard_gaussian(X, Y, 1, -1, 1, 1, 6, 2)
    gaussian_filter_Dyx = convolve2d(gaussian_filter_yx, Dyx, mode="same")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_yx, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("oriented_gaussian_filter_Dyx.jpg")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, gaussian_filter_Dyx, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.savefig("derivative_oriented_gaussian_filter_Dyx.jpg")

    I_filtered_Dx = np.zeros(I.shape)
    I_filtered_Dy = np.zeros(I.shape)
    I_filtered_Dxy = np.zeros(I.shape)
    I_filtered_Dyx = np.zeros(I.shape)
    for i in range(3):
        I_filtered_Dx[:, :, i] = convolve2d(I[:, :, i], gaussian_filter_Dx, mode="same")
        I_filtered_Dy[:, :, i] = convolve2d(I[:, :, i], gaussian_filter_Dy, mode="same")
        I_filtered_Dxy[:, :, i] = convolve2d(I[:, :, i], gaussian_filter_Dxy, mode="same")
        I_filtered_Dyx[:, :, i] = convolve2d(I[:, :, i], gaussian_filter_Dyx, mode="same")

    return I_filtered_Dx, I_filtered_Dy, I_filtered_Dxy, I_filtered_Dyx


if __name__ == "__main__":

    image_name_list = ["bsds_3096.jpg", "bsds_300091.jpg", "bsds_253027.jpg", "bsds_156065.jpg", "bsds_101087.jpg"]

    for image_name in image_name_list:
        name = image_name.split('.')[0] + "_"
        im = plt.imread(image_name)

        # Simple derivative filters

        im_filtered_Dx, im_filtered_Dy = difference_filter(im)

        im_filtered_Dx_normalized = normalize(im_filtered_Dx)
        plt.imsave(name + "im_filtered_Dx.jpg", im_filtered_Dx_normalized, format="jpg")
        plt.imshow(im_filtered_Dx_normalized)
        plt.show()

        im_filtered_Dy_normalized = normalize(im_filtered_Dy)
        plt.imsave(name + "im_filtered_Dy.jpg", im_filtered_Dy_normalized, format="jpg")
        plt.imshow(im_filtered_Dy_normalized)
        plt.show()

        im_gradient_magnitude = (im_filtered_Dx ** 2 + im_filtered_Dy ** 2) ** 0.5
        im_gradient_magnitude_2d = np.sum(im_gradient_magnitude, axis=2)
        plt.imsave(name + "im_gradient_magnitude.jpg", im_gradient_magnitude_2d, format="jpg", cmap="gray")
        plt.imshow(im_gradient_magnitude_2d, cmap="gray")
        plt.show()

        im_gradient_orientation = np.angle(im_filtered_Dx + 1j * im_filtered_Dy)
        im_gradient_orientation[np.where(im_gradient_magnitude < 10)] = 0
        im_gradient_orientation_2d = np.sum(im_gradient_orientation, axis=2)
        plt.imsave(name + "im_gradient_orientation.jpg", im_gradient_orientation_2d, format="jpg")
        plt.imshow(im_gradient_orientation_2d)
        plt.show()

        # Derivative Gaussian filters

        im_gaussian_filtered_Dx, im_gaussian_filtered_Dy = derivative_gaussian_filter(im, 3)
        im_gaussian_filtered_Dx_normalized = normalize(im_gaussian_filtered_Dx)
        plt.imsave(name + "im_gaussian_filtered_Dx.jpg", im_gaussian_filtered_Dx_normalized, format="jpg")
        plt.figure()
        plt.imshow(im_gaussian_filtered_Dx_normalized)
        plt.show()

        im_gaussian_filtered_Dy_normalized = normalize(im_gaussian_filtered_Dy)
        plt.imsave(name + "im_gaussian_filtered_Dy.jpg", im_gaussian_filtered_Dy_normalized, format="jpg")
        plt.imshow(im_gaussian_filtered_Dy_normalized)
        plt.show()

        im_gaussian_gradient_magnitude = (im_gaussian_filtered_Dx ** 2 + im_gaussian_filtered_Dy ** 2) ** 0.5
        im_gaussian_gradient_magnitude_2d = np.sum(im_gaussian_gradient_magnitude, axis=2)
        plt.imsave(name + "im_gaussian_gradient_magnitude.jpg", im_gaussian_gradient_magnitude_2d, cmap="gray")
        plt.imshow(im_gaussian_gradient_magnitude_2d, cmap="gray")
        plt.show()

        im_gaussian_gradient_orientation = np.angle(im_gaussian_filtered_Dx + 1j * im_gaussian_filtered_Dy)
        im_gaussian_gradient_orientation[np.where(im_gaussian_gradient_magnitude < 20)] = 0
        im_gaussian_gradient_orientation_2d = np.sum(im_gaussian_gradient_orientation, axis=2)
        plt.imsave(name + "im_gaussian_gradient_orientation.jpg", im_gaussian_gradient_orientation_2d)
        plt.imshow(im_gaussian_gradient_magnitude_2d)
        plt.show()

        # Oriented Gaussian filters

        im_oriented_gaussian_filtered_Dx, im_oriented_gaussian_filtered_Dy, im_oriented_gaussian_filtered_Dxy, im_oriented_gaussian_filtered_Dyx = oriented_filter(im)
        im_oriented_gaussian_filtered_Dx_normalized = normalize(im_oriented_gaussian_filtered_Dx)
        plt.imsave(name + "im_oriented_gaussian_filtered_Dx.jpg", im_oriented_gaussian_filtered_Dx_normalized, format="jpg")
        plt.figure()
        plt.imshow(im_oriented_gaussian_filtered_Dx_normalized)
        plt.show()

        im_oriented_gaussian_filtered_Dy_normalized = normalize(im_oriented_gaussian_filtered_Dy)
        plt.imsave(name + "im_oriented_gaussian_filtered_Dy.jpg", im_oriented_gaussian_filtered_Dy_normalized, format="jpg")
        plt.imshow(im_oriented_gaussian_filtered_Dy_normalized)
        plt.show()

        im_oriented_gaussian_filtered_Dxy_normalized = normalize(im_oriented_gaussian_filtered_Dxy)
        plt.imsave(name + "im_oriented_gaussian_filtered_Dxy.jpg", im_oriented_gaussian_filtered_Dxy_normalized, format="jpg")
        plt.imshow(im_oriented_gaussian_filtered_Dxy_normalized)
        plt.show()

        im_oriented_gaussian_filtered_Dyx_normalized = normalize(im_oriented_gaussian_filtered_Dyx)
        plt.imsave(name + "im_oriented_gaussian_filtered_Dyx.jpg", im_oriented_gaussian_filtered_Dyx_normalized, format="jpg")
        plt.imshow(im_oriented_gaussian_filtered_Dyx_normalized)
        plt.show()

        im_oriented_gaussian_gradient_magnitude = (im_oriented_gaussian_filtered_Dx ** 2 + im_oriented_gaussian_filtered_Dy ** 2
                                                    + im_oriented_gaussian_filtered_Dxy ** 2 + im_oriented_gaussian_filtered_Dyx ** 2) ** 0.5
        im_oriented_gaussian_gradient_magnitude_2d = np.sum(im_oriented_gaussian_gradient_magnitude, axis=2)
        plt.imsave(name + "im_oriented_gaussian_gradient_magnitude.jpg", im_oriented_gaussian_gradient_magnitude_2d, format="jpg", cmap="gray")
        plt.imshow(im_oriented_gaussian_gradient_magnitude_2d, cmap="gray")
        plt.show()

        im_oriented_gaussian_gradient_orientation = np.angle(im_oriented_gaussian_filtered_Dx + 1j * im_oriented_gaussian_filtered_Dy) \
                                                    + np.angle(im_oriented_gaussian_filtered_Dxy + 1j * im_oriented_gaussian_filtered_Dyx)
        im_oriented_gaussian_gradient_orientation[np.where(im_oriented_gaussian_gradient_magnitude < 20)] = 0
        im_oriented_gaussian_gradient_orientation_2d = np.sum(im_oriented_gaussian_gradient_orientation, axis=2)
        plt.imsave(name + "im_oriented_gaussian_gradient_orientation.jpg", im_oriented_gaussian_gradient_orientation_2d, format="jpg")
        plt.imshow(im_oriented_gaussian_gradient_orientation_2d)
        plt.show()

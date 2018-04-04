import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.io as scio
import sklearn.preprocessing as skpp
import numpy.linalg as nla
import sys


def reconstruct_3d(name):
    """ Homework 3: 3D reconstruction from two Views

    This function takes as input the name of the image pairs (i.e. 'house' or 
    library') and returns the 3D points as well as the camera matrices... but
    some functions are missing.

    NOTES
    (1) The code has been written so that it can be easily understood. It has
    not been written for efficiency.
    (2) Don't make changes to this main function since I will run my
    reconstruct_3d.m and not yours. I only want from you the missing functions
    and they should be able to run without crashing with my reconstruct_3d.m
    (3) Keep the names of the missing functions as they are defined here,
    otherwise things will crash
    """

    ##--------Load images, K matrices and matches--------
    data_dir = "../data/" + name

    # images
    I1 = plt.imread(data_dir + '/' + name + "1.jpg")
    I2 = plt.imread(data_dir + '/' + name + "2.jpg")

    # K matrices
    qq = scio.loadmat(data_dir + '/' + name + "1_K.mat")
    K1 = qq['K']
    del qq
    qq = scio.loadmat(data_dir + '/' + name + "2_K.mat")
    K2 = qq['K']
    del qq

    # corresponding points
    matches = np.loadtxt(data_dir + '/' + name + "_matches.txt")
    # this is a N x 4 where:
    # matches(i, 0:2) is a point in the first image
    # matches(i, 2:) is the corresponding point in the second image

    # visualize matches (disable or enable this whenever you want)
    if True:
        plt.figure()
        plt.imshow(np.concatenate((I1, I2), axis=1))
        plt.plot(matches[:, 0], matches[:, 1], '+r')
        plt.plot(matches[:, 2] + I1.shape[1], matches[:, 3], '+r')

        for i in range(0, matches.shape[0]):
            X = np.concatenate((matches[i, [0]], matches[i, [2]] + I1.shape[1]))
            Y = matches[i, [1, 3]]
            plt.plot(X, Y, 'r')

        plt.show()

    #---------------------------------------------------------------------------
    ##--------Find fundamental matrix--------

    # F: the 3x3 fundamental matrix,
    # res_err: mean squered distance between points in the two images and their
    # corresponding epipolar lines
    F, res_err = fundamental_matrix(matches);

    print("Residual in F = %f", res_err)

    # the essential matrix
    E = K2.T.dot(F).dot(K1)

    #---------------------------------------------------------------------------
    ##--------Rotation and translation of camera 2--------

    # R: list of possible rotation matrices of second camera
    # t: list of the possible translation vectors of second camera
    R, t = find_rotation_translation(E)
    
    # Find R2, t2 from R, t such that largest number of points lie in front of
    # the image planes of the two cameras

    P1 = K1.dot( np.concatenate( (np.identity(3), np.zeros( (3, 1) )), axis=1) )

    # the number of points
    num_points = np.zeros( (len(t), len(R)) )

    # the reconstruction error for all combinations
    errs = np.full( (len(t), len(R)), np.inf )

    for it in range(len(t)):
        t2 = t[it]
        for ir in range(len(R)):
            R2 = R[ir]

            P2 = K2.dot( np.concatenate( (R2, t2.reshape(-1, 1)), axis=1 ) )

            points_3d, errs[it, ir] = find_3d_points(matches, P1, P2)

            Z1 = points_3d[:, 2]
            Z2 = R2[2, :].dot(points_3d.T) + t2[2]
            Z2 = Z2.T
            num_points[it, ir] = np.sum( np.logical_and(Z1 > 0, Z2 > 0) )

    its, irs = np.where( num_points == np.max(num_points) )

    # pick onw out the best combinations
    j = 0

    print("Reconstruction error = %f", errs[its[j], irs[j]])

    t2 = t[its[j]]
    R2 = R[irs[j]]
    P2 = K2.dot( np.concatenate( (R2, t2.reshape(-1, 1)), axis=1 ) )

    # compute the 3D points with the final P2
    points, _ = find_3d_points(matches, P1, P2)

    ##--------plot points and centers of cameras--------

    plot_3d(points, R2, t2)

    return points, P1, P2


def linear_transformation(X):
    means = np.mean(X, axis=1)
    standard_deviations = np.std(X, axis=1)
    standard_deviations[2] = -1
    transformation_matrix = np.diag(1 / standard_deviations)
    transformation_matrix[:, 2] = (-1) * means / standard_deviations
    X_standardized = np.dot(transformation_matrix, X)
    return X_standardized, transformation_matrix


def fundamental_matrix(matches):
    n = matches.shape[0]
    X1 = np.concatenate((matches[:, [0,1]].T, np.ones((1, n))), axis=0)
    X2 = np.concatenate((matches[:, [2,3]].T, np.ones((1, n))), axis=0)
    X1_standardized, T1 = linear_transformation(X1)
    X2_standardized, T2 = linear_transformation(X2)
    A = np.concatenate((X1_standardized[[0], :] * X2_standardized[[0], :],
                        X1_standardized[[1], :] * X2_standardized[[0], :],
                        X2_standardized[[0], :],
                        X1_standardized[[0], :] * X2_standardized[[1], :],
                        X1_standardized[[1], :] * X2_standardized[[1], :],
                        X2_standardized[[1], :],
                        X1_standardized[[0], :],
                        X1_standardized[[1], :],
                        np.ones((1, n))), axis=0)
    A = A.T
    U_A, s_A, V_A_transpose = nla.svd(A)
    f = V_A_transpose.T[:, -1]
    F_fullrank = f.reshape(3, 3)
    U_F, s_F, V_F_transpose = nla.svd(F_fullrank)
    s_F[-1] = 0
    F_rank2 = U_F.dot(np.diag(s_F)).dot(V_F_transpose)
    F_ret = T2.T.dot(F_rank2).dot(T1)

    residual = 0
    for i in range(n):
        x1 = X1[:, i]
        x2 = X2[:, i]
        residual += x1.dot(F_ret.T).dot(x2) ** 2 / nla.norm(F_ret.T.dot(x2)) ** 2 \
                    + x2.dot(F_ret).dot(x1) ** 2 / nla.norm(F_ret.dot(x1)) ** 2

    residual /= 2 * n

    print("Fundamental matrix:")
    print(F_ret)
    print("Residual:")
    print(residual)

    return F_ret, residual


def find_rotation_translation(E):
    U, s, V_transpose = nla.svd(E)
    t = U[:, -1]
    t_list = [t, -t]

    R_pos_90_degree = np.array([[0, -1, 0],
                                [1,  0, 0],
                                [0,  0, 1]])
    R_neg_90_degree = np.array([[ 0, 1, 0],
                                [-1, 0, 0],
                                [ 0, 0, 1]])
    R_list = []
    R1 = U.dot(R_pos_90_degree.T).dot(V_transpose)
    if abs(nla.det(R1) - 1) <= 1e-5:
        R_list.append(R1)
    
    R2 = R1 * (-1)
    if abs(nla.det(R2) - 1) <= 1e-5:
        R_list.append(R2)

    R3 = U.dot(R_neg_90_degree.T).dot(V_transpose)
    if abs(nla.det(R3) - 1) <= 1e-5:
        R_list.append(R3)
    
    R4 = R3 * (-1)
    if abs(nla.det(R4) - 1) <= 1e-5:
        R_list.append(R4)

    print("Possible rotation matrices:")
    print(R_list)
    print("Possible translation vector:")
    print(t_list)

    return R_list, t_list


def find_3d_points(matches, P1, P2):
    n = matches.shape[0]
    points = np.ones( (n, 4) )
    P_pair = [P1, P2]

    for i in range(n):
        x_pair = [matches[i, 0:2], matches[i, 2:]]

        A = []
        b = []
        for j in range(2):
            P = P_pair[j]
            x = x_pair[j]
            A.append( [ P[0, 0] - x[0] * P[2, 0],
                        P[0, 1] - x[0] * P[2, 1],
                        P[0, 2] - x[0] * P[2, 2] ] )
            A.append( [ P[1, 0] - x[1] * P[2, 0],
                        P[1, 1] - x[1] * P[2, 1],
                        P[1, 2] - x[1] * P[2, 2] ] )
            b.append(x[0] * P[2, 3] - P[0, 3])
            b.append(x[1] * P[2, 3] - P[1, 3])

        A = np.array(A)
        b = np.array(b)
        points[i, :-1] = nla.inv( A.T.dot(A) ).dot(A.T).dot(b)

    rec_err = 0
    for i in range(n):
        x1 = matches[i, 0:2]
        x2 = matches[i, 2:]
        x1_rec_homo = P1.dot(points[i, :])
        x1_rec_homo /= x1_rec_homo[-1]
        x1_rec = x1_rec_homo[:-1]
        x2_rec_homo = P2.dot(points[i, :])
        x2_rec_homo /= x2_rec_homo[-1]
        x2_rec = x2_rec_homo[:-1]
        rec_err += nla.norm(x1 - x1_rec) + nla.norm(x2 - x2_rec)

    rec_err /= 2 * n

    print("Reconstruction error:")
    print(rec_err)
        
    return points[:, :-1], rec_err


def plot_3d(points, R2, t2):
    C1 = np.zeros(3)
    C2 = -nla.inv(R2).dot(t2)

    ax = plt.axes(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 0])

    ax.scatter(C1[0], C1[1], C1[2], c='r', marker='^')
    ax.text(C1[0], C1[1], C1[2], 'C_1')
    ax.scatter(C2[0], C2[1], C2[2], c='r', marker='^')
    ax.text(C2[0], C2[1], C2[2], 'C_2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        reconstruct_3d(sys.argv[1])
    else:
        print("Sample Input 1: house")
        reconstruct_3d("house")
        print()
        print("Sample Input 2: library")
        reconstruct_3d("library")

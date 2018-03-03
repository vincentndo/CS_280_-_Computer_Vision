import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

times_square = "./hw1_package/hw1_package/images/times_square.jpg"
hometown = "./hw1_package/hw1_package/images/Rach-Gia-City-worth-exploring.jpg"
my_pic = "./hw1_package/hw1_package/images/my_pic.jpg"
computer_screen = "./hw1_package/hw1_package/images/computer_screen.png"
flagellation = "./hw1_package/hw1_package/images/the_flagellation.jpg"

times_square_surfaces = [[[497, 810], [499, 893], [611, 801], [610, 894]],
							[[155, 581], [241,617], [284, 530], [355, 575]],
							[[676, 363], [727, 430], [866, 279], [887, 370]],
							[[726, 1124], [664, 1193], [934, 1174], [916, 1270]],
							[[511, 1229], [352, 1388], [701, 1308], [564, 1530]],
							[[14, 311], [152, 403], [153, 155], [319, 278]]]

hometown_surfaces = [[[300, 1344], [236, 1561], [619, 1348], [586, 1565]],
						[[1081, 827], [1065, 1147], [1267, 828], [1228, 1147]]]

computer_screen_surface = [[203, 634], [302, 864], [575, 690], [826, 903]]
computer_screen_surface_rectified = [[0, 0], [0, 300], [200, 0], [200, 300]]

flagellation_surface = [[627, 280], [626, 567], [676, 84], [676, 578]]
flagellation_surface_rectified = [[0, 0], [0, 200], [600, 0], [600, 200]]

my_pic_surface = [[0, 0], [0, 719], [719, 0], [719, 719]]

def affine_solve(u, v):
	X = []

	for j in range(u.shape[1]):
		X_temp = [[u[0,j], u[1,j], 1, 0, 0, 0],
					[0, 0, 0, u[0,j], u[1,j], 1]]
		X.extend(X_temp)

	X = np.array(X)
	y = v.T.flatten()
	h = LA.inv(X.T.dot(X)).dot(X.T).dot(y)
	h = np.concatenate((h, np.array([0, 0, 1])))
	H = h.reshape(3, 3)
	return H


def homography_solve(u, v):
	X = []

	for j in range(u.shape[1]):
		X_temp = [[u[0,j], u[1,j], 1, 0, 0, 0, -u[0,j]*v[0,j], -u[1,j]*v[0,j]],
					[0, 0, 0, u[0,j], u[1,j], 1, -u[0,j]*v[1,j], -u[1,j]*v[1,j]]]
		X.extend(X_temp)

	X = np.array(X)
	y = v.T.flatten()
	h = LA.inv(X.T.dot(X)).dot(X.T).dot(y)
	h = np.concatenate((h, np.array([1])))
	H = h.reshape(3, 3)
	return H

def homography_transform(u, H):
	U = np.concatenate([u, np.ones([1, u.shape[1]])])
	V = H.dot(U)
	for j in range(V.shape[1]):
		V[:, j] = V[:, j] / V[2, j]
	v = V[:-1,:].astype(int)
	print(v)
	return v


if __name__ == "__main__":
	times_square_im = plt.imread(times_square)
	hometown_im = plt.imread(hometown)
	my_pic_im = plt.imread(my_pic)
	x_size, y_size, _ = my_pic_im.shape

	times_square_im_homo = np.copy(times_square_im)
	times_square_im_homo.setflags(write=1)
	for surface in times_square_surfaces:
		u = np.array(my_pic_surface).T
		v = np.array(surface).T
		H = homography_solve(u, v)
		U = np.array([[i, j] for i in range(x_size) for j in range(y_size)]).T
		V = homography_transform(U, H)

		for i in range(U.shape[1]):
			times_square_im_homo[V[0,i], V[1,i], :] = my_pic_im[U[0,i], U[1,i], :] 
		
	plt.imsave("times_square_homography.jpg", times_square_im_homo, format="jpg")
	plt.imshow(times_square_im_homo)
	plt.show()

	hometown_im_homo = np.copy(hometown_im)
	hometown_im_homo.setflags(write=1)
	for surface in hometown_surfaces:
		u = np.array(my_pic_surface).T
		v = np.array(surface).T
		H = homography_solve(u, v)
		U = np.array([[i, j] for i in range(x_size) for j in range(y_size)]).T
		V = homography_transform(U, H)

		for i in range(U.shape[1]):
			hometown_im_homo[V[0,i], V[1,i], :] = my_pic_im[U[0,i], U[1,i], :] 
		
	plt.imsave("Rach-Gia-City-worth-exploring_homography.jpg", hometown_im_homo, format="jpg")
	plt.imshow(hometown_im_homo)
	plt.show()

	times_square_im_affine = np.copy(times_square_im)
	times_square_im_affine.setflags(write=1)
	for surface in times_square_surfaces:
		u = np.array(my_pic_surface[:-1]).T
		v = np.array(surface[:-1]).T
		H = affine_solve(u, v)
		U = np.array([[i, j] for i in range(x_size) for j in range(y_size)]).T
		V = homography_transform(U, H)

		for i in range(U.shape[1]):
			times_square_im_affine[V[0,i], V[1,i], :] = my_pic_im[U[0,i], U[1,i], :] 
		
	plt.imsave("times_square_affine.jpg", times_square_im_affine, format="jpg")
	plt.imshow(times_square_im_affine)
	plt.show()

	hometown_im_affine = np.copy(hometown_im)
	hometown_im_affine.setflags(write=1)
	for surface in hometown_surfaces:
		u = np.array(my_pic_surface[:-1]).T
		v = np.array(surface[:-1]).T
		H = affine_solve(u, v)
		U = np.array([[i, j] for i in range(x_size) for j in range(y_size)]).T
		V = homography_transform(U, H)

		for i in range(U.shape[1]):
			hometown_im_affine[V[0,i], V[1,i], :] = my_pic_im[U[0,i], U[1,i], :] 
		
	plt.imsave("Rach-Gia-City-worth-exploring_affine.jpg", hometown_im_affine, format="jpg")
	plt.imshow(hometown_im_affine)
	plt.show()

	#### Rectify computer screen and flagellation ####

	computer_screen_im = plt.imread(computer_screen)
	flagellation_im = plt.imread(flagellation)

	x_size, y_size = 200, 300
	computer_screen_im_rectified_homo = np.ones([x_size, y_size, 3], "float")
	u = np.array(computer_screen_surface_rectified).T
	v = np.array(computer_screen_surface).T
	H = homography_solve(u, v)
	U = np.array([[i, j] for i in range(x_size) for j in range(y_size)]).T
	V = homography_transform(U, H)

	for i in range(U.shape[1]):
		computer_screen_im_rectified_homo[U[0,i], U[1,i], :] = computer_screen_im[V[0,i], V[1,i], :-1]
		
	plt.imsave("computer_screen_rectified_homography.jpg", computer_screen_im_rectified_homo, format="jpg")
	plt.imshow(computer_screen_im_rectified_homo)
	plt.show()

	computer_screen_im_rectified_affine = np.ones([x_size, y_size, 3], "float")
	u = np.array(computer_screen_surface_rectified[:-1]).T
	v = np.array(computer_screen_surface[:-1]).T
	H = affine_solve(u, v)
	U = np.array([[i, j] for i in range(x_size) for j in range(y_size)]).T
	V = homography_transform(U, H)

	for i in range(U.shape[1]):
		computer_screen_im_rectified_affine[U[0,i], U[1,i], :] = computer_screen_im[V[0,i], V[1,i], :-1]
		
	plt.imsave("computer_screen_rectified_affine.jpg", computer_screen_im_rectified_affine, format="jpg")
	plt.imshow(computer_screen_im_rectified_affine)
	plt.show()


	x_size, y_size = 600, 200
	flagellation_im_rectified_homo = np.ones([x_size, y_size, 3], "uint8") * 255
	u = np.array(flagellation_surface_rectified).T
	v = np.array(flagellation_surface).T
	H = homography_solve(u, v)
	U = np.array([[i, j] for i in range(x_size) for j in range(y_size)]).T
	V = homography_transform(U, H)

	for i in range(U.shape[1]):
		flagellation_im_rectified_homo[U[0,i], U[1,i], :] = flagellation_im[V[0,i], V[1,i], :]
		
	plt.imsave("flagellation_rectified_homography.jpg", flagellation_im_rectified_homo, format="jpg")
	plt.imshow(flagellation_im_rectified_homo)
	plt.show()

	x_size, y_size = 600, 200
	flagellation_im_rectified_affine = np.ones([x_size, y_size, 3], "uint8") * 255
	u = np.array(flagellation_surface_rectified[:-1]).T
	v = np.array(flagellation_surface[:-1]).T
	H = affine_solve(u, v)
	U = np.array([[i, j] for i in range(x_size) for j in range(y_size)]).T
	V = homography_transform(U, H)

	for i in range(U.shape[1]):
		flagellation_im_rectified_affine[U[0,i], U[1,i], :] = flagellation_im[V[0,i], V[1,i], :]
		
	plt.imsave("flagellation_rectified_affine.jpg", flagellation_im_rectified_affine, format="jpg")
	plt.imshow(flagellation_im_rectified_affine)
	plt.show()

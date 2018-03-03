import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def compute_R(phi, s):
	S = [0, -s[2], s[1],
		 s[2], 0, -s[0],
		 -s[1], s[0], 0]
	S = np.array(S).reshape(3, 3)
	I = np.identity(3)
	R = I + np.sin(phi) * S + (1 - np.cos(phi)) * S.dot(S)
	return R


def find_suv_coordinates(s, u, v, vector):
	SUV = np.stack((s, u, v))
	SUV = SUV.T
	return LA.inv(SUV).dot(vector)


def rot_to_ax_phi(R):
	trR = R[0, 0] + R[1, 1] + R[2, 2]
	phi = np.arccos(0.5 * (trR - 1))
	ax = 1 / (2 * np.sin(phi)) \
		* np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
	return phi, ax


if __name__ == "__main__":
	origin = np.array([0, 0, 0])
	s = np.array([0.6, 0.8, 0])
	u = np.array([-0.8, 0.6, 0])
	v = np.array([0, 0, 1])
	p = np.array([0, 1, 0])
	p_list = []
	phi_list = [0, np.pi/12, np.pi/8, np.pi/6,
				np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
	phi_arr = np.array(phi_list)

	print("s = " + str(s))
	print("u = " + str(u))
	print("v = " + str(v))
	print("Point: " + str(p))

	for phi in phi_arr:
		R = compute_R(phi, s)
		print("Phi = " + str(phi) + ":")
		print("The rotation matrix R:")
		print(R)
		p_new = R.dot(p)
		p_list.append(p_new)
		eigenvalues, eigenvectors = LA.eig(R)
		print("Eigenvalues:")
		print(eigenvalues)
		for vector in eigenvectors:
			print("Eigenvector: " + str(vector))
			x, y, z = find_suv_coordinates(s, u, v, vector)
			print("SUV representation: " + str(x) + "s + " + str(y) + "u + " + str(z) + "v")
		print()

	rotation_axis = np.stack((origin, s))
	rotated_points = np.stack(p_list)

	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.plot3D(rotation_axis[:,0], rotation_axis[:,1], rotation_axis[:,2])
	c = rotated_points[:,0] + rotated_points[:,1] + rotated_points[:,2]
	ax.plot3D(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2])
	ax.scatter(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2], c=c)
	for i in range(len(phi_list)):
		ax.text(rotated_points[i,0], rotated_points[i,1], rotated_points[i,2], str(i))
	plt.show()

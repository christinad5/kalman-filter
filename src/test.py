import numpy as np
from numpy import sin, cos
import quaternion as q
import numdifftools as nd

import rospy
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Quaternion
from scipy.spatial.transform import Rotation as R


def vec_to_quaternion(vec):
	"""Computes the quaternion format of a vector. Found on page 20, equation 3.26.
	vec is a 3x1 array.
	Returns a 4x1 array."""
	quat = np.concatenate( (np.array([[0.0]]), vec), axis=0 )
	return quat


def get_matrix_vector_product(u):
	"""Cmputes the matrix vector product. Found on page 18, equation 3.14.
	u is a 3X1 array.
	Returns a 3x3 array."""
	u1 = u[0][0]
	u2 = u[1][0]
	u3 = u[2][0]

	return np.array([[ 0.0, -u3, u2 ],
					[ u3, 0.0, -u1 ],
					[ -u2, u1, 0.0]])

def quat_left_multipl(p): 
	"""Computes the left quaternion multiplication. Found on page 20, equation 3.28.
	p is a 4x1 array.
	Returns a 4x4 array."""
	p0 = p[0]
	pv = p[1:]
	p_L = np.block([
					[ p0, -pv.transpose()],
					[pv, p0*np.eye(3) + get_matrix_vector_product(pv)]
					])
	return p_L


def quat_right_multipl(u):
	"""Cmputes the right quaternion multiplication. Found on page 20, equation 3.28.
	p is a 4x1 array.
	Returns a 4x4 array."""
	q0 = u[0]
	qv = u[1:]
	q_R = np.block([
					[ q0, -qv.transpose()],
					[qv, q0*np.eye(3) - get_matrix_vector_product(qv)]
					])
	return q_R


def exp_map(vec):
    """"Turns a vector into a quaternion using exponential mapping. Found on page 22, equation 3.36a.
	vec is a 3x1 array.
	Returns a 4x1 array."""
    vec_x = vec[0][0]
    vec_y = vec[1][0]
    vec_z = vec[2][0]
    norm  = np.linalg.norm(vec)
    exp_map = np.array([[cos(norm)],  
                        [(vec_x)*sin(norm)/norm],
                        [(vec_y)*sin(norm)/norm],
                        [(vec_z)*sin(norm)/norm]])
    return exp_map


def sigma_q_i(initial_q):
	"""Computes the covariance of the estimated quaternion for time t. Found on page 28, equation 3.69.
	initial_q is a 4x1 array with the quaternion for time t.
	Returns a 4x4 array."""
	little_sigma_eta_i = (20*np.pi)/180
	sigma_eta_i = np.square(little_sigma_eta_i)*np.eye(3)
	err_eta = np.random.multivariate_normal(np.array([0,0,0]), sigma_eta_i)
	x = err_eta[0]
	y = err_eta[1]
	z = err_eta[2]
	e_norm = np.linalg.norm(err_eta)

	err_jac = np.array([
		[-np.square(x)*sin(e_norm)/e_norm, -np.square(y)*sin(e_norm)/e_norm, -np.square(z)*sin(e_norm)/e_norm],
		[sin(e_norm)+(np.power(x,3)*cos(e_norm)/e_norm), x*np.square(y)*cos(e_norm)/e_norm, x*np.square(z)*cos(e_norm)/e_norm],
		[y*np.square(x)*cos(e_norm)/e_norm, sin(e_norm)+(np.power(y,3)*cos(e_norm)/e_norm), y*np.square(z)*cos(e_norm)/e_norm],
		[z*np.square(x)*cos(e_norm)/e_norm, z*np.square(y)*cos(e_norm)/e_norm, sin(e_norm)+(np.power(z,3)*cos(e_norm)/e_norm)]
	])

	sigma_q_i_mat = (1/4)*quat_right_multipl(initial_q)@err_jac@sigma_eta_i@err_jac.transpose()@quat_right_multipl(initial_q)
	return sigma_q_i_mat


def quat_propogate(initial_q, y_ang_vel, dt):
	"""Estimates the quaternion for time t+1. Found on page 40, equation 4.45a.
	initial_q is a 4x1 array with the quaternion for time t.
	y_ang_vel is a 3x1 array with the angular velocity measurements at time t.
	dt is a scalar number of the time step.
	Returns a 4x1 array."""
	ang_vel_quat = exp_map((dt*y_ang_vel)/2)
	# transform initial quaternion and exp_map of ang vel into quaternion types
	q_hat_nb_t = (q.from_float_array(initial_q.transpose()))*(q.from_float_array(ang_vel_quat.transpose()))
	# view quaternion multiplication as a 1x4 array
	q_hat_nb_t = q.as_float_array(q_hat_nb_t)
	# make 1x4 quaternion array 4x1
	q_hat_nb_t = np.array([[q_hat_nb_t[0][0]],
						[q_hat_nb_t[0][1]],
						[q_hat_nb_t[0][2]],
						[q_hat_nb_t[0][3]]])
	return q_hat_nb_t


def F_t1(y_ang_vel, dt):
	"""Computes the F Jacobian.`Found on page 40, under equation 4.45b.
	y_ang_vel is a 3x1 array with the angular velocity measurements at time t.
	dt is a scalar number of the time step.
	Returns a 4x4 array."""
	ang_vel_quat = exp_map((dt*y_ang_vel)/2)
	F_jacobian = quat_right_multipl(ang_vel_quat)
	return F_jacobian


def G_t1(initial_q, sigma_w, dt):
	"""Computes G jacobian. Found on page 40, under equation 4.45b.
	initial_q is a 4x1 array with the quaternion for time t.
	sigma_w is a 3x3 array with angular velocity covariance.
	dt is a scalar number for the time step.
	Returns a 4x3 array."""
	q_L = quat_left_multipl(initial_q)
	err_w = np.random.multivariate_normal(np.array([0,0,0]), sigma_w)

	x = err_w[0]
	y = err_w[1]
	z = err_w[2]
	e_norm = np.linalg.norm(err_w)

	err_jac = np.array([
		[-np.square(x)*sin(e_norm)/e_norm, -np.square(y)*sin(e_norm)/e_norm, -np.square(z)*sin(e_norm)/e_norm],
		[sin(e_norm)+(np.power(x,3)*cos(e_norm)/e_norm), x*np.square(y)*cos(e_norm)/e_norm, x*np.square(z)*cos(e_norm)/e_norm],
		[y*np.square(x)*cos(e_norm)/e_norm, sin(e_norm)+(np.power(y,3)*cos(e_norm)/e_norm), y*np.square(z)*cos(e_norm)/e_norm],
		[z*np.square(x)*cos(e_norm)/e_norm, z*np.square(y)*cos(e_norm)/e_norm, sin(e_norm)+(np.power(z,3)*cos(e_norm)/e_norm)]
	])

	G_jacobian = -(dt/2)*(q_L @ err_jac)
	return G_jacobian


def P_propogate(initial_P, F_jacobian, G_jacobian, sigma_w):
	"""Estimates the P matrix for time t+1. Found on page 40, equation 4.45b.
	initial_P is a 4x4 array the P matrix for time t. Can be computed using sigma_q_i function above.
	F_jacobian is a 4x4 array for the F jacobian matrix. Can be computed using F_t1 function above.
	G_jacobian is a 4x3 array for the G jacobian matrix. Can be computed using G_t1 function above.
	Q_mat is a 3x3 matrix for the covariance matrix of the angular velocity data.
	Returns a 4x4 array.
	"""
	# this function estimates the P matrix for the next time step
	Q_mat = sigma_w
	P_mat = F_jacobian @ initial_P @ F_jacobian.transpose() + G_jacobian @ Q_mat @ G_jacobian.transpose()
	return P_mat

def R_bn(initial_q):
	"""This function computes the rotation matrix transpose associated with our predicted quaternion (from frame n to b).
	quat_initial is a 4x1 array in (q0, q1, q2, q3) form with scalar first.
	Returns 3x3 array.
	"""
	q0 = initial_q[0][0]
	q1 = initial_q[1][0]
	q2 = initial_q[2][0]
	q3 = initial_q[3][0]
	R_nb = np.array([[2*np.square(q0)+2*np.square(q1)-1, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
				[2*q1*q2+2*q0*q3, 2*np.square(q0)+2*np.square(q2)-1, 2*q2*q3-2*q0*q1],
				[2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*np.square(q0)+2*np.square(q3)-1]])

	R_bn = R_nb.transpose()
	return R_bn

def H_t(initial_q):
	"""This function computes the H matrix.
	quat_initial is a 4x1 array in (q0, q1, q2, q3) form with scalar first.
	Returns a 6x4 array."""
	q0 = initial_q[0][0]
	q1 = initial_q[1][0]
	q2 = initial_q[2][0]
	q3 = initial_q[3][0]

	H_g = -1*np.array([
		[-2*q2, 2*q3, -2*q0, 2*q1],
		[2*q1, 2*q0, 2*q3, 2*q2],
		[4*q0, 0, 0, 4*q3]
	])

	H_m = np.array([
		[4*q0, 4*q1, 0, 0],
		[-2*q3, 2*q2, 2*q1, -2*q0],
		[2*q2, 2*q3, 2*q0, 2*q1]
	])

	H_mat = np.block([[H_g],[H_m]])

	return H_mat


def S_t(P_mat, H_mat, sigma_a, sigma_m):
	"""Computes the S matrix during the measurement update. Found on page 38, equation 4.38.
	P_mat is a 4x4 array computed using the P_propogate function above.
	H_mat is a 6x4 array. Obtained using H_t function. 
	sigma_a is a 3x3 array for the covariance matrix of the accelerometer data.
	sigma_m is a 3x3 array for the covariance matrix of the magnometer data.
	Returns 6x6 array."""
	R_mat = np.block([
					[sigma_a, np.zeros((3,3))],
					[np.zeros((3,3)), sigma_m]
					])
	S_mat = H_mat @ P_mat @ H_mat.transpose() + R_mat
	return S_mat


def K_t(P_mat, H_mat, S_mat):
	"""Computes the Kalman gain for the EKF. Found on page 38, equation 4.38.
	P_mat is a 4x4 array computed using the P_propogate function above.
	H_mat is a 6x4 array. Obtained with H_t function.
	S_mat is a 6x6 array. Obtained with S_t function.
	Returns 4x6 array.
	"""
	K_mat = P_mat @ H_mat.transpose() @ np.linalg.inv(S_mat)
	return K_mat


def error_t(y_acc, y_mag, R_bn_mat):
	"""Computes the error of estimated versus measured data. Found on page 38, equation 4.38.
	y_acc is a 3x1 array containing measured acceleration data.
	y_mag is a 3x1 array containing measured magnometer data. 
	R_bn_mat is a 3x3 array describing the rotation between from the 'n' to 'b' frame. Obtained using R_bn function.
	Returns 6x1 array."""
	g_n = np.array([[0], [0], [1]])
	m_n = np.array([[1], [0], [0]])

	y_t = np.block([
					[y_acc],
					[y_mag]
					])

	y_hat_t = np.block([
						[-R_bn_mat@g_n],
						[R_bn_mat@m_n]
						])
	error_mat = y_t - y_hat_t
	return error_mat


def q_measure(initial_q, K_mat, error_mat):
	"""Computes and renormalizes the quaternion measurement update. Found on page 40, equation 4.46a. 
	q_prop is a 4x1 array of the estimated quaternion. Obtained with quat_propogate function.
	K_mat is a 4x6 array of the Kalman gain. Obtained with K_t function.
	error_mat is a 6x1 of the error of estimated and measured data. Obtained with error_t function.
	Returns a 4x1 array."""
	q_update = initial_q + K_mat@error_mat # update quaternion measurement
	q_update = q_update/np.linalg.norm(q_update) # renormalize quaternion
	return q_update


def P_measure(P_prop, K_mat, S_mat, q_update):
	"""Computes and renormalizes the P matrix (covariance) measurement update.
	P_prop is a 4x4 array of the estimated P matrix. Obtained with P_propogate function.
	K_mat is a 4x6 array of the Kalman gain. Obtained with K_t function.
	S_mat is a 6x6 array of the S matrix. Obtained with S_t function."
	Returns a 4x4 array."""
	P_update = P_prop - K_mat @ S_mat @ K_mat.transpose() # update P matrix measurement
	J_mat = (1/np.power((np.linalg.norm(q_update)), 3))*(q_update @ q_update.transpose()) # calculate J matrix
	P_update = J_mat @ P_update @ J_mat.transpose() # renormalize P matrix
	return P_update


"""the variables below are the initial inputs to the EKF algorithm"""
vector_imu_acc_xyz = np.array([[-0.853], [-1.221], [-9.357]]) # acceleration data
vector_imu_ang_vel_xyz = np.array([[-0.000316], [-0.000629], [-0.001614]]) # angular velocity data quat
vector_imu_mag_xyz = np.array([[-0.1966], [0.0546], [0.3987]]) # magnometer data

# covariance for angular velocity. Found on page 23, equation 3.42. Must input values along diagonal.
sigma_w = np.array([
	[np.square(0.0049), 0, 0],
	[0, np.square(0.0049), 0],
	[0, 0, np.square(0.0049)]
]) 
# covariance for acceleration. Found on page 24, under equatioon 3.46. Must input values along diagonal.
sigma_a = np.array([
	[np.square(0.26), 0, 0],
	[0, np.square(0.26), 0],
	[0, 0, np.square(0.26)]
])
# covariance for magnometer. Found on page 25, under equation 3.52. Must input values along the diagonal. 
sigma_m = np.array([
	[np.square(0.25), 0, 0],
	[0, np.square(0.25), 0],
	[0, 0, np.square(0.25)]
])

# initiating quaternion data in (q0, q1, q3, q4) scalar first format
quat_initial = np.array([[-0.233556], [-0.016384], [0.006446], [0.972184]])
# initiating covariance matrix P
P_initial = sigma_q_i(quat_initial) 

def kalman_filter(quat_initial, P_initial, y_acc, y_ang_vel, y_mag, sigma_w, sigma_a, sigma_m):
	"""Step One initiate data"""
	dt = 0.1 # initiating time step

	"""Step Two (a)"""
	quat_estimate = quat_propogate(quat_initial, y_ang_vel, dt)
	F_mat = F_t1(y_ang_vel, dt)
	G_mat = G_t1(quat_initial, sigma_w, dt)
	P_estimate = P_propogate(P_initial, F_mat, G_mat, sigma_w)

	"""Step Two (b) and Three"""
	R_bn_mat = R_bn(quat_estimate)

	H_mat = H_t(quat_estimate)
	S_mat = S_t(P_estimate, H_mat, sigma_a, sigma_m)
	K_mat = K_t(P_estimate, H_mat, S_mat)
	
	err_t = error_t(y_acc, y_mag, R_bn_mat)

	quat_update = q_measure(quat_estimate, K_mat, err_t)
	P_update = P_measure(P_estimate, K_mat, S_mat, quat_update)

	return quat_update, P_update


def callback_acc(data):
	global vector_imu_acc_xyz
	vector_imu_acc_xyz[0,0] = data.linear_acceleration.x 
	vector_imu_acc_xyz[1,0] = data.linear_acceleration.y
	vector_imu_acc_xyz[2,0] = data.linear_acceleration.z


def callback_ang_vel(data):
    global vector_imu_ang_vel_xyz
    vector_imu_ang_vel_xyz[0,0] = data.angular_velocity.x
    vector_imu_ang_vel_xyz[1,0] = data.angular_velocity.y
    vector_imu_ang_vel_xyz[2,0] = data.angular_velocity.z


def callback_magnetic_field(data):
	global vector_imu_mag_xyz
	vector_imu_mag_xyz[0,0] = data.magnetic_field.x 
	vector_imu_mag_xyz[1,0] = data.magnetic_field.y
	vector_imu_mag_xyz[2,0] = data.magnetic_field.z 

def listener():
	global quat_initial
	global P_initial

	rospy.init_node('listener', anonymous=True)
	rospy.Subscriber("/sensor/Imu", Imu, callback_acc)
	rospy.Subscriber("/sensor/Magnetometer", MagneticField, callback_magnetic_field)
    
	pub_quaternion_transform = rospy.Publisher('/orientation', Quaternion, queue_size=10)
	pub_covariance_matrix = rospy.Publisher('/covariance', Float32MultiArray, queue_size = 10)
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		quat, covar = kalman_filter(quat_initial, P_initial, vector_imu_acc_xyz, vector_imu_ang_vel_xyz, vector_imu_mag_xyz, sigma_w, sigma_a, sigma_m)

		quat_initial = quat
		P_initial = covar

		quaternion = Quaternion()
		
		quaternion.w = quat_initial[0]
		quaternion.x = quat_initial[1]
		quaternion.y = quat_initial[2]
		quaternion.z = quat_initial[3]

		covariance_matrix = Float32MultiArray([3,3], P_initial)

		rospy.loginfo(quaternion)
		pub_quaternion_transform.publish(quaternion)
		pub_covariance_matrix.publish(covariance_matrix)
	rospy.spin()

def quat_conj(quat):
	"""Computes the conjugate of a quaternion. 
	quat is a 4x1 array of a quaternion (q0, q1, q2, q3) scalar first format.
	Returns a 4x1 array."""
	q0 = quat[0]
	qv = -quat[1:]
	quat = np.block([[q0], [qv]])
	return quat

def rot_mat_to_quat(R_mat):
	q0 = np.sqrt(1 + np.trace(R_mat))/2
	qv = (1/(4*q0) )* np.array([
								[R_mat[2][1]- R_mat[1][2]],
								[R_mat[0][2]- R_mat[2][0]],
								[R_mat[1][0] - R_mat[0][1]]
	])
	quat = np.block([[q0], [qv]])
	return quat

def R_nb(initial_q):
	"""This function computes the rotation matrix transpose associated with our predicted quaternion (from frame b to n).
	quat_initial is a 4x1 array in (q0, q1, q2, q3) form with scalar first.
	Returns 3x3 array.
	"""
	q0 = initial_q[0][0]
	q1 = initial_q[1][0]
	q2 = initial_q[2][0]
	q3 = initial_q[3][0]
	R_nb = np.array([[2*np.square(q0)+2*np.square(q1)-1, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
				[2*q1*q2+2*q0*q3, 2*np.square(q0)+2*np.square(q2)-1, 2*q2*q3-2*q0*q1],
				[2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*np.square(q0)+2*np.square(q3)-1]])
	return R_nb


def quat_mult(quat1, quat2):
	p0 = quat1[0]
	pv = quat1[1:]
	q0 = quat2[0]
	qv = quat2[1:]
	quat_mult = np.block([
		[p0*q0 - pv.transpose()@qv],
		[p0*qv + q0*pv + np.cross(pv, qv, axis = 0)]
	])
	return quat_mult

def func_two_returns():
	x = 1
	y = 2
	return x, y


if __name__ == '__main__':
	# generate random 1x4 array
	quat = np.random.randn(4)
	print(quat)
	quat_norm = np.linalg.norm(quat)     
	quat = quat/quat_norm
	print(quat)
	print(np.linalg.norm(quat))

	# quaternion in vertical list form for R_bn function
	quat_vert = [[quat[0]], [quat[1]], [quat[2]], [quat[3]]]
	print(quat_vert)

	# view array as quaternion type
	quat = q.as_quat_array(quat)
	# compute rotation matrix
	rot_mat = q.as_rotation_matrix(quat)
	print(rot_mat)


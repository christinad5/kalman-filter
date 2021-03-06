import unittest
import numdifftools as nd
import numpy as np
from kalman_filter import err_jac, R_bn, H_t
import quaternion as q

class TestKalmanFilter(unittest.TestCase):


    def test_jac(self):
    # test that err_jac function is working properly
        for trial in range(20):
            err_vec = np.random.randn(3)
            err_vec = np.ndarray.tolist(err_vec)
            fun = lambda e: np.r_[np.cos(np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2]))), 
                                    e[0]*np.sin(np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])))/np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])), 
                                    e[1]*np.sin(np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])))/np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])), 
                                    e[2]*np.sin(np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])))/np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2]))]
            jac = nd.Jacobian(fun)(err_vec)

            for i in range(4):
                for j in range(3):
                    self.assertAlmostEqual(err_jac(err_vec)[i][j], jac[i][j])

    def test_rotation_mat(self):
        for trial in range(20):
            # generate random 1x4 array
            quat = np.random.randn(4)
            quat_norm = np.linalg.norm(quat)     
            quat = quat/quat_norm

            # quaternion in vertical list form for R_bn function
            quat_vert = [[quat[0]], [quat[1]], [quat[2]], [quat[3]]]

            # view array as quaternion type
            quat = q.as_quat_array(quat)
            # compute rotation matrix
            rot_mat = q.as_rotation_matrix(quat)

            # check that matrices match
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(R_bn(quat_vert)[i][j], rot_mat.transpose()[i][j])

    def test_hessian(self):
        for trial in range(20):
            # generate random 1x4 array
            quat = np.random.randn(4)
            quat_norm = np.linalg.norm(quat)     
            quat = quat/quat_norm

            # quaternion in vertical list form for R_bn function
            quat_vert = [[quat[0]], [quat[1]], [quat[2]], [quat[3]]]

            fun_g = lambda q: np.r_[(-2*q[0]*q[2]+2*q[1]*q[3]),
                                    (2*q[0]*q[1]+2*q[2]*q[3]),
                                    (-1+2*np.square(q[0])+2*np.square(q[3]))] 
            
            hessian_g = -1*nd.Jacobian(fun_g)(quat)
            fun_m = lambda q: np.r_[(-1+2*np.square(q[0])+2*np.square(q[1])),
                                    (2*q[1]*q[2]-2*q[0]*q[3]),
                                    (2*q[0]*q[2]+2*q[1]*q[3])] 
            hessian_m = nd.Jacobian(fun_m)(quat)
            hessian = np.block([[hessian_g],[hessian_m]])
            for i in range(6):
                for j in range(4):
                    self.assertAlmostEqual(H_t(quat_vert)[i][j], hessian[i][j])



if __name__ == '__main__':
    unittest.main()

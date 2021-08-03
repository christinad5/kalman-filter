import unittest
import numdifftools as nd
import numpy as np
from kalman_filter import err_jac

class TestJacobian(unittest.TestCase):

    def test_jac_num(self, err_vec = [-0.00557302, 0.0036593, 0.00189644]):
        
        # test that err_jac function is working properly
        fun = lambda e: np.r_[np.cos(np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2]))), 
                                e[0]*np.sin(np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])))/np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])), 
                                e[1]*np.sin(np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])))/np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])), 
                                e[2]*np.sin(np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2])))/np.sqrt(np.square(e[0])+np.square(e[1])+np.square(e[2]))]
        jac = nd.Jacobian(fun)(err_vec)

        for i in range(4):
            for j in range(3):
                self.assertAlmostEqual(err_jac(err_vec)[i][j], jac[i][j])


if __name__ == '__main__':
    unittest.main()

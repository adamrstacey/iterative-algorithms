import unittest
import numpy as np
import iterative_algs as ia

class TestIterativeAlgorithms(unittest.TestCase):

    def test_arnoldi(self):
        """ Tests the arnoldi iteration to make sure the Krylov space is generated correctly """
        # Form matrix and generate orthogonal basis for Krylov space
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        # Krylov space
        K = [b]
        for k in range(1, 5):
            K.append(np.matmul(A, K[-1]))
        # Q is orthogonal basis of Krylov space
        Q = np.linalg.qr(np.concatenate(K, 1))[0]
        
        arnoldi = ia.Arnoldi(A, b)
        arnoldi.iterate(4)
        Q_hat = arnoldi.Q

        # RMSE in Q_hat
        rmse = np.sqrt(np.mean((np.abs(Q) - np.abs(Q_hat))**2))

        self.assertLessEqual(rmse, 1e-5)

    def test_gmres(self):
        """ Tests GMRES solver """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        x = np.linalg.solve(A, b)
        x_hat = ia.gmres(A, b)
        rmse = np.sqrt(np.mean((x - x_hat)**2))
        self.assertLessEqual(rmse, 1e-5)
        #self.assertLessEqual(rmse, rmse)


if __name__ == "__main__":
    unittest.main()
        


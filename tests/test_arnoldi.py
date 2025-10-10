import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
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
        Q_hat = arnoldi.form_Q()
        
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
        self.assertLessEqual(rmse, 1e-8)

    def test_arnoldi_eigs(self):
        """ Tests computing eigenvalues and eigenvectors using the Arnoldi iteration """
        num_eigs = 1
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        arnoldi = ia.Arnoldi(A, b)
        arnoldi.iterate(5)
        D_hat = arnoldi.get_eigs(num_eigs)[0]
        D, V = np.linalg.eig(A)
        idx = np.argmin(np.abs(D - D_hat))
        self.assertLessEqual(np.max(np.abs(D[idx] - D_hat)), 1e-14)
    
    def test_too_many_iterations(self):
        """ Tests that the correct number of basis vectors are produced """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        arnoldi = ia.Arnoldi(A, b)
        arnoldi.iterate(10)
        Q = arnoldi.form_Q()
        H = arnoldi.form_H()
        self.assertEqual(Q.shape[1], A.shape[1])
        self.assertEqual(H.shape[0], A.shape[1] + 1)
        self.assertEqual(H.shape[1], A.shape[1])

if __name__ == "__main__":
    unittest.main(verbosity=2)
        


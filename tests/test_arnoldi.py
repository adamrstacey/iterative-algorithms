import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
import iterative_algs as ia

class TestIterativeAlgorithms(unittest.TestCase):
    
    # /////////////////////////////////////////////
    # ARNOLDI ITERATION TESTS
    # /////////////////////////////////////////////
    
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

    def test_rectangular_matrix(self):
        """ Ensures proper response from supplying rectangular matrix """
        A = np.random.randn(5, 4)
        b = np.random.randn(5, 1)
        with self.assertRaises(ValueError):
            arnoldi = ia.Arnoldi(A, b)

    def test_dim_mismatch(self):
        """ Ensures proper response when A and b have different number of rows/cols"""
        A = np.random.randn(5, 5)
        b = np.random.randn(3, 1)
        with self.assertRaises(ValueError):
            arnoldi = ia.Arnoldi(A, b)

    def test_1d_RHS(self):
        """ Ensures that 1D RHS is handled correctly """
        A = np.random.randn(5, 5)
        b = np.random.randn(5)
        arnoldi = ia.Arnoldi(A, b)
        self.assertEqual(arnoldi.b.ndim, 2)

    def test_zero_iterations(self):
        """ Tests that nothing happens when 0 iterations are called """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        arnoldi = ia.Arnoldi(A, b)
        arnoldi.iterate(0)
        Q = arnoldi.form_Q()
        self.assertEqual(Q.shape[1], 1)

    def test_negative_iterations(self):
        """ Tests that nothing happens when negative iterations are called """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        arnoldi = ia.Arnoldi(A, b)
        arnoldi.iterate(-2)
        Q = arnoldi.form_Q()
        self.assertEqual(Q.shape[1], 1)

    def test_zero_or_negative_eigenvalues(self):
        """ Tests that no eigenvalues are returned when zero or negative are requested """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        arnoldi = ia.Arnoldi(A, b)
        arnoldi.iterate(5)
        D = arnoldi.get_eigs(0)
        self.assertEqual(D, None)
        D = arnoldi.get_eigs(-1)
        self.assertEqual(D, None)

    def test_request_too_many_eigenvalues(self):
        """ Test that the appropriate number of eigenvalues are returned """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        arnoldi = ia.Arnoldi(A, b)
        arnoldi.iterate(2)
        H = arnoldi.form_H()
        D = arnoldi.get_eigs(5)[0]
        self.assertEqual(H.shape[1], D.shape[0])
        arnoldi.iterate(3)
        D = arnoldi.get_eigs(10)[0]
        self.assertEqual(A.shape[0], D.shape[0])

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

    # /////////////////////////////////////////////
    # GMRES TESTS
    # /////////////////////////////////////////////

    def test_gmres_solve(self):
        """ Tests GMRES solver """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        x = np.linalg.solve(A, b)
        x_hat = ia.gmres(A, b)
        rmse = np.sqrt(np.mean((x - x_hat)**2))
        self.assertLessEqual(rmse, 1e-8)

    def test_1D_array(self):
        """ Ensure that 1D RHS is handlded correctly """
        A = np.random.randn(5, 5)
        b = np.random.randn(5)
        x_hat = ia.gmres(A, b)[:, 0]
        x = np.linalg.solve(A, b)
        rmse = np.sqrt(np.mean((x - x_hat)**2))
        self.assertLessEqual(rmse, 1e-8)

    def test_zero_or_negative_max_iters(self):
        """ Ensures that None is returned when max_iters set <= 0 """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        x_hat = ia.gmres(A, b, max_iters=0)
        self.assertEqual(x_hat, None)
        x_hat = ia.gmres(A, b, max_iters=-2)
        self.assertEqual(x_hat, None)

if __name__ == "__main__":
    unittest.main(verbosity=2)
        


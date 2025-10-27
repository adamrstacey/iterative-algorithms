import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
import iterative_algs as ia

class TestLanczos(unittest.TestCase):

    # /////////////////////////////////////////////
    # Lanczos ITERATION TESTS
    # /////////////////////////////////////////////
    
    def test_lanczos(self):
        """ Test generating Krylov space with Lanczos iteration """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A) # Make A symmetric
        b = np.random.randn(5, 1) 
        
        # Form Krylov Space with Lanczos Iteration
        lanczos = ia.Lanczos(A, b)
        lanczos.iterate(4)
        Q_hat = lanczos.form_Q()

        # Form Krylov Space Manually
        K = [b/np.linalg.norm(b)]
        for j in range(1, A.shape[1]):
            K.append(np.matmul(A, K[-1]))
        Q = np.linalg.qr(np.concatenate(K, 1))[0]

        # RMSE in Q_hat
        rmse = np.sqrt(np.mean((np.abs(Q) - np.abs(Q_hat))**2))
        self.assertLessEqual(rmse, 1e-8)

    def test_non_symmetric(self):
        """ Test that correct exception is raised when A is not symmetric """
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        with self.assertRaises(ValueError):
            lanczos = ia.Lanczos(A, b)

    def test_low_rank(self):
        """ Tests that Krylov space is generated correctly for low-rank matrix """
        A = np.random.randn(5, 1)
        A = np.matmul(A, A.T) # Make A symmetric
        b = np.random.randn(5, 1)
        lanczos = ia.Lanczos(A, b)
        lanczos.iterate(5)
        Q = lanczos.form_Q()
        rmse = np.sqrt(np.mean((np.matmul(Q.T, Q) - np.identity(Q.shape[1]))**2))
        self.assertLessEqual(rmse, 1e-14)

    def test_rectangular_matrix(self):
        """ Ensures proper response from supplying rectangular matrix """
        A = np.random.randn(5, 4)
        b = np.random.randn(5, 1)
        with self.assertRaises(ValueError):
            lanczos = ia.Lanczos(A, b)

    def test_dim_mismatch(self):
        """ Ensures proper response when A and b have different number of rows/cols"""
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(3, 1)
        with self.assertRaises(ValueError):
            arnoldi = ia.Lanczos(A, b)

    def test_1d_RHS(self):
        """ Ensures that 1D RHS is handled correctly """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5)
        lanczos = ia.Lanczos(A, b)
        self.assertEqual(lanczos.b.ndim, 2)

    def test_zero_iterations(self):
        """ Tests that nothing happens when 0 iterations are called """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)
        lanczos = ia.Lanczos(A, b)
        lanczos.iterate(0)
        Q = lanczos.form_Q()
        self.assertEqual(Q.shape[1], 1)

    def test_negative_iterations(self):
        """ Tests that nothing happens when negative iterations are called """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)
        lanczos = ia.Lanczos(A, b)
        lanczos.iterate(-2)
        Q = lanczos.form_Q()
        self.assertEqual(Q.shape[1], 1)

    def test_too_many_iterations(self):
        """ Tests that the correct number of basis vectors are produced """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)
        lanczos = ia.Lanczos(A, b)
        lanczos.iterate(10)
        Q = lanczos.form_Q()
        H = lanczos.form_H()
        self.assertEqual(Q.shape[1], A.shape[1])
        self.assertEqual(H.shape[0], A.shape[1] + 1)
        self.assertEqual(H.shape[1], A.shape[1])
    
    def test_form_H_one_iter(self):
        """ Tests forming H after only 1 iteration """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)
        lanczos = ia.Lanczos(A, b)
        lanczos.iterate()
        H = lanczos.form_H()
        self.assertEqual(H.shape[0], 2)
        self.assertEqual(H.shape[1], 1)

    def test_lanczos_eigs(self):
        """ Tests computing eigenvalues using Lanczos iteration """
        num_eigs = 1
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)
        
        # Compute Eigenvalues Directly
        D, V = np.linalg.eigh(A)
        D = D[::-1][:num_eigs]
        V = V[:, ::-1][:, :num_eigs]

        lanczos = ia.Lanczos(A, b)
        lanczos.iterate(5)
        D_hat, V_hat = lanczos.get_eigs(num_eigs)
        
        self.assertEqual(D_hat.shape[0], D.shape[0])
        self.assertEqual(V_hat.shape[1], V.shape[1])
        self.assertEqual(V_hat.shape[0], V.shape[0])
        
        rmse = np.sqrt(np.mean((D - D_hat)**2))
        self.assertLessEqual(rmse, 1e-8)
    
    def test_zero_or_negative_eigenvalues(self):
        """ Tests that no eigenvalues are returned when zero or negative are requested """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)
        lanczos = ia.Lanczos(A, b)
        lanczos.iterate(5)
        num_eigs = 0
        D = lanczos.get_eigs(num_eigs)
        self.assertEqual(D, None)
        num_eigs = -2
        D = lanczos.get_eigs(num_eigs)
        self.assertEqual(D, None)

    def test_too_many_eigenvalues(self):
        """ Tests that appropriate number of eigs are returned when too many are requested """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)
        lanczos = ia.Lanczos(A, b)
        num_iters = 3
        num_eigs = 10
        lanczos.iterate(num_iters)
        D = lanczos.get_eigs(num_eigs)[0]
        self.assertEqual(D.shape[0], num_iters)
        lanczos.iterate(10)
        D = lanczos.get_eigs(num_eigs)[0]
        self.assertEqual(D.shape[0], A.shape[0])

    # /////////////////////////////////////////////
    # CONJUGATE GRADIENT TESTS
    # /////////////////////////////////////////////
    
    def test_conjugate_gradient(self):
        """ Tests solving Ax = b when A is s.p.d """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)

        # True Solution
        x = np.linalg.solve(A, b)

        # Approx Solution
        x_hat = ia.conjugate_gradient(A, b)

        # Error
        rmse = np.sqrt(np.mean((x - x_hat)**2))

        self.assertLessEqual(rmse, 1e-8)

    def test_1D_array(self):
        """ Tests that CG solver handles 1D RHS array """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)

        # True Solution
        x = np.linalg.solve(A, b)
        
        # Remove Unitary Dims
        b = b.squeeze()
        x_hat = ia.conjugate_gradient(A, b)

        # Error
        rmse = np.sqrt(np.mean((x - x_hat)**2))

        self.assertLessEqual(rmse, 1e-8)

    def test_zero_or_negative_iters(self):
        """ Tests that nothing is returned when max_iters <= 0 """
        A = np.random.randn(5, 5)
        A = np.matmul(A.T, A)
        b = np.random.randn(5, 1)

        x_hat = ia.conjugate_gradient(A, b, max_iters=0)
        self.assertEqual(x_hat, None)
        x_hat = ia.conjugate_gradient(A, b, max_iters=-2)
        self.assertEqual(x_hat, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)



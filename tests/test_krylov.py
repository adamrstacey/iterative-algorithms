import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
import iterative_algs as ia

class TestKrylov(unittest.TestCase):
    """ Class for testing Krylov class """

    def test_symmetric(self):
        """ Tests that Krylov class correctly identifies symmetric matrices """
        A = np.random.randn(5, 5)
        krylov = ia.Krylov(A, A[:, 0])
        self.assertFalse(krylov.symmetric)
        A = np.matmul(A.T, A)
        krylov = ia.Krylov(A, A[:, 0])
        self.assertTrue(krylov.symmetric)
    
    def test_positive_definite(self):
        """ Tests that Krylov class correctly identifes positive definite matrices """
        A = 1.0 * np.diag(np.array([-1, 2, 3, 4, 5]))
        b = np.random.randn(5, 1)
        krylov = ia.Krylov(A, b)
        self.assertFalse(krylov.positive_definite)
        A = 1.0 * np.diag(np.array([1, 2, 3, 4, 5]))
        krylov = ia.Krylov(A, b)
        self.assertTrue(krylov.positive_definite)

    def test_form_q(self):
        """ Tests that Krylov class forms orthogonal basis for Krylov subspace """
        # Non symmetric case
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        
        krylov = ia.Krylov(A, b)
        self.assertFalse(krylov.symmetric)
        krylov.iterate(5)
        Q_hat = krylov.form_Q()
        self.assertEqual(Q_hat.shape[1], A.shape[1])
        # Krylov space
        K = [b]
        for k in range(1, 5):
            K.append(np.matmul(A, K[-1]))
        # Q is orthogonal basis of Krylov space
        Q = np.linalg.qr(np.concatenate(K, 1))[0]
        rmse = np.sqrt(np.mean((np.abs(Q) - np.abs(Q_hat))**2))
        self.assertLessEqual(rmse, 1e-5); del Q, K, Q_hat, krylov

        # Symmetric case
        A = np.matmul(A.T, A)
        krylov = ia.Krylov(A, b)
        self.assertTrue(krylov.symmetric)
        krylov.iterate(5)
        Q_hat = krylov.form_Q()
        self.assertEqual(Q_hat.shape[1], A.shape[1])
         # Krylov space
        K = [b]
        for k in range(1, 5):
            K.append(np.matmul(A, K[-1]))
        # Q is orthogonal basis of Krylov space
        Q = np.linalg.qr(np.concatenate(K, 1))[0]
        rmse = np.sqrt(np.mean((np.abs(Q) - np.abs(Q_hat))**2))
        self.assertLessEqual(rmse, 1e-5)

    def test_form_h(self):
        """ Tests that Krylov class forms H properly """
        # Non symmetric case
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        krylov = ia.Krylov(A, b)
        self.assertFalse(krylov.symmetric)
        krylov.iterate(8)
        H = krylov.form_H()
        self.assertEqual(H.shape[0], A.shape[0] + 1)
        self.assertEqual(H.shape[1], A.shape[1])
        
        # Symmetric case
        A = np.matmul(A.T, A)
        krylov = ia.Krylov(A, b)
        self.assertTrue(krylov.symmetric) 
        krylov.iterate(7)
        H = krylov.form_H()
        self.assertEqual(H.shape[0], A.shape[0] + 1)
        self.assertEqual(H.shape[1], A.shape[1])

    def test_solve(self):
        """ Tests that Krylov class correctly solves Ax = b """

        # Non-symmetric case
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        x = np.linalg.solve(A, b)
        krylov = ia.Krylov(A, b)
        self.assertFalse(krylov.symmetric)
        x_hat = krylov.solve()
        rmse = np.sqrt(np.mean((x - x_hat)**2))
        self.assertLessEqual(rmse, 1e-8)

        # Symmetic, non-positive definite case
        A = -1.0 * np.diag(np.array([1, 2, 3, 4, 5]))
        x = np.linalg.solve(A, b)
        krylov = ia.Krylov(A, b)
        self.assertTrue(krylov.symmetric)
        self.assertFalse(krylov.positive_definite)
        x_hat = krylov.solve()
        rmse = np.sqrt(np.mean((x - x_hat)**2))
        self.assertLessEqual(rmse, 1e-8)

        # SPD case
        A *= -1.0
        krylov = ia.Krylov(A, b)
        self.assertTrue(krylov.symmetric)
        self.assertTrue(krylov.positive_definite)
        x = np.linalg.solve(A, b)
        x_hat = krylov.solve()
        rmse = np.sqrt(np.mean((x - x_hat)**2))
        self.assertLessEqual(rmse, 1e-8)

    def test_get_eigs(self):
        """ Tests that Krylov class correctly returns eigenvalues of A """
        # Non symmetric case
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 1)
        krylov = ia.Krylov(A, b)
        self.assertFalse(krylov.symmetric)
        krylov.iterate(3)
        self.assertEqual(None, krylov.get_eigs(-1))
        D, V = krylov.get_eigs(3)
        self.assertEqual(3, D.shape[0])
        self.assertEqual(3, V.shape[1])
        self.assertEqual(A.shape[0], V.shape[0])

        # Symmetric case
        A = np.matmul(A.T, A)
        krylov = ia.Krylov(A, b)
        self.assertTrue(krylov.symmetric)
        krylov.iterate(3)
        self.assertEqual(None, krylov.get_eigs(-1))
        D, V = krylov.get_eigs(3)
        self.assertEqual(3, D.shape[0])
        self.assertEqual(3, V.shape[1])
        self.assertEqual(A.shape[0], V.shape[0])

if __name__ == "__main__":
    unittest.main(verbosity=2)

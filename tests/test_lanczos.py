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
        lanczos = ia.Lanczos(A, b)
        self.assertEqual(lanczos.A.shape[0], A.shape[0])

if __name__ == "__main__":
    unittest.main(verbosity=2)



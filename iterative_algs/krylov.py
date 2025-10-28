import numpy as np
from .arnoldi import Arnoldi
from .lanczos import Lanczos
from .gmres import gmres
from .conjugate_gradient import conjugate_gradient

class Krylov:
    """ Class for generating Krylov Subspaces and solving Ax=b or Av = av """

    def __init__(self, A, b):
        """ 
        Creates an instance of the Krylov Class 
        Args:
            A: m x m matrix stored as ndarray
            b: m x 1 array stored as ndarray
        """
        self.A = A
        self.b = b
        self.m = self.A.shape[0]

        # Test if A is Square
        if A.shape[1] != self.m:
            raise ValueError("A must be a squre matrix")

        # Test if A is SPD
        self.symmetric = self.test_symmetric()
        if self.symmetric is True:
            self.positive_definite = self.test_positive_definite()
        else:
            self.positive_definite = False

        # Initialize Class
        if self.symmetric:
            self.iteration = Lanczos(A, b)
        else:
            self.iteration = Arnoldi(A, b)
        
    def iterate(self, k=1):
        """ Performs iteration of Arnoldi/Lanczos Iteration """
        self.iteration.iterate(k)
    
    def form_Q(self):
        """ Forms basis of Krylov subspace """
        return self.iteration.form_Q()

    def form_H(self):
        """ Returns H from iteration """
        return self.iteration.form_H()

    def solve(self):
        """ Sovle Ax = b using prefered method """
        if self.symmetric and self.positive_definite:
            return conjugate_gradient(self.A, self.b)
        else:
            return gmres(self.A, self.b)

    def test_symmetric(self):
        """ Tests whether A is symmetric """
        return np.allclose(self.A, self.A.T)
            
    def test_positive_definite(self):
        """ Tests whether A is positive definite """
        try:
            np.linalg.cholesky(self.A)
            return True
        except np.linalg.LinAlgError:
            return False


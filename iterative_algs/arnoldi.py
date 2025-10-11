import sys
import numpy as np

class Arnoldi:
    """ Class for implementing the Arnoldi Iteration """
    def __init__(self, A, b):
        """ 
        Initializes an instance of the Arnoldi iteration class 
        Args:
            A: square m x m matrix, saved as a numpy ndarray
            b: m x 1 vector, saved as a numpy ndarray
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError("Arnoldi iteration requires a square matrix.")
        if A.shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch between LHS and RHS.")
        if b.ndim == 1:
            b = np.expand_dims(b, 1)

        self.A = A
        self.b = b
        self.q_list = [self.b/np.linalg.norm(b)]
        self.m = A.shape[0]
        self.Q = self.form_Q()
        self.H = []

    def iterate(self, k=1):
        """ Performs k iterations of the Arnoldi iteration """
        
        # Handle zero- or negative-iteration case
        if k <= 0:
            return

        # Handle case when space is already complete
        if len(self.q_list) > self.m:
            return
        
        # Iterate and compute vectors
        for j in range(k):
            v = np.matmul(self.A, self.q_list[-1])
            self.H.append([])
            for q in self.q_list:
                self.H[-1].append(np.dot(q.squeeze(), v.squeeze()))
                v -= (self.H[-1][-1] * q)
            self.H[-1].append(np.linalg.norm(v))
            self.q_list.append(v/self.H[-1][-1])
            if len(self.q_list) > self.m:
                return

        return 

    def form_Q(self):
        """ Concatenates list of orthogonal vectors into an ndarray """
        Q = np.concatenate(self.q_list, 1)
        return Q[:, :self.m]
    
    def form_H(self):
        """ Forms (n + 1) x n Hessenberg matrix after n applications of iteration """
        n = len(self.H)
        H = np.zeros((n + 1, n), dtype=self.b.dtype)
        for j in range(n):
            for i in range(len(self.H[j])):
                H[i, j] = self.H[j][i]
        return H[:self.m + 1, :self.m]

    def get_eigs(self, n=1):
        """ 
        Approximates k eigenvalues and eigenvectors after k iterations 
        Args:
            n: number of eigenvalues/eigenvectors to be returned.
                Note: n < k, the number of completed iterations.
        Returns:
            D: array containing eigenvalue estimates
            V: array containing eigenvector estimates
        """

        # Change number of eigenvalues, if necessary
        if n <= 0:
            return
        if n > self.A.shape[0]:
            n = self.A.shape[0]
        if n > len(self.H):
            n = len(self.H)
        
        # Get Hessenberg matrix
        H = self.form_H()[:-1, :]

        # Solve for eigenvalues/eigenvectors and sort
        D, V = np.linalg.eig(H)
        idx = np.argsort(np.abs(D))[::-1]
        D = D[idx]
        V = V[:, idx]
        return D[:n], V[:, :n]




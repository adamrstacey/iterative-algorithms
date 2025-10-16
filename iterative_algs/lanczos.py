import numpy as np

class Lanczos:
    """ Class for Implementing the Lanczos Iteration """
    def __init__(self, A, b):
        """ 
        Initializes instance of Lanczos class
        Args:
            A: square, symmetric m x m matrix, saved as ndarray
            b: m x 1 vector, saved as ndarray
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError("Lanczos iteration requires a square matrix.")
        if np.allclose(A, A.T) is False:
            raise ValueError("Lanczos iteration requires a symmetric matrix, consider using the Arnoldi iteration")
        if A.shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch between LHS and RHS.")
        if b.ndim == 1:
            b = np.expand_dims(b, 1)

        self.A = A
        self.b = b
        self.q_list = [self.b/np.linalg.norm(b)]
        self.m = A.shape[0]
        self.alpha = []
        self.beta = []

    def iterate(self, k=1):
        """ Performs k iterations of the Lanczos iteration """

        # Handle zero or negative iteration case
        if k <= 0:
            return
        
        # Handle case when space is already complete
        if len(self.q_list) > self.m:
            return

        # Iterate and compute
        for j in range(k):
            v = np.matmul(self.A, self.q_list[-1])
            self.alpha.append(np.dot(v.squeeze(), self.q_list[-1].squeeze()))
            if len(self.beta) > 0:
                v -= ((self.beta[-1] * self.q_list[-2]) + (self.alpha[-1] * self.q_list[-1]))
            else:
                v -= (self.alpha[-1] * self.q_list[-1])
            self.beta.append(np.linalg.norm(v))
            self.q_list.append(v/self.beta[-1])
            if len(self.q_list) > self.m:
                return
        
        return
    
    def form_Q(self):
        """ Concatenates list of orthogonal vectors into an ndarray """
        Q = np.concatenate(self.q_list, 1)
        return Q[:, :self.m]

    def form_H(self):
        """ Returns (n + 1) x n Hessenberg matrix after n iters """
        n = len(self.alpha)
        H = np.zeros((n, n - 1), dtype=self.b.dtype)
        H[0, :2] = np.array([self.alpha[0], self.beta[0]])
        H[-1, -1] = self.beta[-1]
        for j in range(1, n - 1):
            H[j, j - 1: j + 2] = np.array([self.beta[j - 1], self.alpha[j], self.beta[j]])
        return H

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

        if n > self.m or n > len(self.alpha):
            n = min(self.m, len(self.alpha))

        # Get Hessenberg Matrix and Q
        H = self.form_H()[:-1, :]
        Q = self.form_Q()[:, :H.shape[0]]

        D, V = np.linalg.eigh(H) # H is symmetric in this case
        V = np.matmul(Q, V); del Q
        D = D[::-1]
        V = V[:, ::-1]
        return D[:n], V[:, :n]
        




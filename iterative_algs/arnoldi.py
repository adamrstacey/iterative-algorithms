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
        self.A = A
        self.b = b
        self.q_list = [self.b/np.linalg.norm(b)]
        self.Q = self.form_Q()
        self.H = [] # List of Lists, each sublist is a column of H

    def iterate(self, k=1):
        """ Performs k iterations of the Arnoldi iteration """
        for j in range(k):
            v = np.matmul(self.A, self.q_list[-1])
            self.H.append([])
            for q in self.q_list:
                self.H[-1].append(np.dot(q.squeeze(), v.squeeze()))
                v -= (self.H[-1][-1] * q)
            self.H[-1].append(np.linalg.norm(v))
            self.q_list.append(v/self.H[-1][-1])
        self.Q = self.form_Q()
    
    def form_Q(self):
        """ Concatenates list of orthogonal vectors into an ndarray """
        return np.concatenate(self.q_list, 1)
    
    def form_H(self):
        """ Forms (n + 1) x n Hessenberg matrix after n applications of iteration """
        n = len(self.H)
        H = np.zeros((n + 1, n), dtype=self.b.dtype)
        for j in range(n):
            for i in range(len(self.H[j])):
                H[i, j] = self.H[j][i]
        return H

def arnoldi_iteration(A, b, k):
    """
    Perform Arnoldi iteration to generate an orthonormal basis of the Krylov subspace.

    Parameters:
    A : ndarray
        Square matrix of size (n, n)
    b : ndarray
        Initial vector of size (n,)
    k : int
        Number of iterations (dimension of Krylov subspace)

    Returns:
    Q : ndarray
        Orthonormal basis of Krylov subspace (n, k+1)
    H : ndarray
        Upper Hessenberg matrix (k+1, k)
    """
    n = A.shape[0]
    Q = np.zeros((n, k+1))
    H = np.zeros((k+1, k))

    # Normalize the initial vector
    Q[:, 0] = b / np.linalg.norm(b)

    for j in range(k):
        v = A @ Q[:, j]
        for i in range(j+1):
            H[i, j] = np.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]
        H[j+1, j] = np.linalg.norm(v)
        if H[j+1, j] != 0:
            Q[:, j+1] = v / H[j+1, j]
        else:
            # Breakdown: Krylov subspace has been exhausted
            print(f"Breakdown at iteration {j}")
            return Q[:, :j+1], H[:j+2, :j+1]

    return Q, H


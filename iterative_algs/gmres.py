import numpy as np
from arnoldi import Arnoldi

def gmres(A, b, x0=None, tol=1e-5, max_iters=None):
    """ 
    GMRES solver to solve Ax = b
    Args:
        A: square m x m matrix, stored as ndarray
        b: m x 1 vector, stores as ndarray
        x0: m x 1 vector initial guess
        tol: tolerance 
        max_iters: maximum number of iterations
    Returns:
        x: Approximate solution to Ax = b
    """
    
    # Store all vectors as m x 1 arrays
    if b.ndim == 1:
        b = np.expand_dims(b, 1)

    # Dimension Info
    m = A.shape[0]
    if x0 is None:
        x0 = np.zeros((m, 1), dtype=A.dtype)

    # Define Residual
    norm_b = np.linalg.norm(b)
    r = b - np.matmul(A, x0)
    beta = np.linalg.norm(r)
    x = x0.copy(); del x0
    
    # Set Maximum Number of Iterations
    if max_iters is None:
        max_iters = m
    elif max_iters <= 0:
        return

    # Initialize Arnoldi Iteration Object
    arnoldi = Arnoldi(A, b)
    
    # Iterate
    for k in range(max_iters):
        if beta < tol:
            return x
        
        # Arnoldi Iteration
        arnoldi.iterate()
        Q = arnoldi.form_Q()
        H = arnoldi.form_H()

        # Right Hand Side
        RHS = np.zeros((H.shape[0], 1))
        RHS[0] = norm_b
        
        # Solve Least Squares Problem and Update Solution
        y = np.linalg.lstsq(H, RHS)[0]; del H, RHS
        x = np.matmul(Q[:, :k+1], y)
        
        # Update Residual
        r = b - np.matmul(A, x)
        beta = np.linalg.norm(r)
    
    return x

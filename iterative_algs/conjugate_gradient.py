import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iters=None):
    """
    Solves Ax = b using the Conjugate Gradient method
    Args:
        A: m x m symmetric positive definite matrix, stored as ndarray
        b: m x 1 right hand side, stored as ndarray
        x0: initial guess
        tol: convergence tolerance
        max_iters: maximum number of iterations
    Returns:
        x: approximate solution
    """
    
    if b.ndim == 1:
        b = np.expand_dims(b, 1)

    m = A.shape[0]
    
    if max_iters is None:
        max_iters = m
    
    if max_iters is not None and max_iters <= 0:
        return

    # Initial Guess
    if x0 is None:
        x = np.zeros((m, 1))
    else:
        x = x0.copy(); del x0

    # Initial Residual
    r = b - np.matmul(A, x)
    # Search Direction
    p = r.copy()
    # Norm of residual
    r_norm_old = np.dot(r.squeeze(), r.squeeze())

    for i in range(max_iters):
        Ap = np.matmul(A, p)
        
        # Step Length
        alpha = r_norm_old / np.dot(p.squeeze(), Ap.squeeze())
        # Update Solution
        x += (alpha * p)
        # Update Residual
        r -= (alpha * Ap)
        r_norm_new = np.dot(r.squeeze(), r.squeeze())
        
        # Convergence Criteria
        if np.sqrt(r_norm_new) < tol:
            break
        
        # Search Direction
        p = r + (r_norm_new / r_norm_old) * p

        r_norm_old = r_norm_new

    return x




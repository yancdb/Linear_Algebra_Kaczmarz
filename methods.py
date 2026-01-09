import numpy as np
import time

def residual_inf(A, x, b):
    """Calcula o resíduo infinito ||Ax - b||_inf"""
    return np.max(np.abs(A @ x - b))


def precond_kaczmarz(A, b, x0=None, max_iter=1000,
                     log_interval=1, use_precond=False, P=None,
                     method="cyclic", sketch_factor=4, seed=None):
    m, n = A.shape
    x0 = np.zeros((n,1)) if x0 is None else x0.reshape(-1,1)
    
    # ------------------------------
    # Precond
    # ------------------------------
    if use_precond:
        if P is None:
            rng = np.random.default_rng(seed)
            r = sketch_factor * n
            idx = rng.choice(m, size=r, replace=False if r<=m else True)
            A_sketch = A[idx,:]
            Q,R = np.linalg.qr(A_sketch, mode='reduced')

            P = np.linalg.pinv(R)
            print(np.shape(P))
            # Optional: rescale so ||A P||_2 <= 1
            sigma_max = np.linalg.norm(A @ P, 2)
            if sigma_max > 1:
                P /= sigma_max
    else:
        P = np.eye(n)

    # ------------------------------
    #  A P y = b
    # ------------------------------
    y = np.zeros((n,1))         
    conv_x = [x0.copy()]
    A_pre = A @ P
    b_vec = b.reshape(-1,1)
    
    for iter_counter in range(max_iter):
        i = iter_counter % m if method=="cyclic" else np.random.randint(m)
        ai = A_pre[i,:].reshape(1,-1)
        residual = ai @ y - b_vec[i]
        y -= (residual / np.sum(ai**2)) * ai.T
        
        if (iter_counter+1) % log_interval==0:
            conv_x.append(P @ y)  ##x = P y

    # ------------------------------
    # Recupera solução #x= P y
    # ------------------------------
    x = P @ y
    return {"x": x, "conv_x": conv_x}


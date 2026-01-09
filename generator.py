import numpy as np

def generate_system(m, n, sigma_min, sigma_max, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # 1. Generate V (always n x n)
    V, _ = np.linalg.qr(np.random.randn(n, n))

    # 2. Singular values
    s = np.linspace(sigma_max, sigma_min, min(m, n))

    # 3. Diagonal S (m x n)
    S = np.zeros((m, n))
    for i in range(len(s)):
        S[i, i] = s[i]

    # 4. Generate U
    if m <= n:
        # underdetermined: U m x m
        U, _ = np.linalg.qr(np.random.randn(m, m),mode='complete')
    else:
        # overdetermined: U m x n
        U, _ = np.linalg.qr(np.random.randn(m, n),mode='complete')

    # 5. Construct A
    A = U @ S @ V.T

    # 6. True solution and rhs
    x_true = np.random.rand(n, 1)
    b = A @ x_true

    # 7. Initial guess
    x0 = np.zeros((n, 1))

    return A, x_true, b, x0
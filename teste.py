import numpy as np
from generator import generate_system
from methods import precond_kaczmarz  # updated version with sketch option

# Example system
m, n = 50, 30
A, x_true, b, x0 = generate_system(m, n, sigma_min=1.0, sigma_max=10.0, seed=42)

# Identity preconditioner
P = np.eye(n)

# Run “preconditioned” Kaczmarz with identity
result = precond_kaczmarz(A, b, x0=x0, max_iter=500, log_interval=50,
                          use_precond=True, P=P, method="cyclic")

# Compare with standard Kaczmarz (no preconditioner)
result_std = precond_kaczmarz(A, b, x0=x0, max_iter=500, log_interval=50,
                              use_precond=False, method="cyclic")

# Check if solutions are the same
print("Difference between identity-preconditioned and standard Kaczmarz:",
      np.linalg.norm(result["x"] - result_std["x"]))

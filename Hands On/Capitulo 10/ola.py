import numpy as np

mu = np.array([1.0, 1.0, 1.0])
Sigma = np.array([[4.0, 6.0, 1.0],
                  [6.0, 25.0, 5.5],
                  [1.0, 5.5, 2.25]])

lower = np.array([-np.inf, 1, 2])
upper = np.array([0.5, np.inf, np.inf])

n_samples = 10_000_000 

samples = np.random.multivariate_normal(mu, Sigma, size=n_samples)
inside = np.all((samples >= lower) & (samples <= upper), axis=1)
p = inside.mean()

print(inside)

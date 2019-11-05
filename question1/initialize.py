import numpy as np

def initialize(c, n, p):
    # Randomly initialize the fuzzy membership degree
    # u( clusters (c), points (n))
    u = np.random.rand(c, n)
    sum = np.sum(u, axis = 0) 
    for i in range(n):
        for j in range(c):
            u[j,i] = u[j,i] / sum[i]

    # Initialize weights of the variables
    # lamb(features (p))
    lamb = np.ones(p)

    # Initialize cluster centroids
    # v(clusters (c), features (p))
    v = np.random.rand(c, p)

    return {'u': u, 'lamb': lamb, 'v': v}
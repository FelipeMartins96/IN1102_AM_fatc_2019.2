import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import pdist

# Calculate Kernel
def gaussian(x,v,sigma):
    return math.exp((-((x-v)**2))/(sigma))


# Read data from Image Segmentation Database
data = pd.read_csv('data/seg.test')

# Splits into shape view and rgb view
# First 9 features
# shape_view([2100]points (n), [9]features (p))
shape_view = data.values[:, 0:9]
# 10 Remaining features
# rgb_view([2100]points (n), [10]features (p))
rgb_view = data.values[:, 9:19]

data = {'rgb': rgb_view, 'shape': shape_view}

# Number of Clusters
c = 7
# Fuzziness of membership
m = 1.6
# Iteration limit
T = 150
# Error threshold
e = 10e-10

for name, view in data.items():

    # Number of points
    n = view.shape[0]
    # Number of features
    p = view.shape[1]

    sigma = []
    for j in range(p):
        dist = pdist(view[:,j].reshape(-1,1)) # Size given by Binominal Coefficient
        mean = np.mean([np.quantile(dist, 0.1), np.quantile(dist, 0.9)])
        sigma.append(mean)

    for epoch in range(100):
        print("epoch ", epoch)
        
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

        J = float("inf")
        best_J = float("inf")

        for it in range(T):
            print("iteration ", it)

            # Update cluster centrois v
            for i in range(c):
                for j in range(p):
                    a = 0
                    b = 0
                    for k in range(n):
                        a += ((u[i,k])**m) * gaussian(view[k,j], v[i,j], sigma[j]) * view[k,j]
                        b +=  ((u[i,k])**m) * gaussian(view[k,j], v[i,j], sigma[j])
                    v[i,j] = a / b

            # Update features weights
            a = 1
            for j in range(p):
                b = 0
                for i in range(c):
                    for k in range(n):
                        b += ((u[i,k])**m) * (2 * (1 - gaussian(view[k,j], v[i,j], sigma[j])))
                a *= b
            for j in range(p):
                b = 0
                for i in range(c):
                        for k in range(n):
                            b += ((u[i,k])**m) * (2 * (1 - gaussian(view[k,j], v[i,j], sigma[j])))

                lamb[j] = (a ** (1/p)) / b
                
            # Update fuzzy membership degree 
            for i in range(c):
                for k in range(n):
                    a = 0
                    for h in range(c):
                        phi_a = 0
                        phi_b = 0
                        for j in range(p):
                            phi_a += lamb[j] * 2 *(1 - gaussian(view[k,j], v[i,j], sigma[j]))
                            phi_b += lamb[j] * 2 *(1 - gaussian(view[k,j], v[h,j], sigma[j]))
                        a += (phi_a / phi_b)**(1/(m-1))
                    u[i,k] = a ** (-1)

            # Calculate J 
            J_prev = J
            J = 0
            for i in range(c):
                for k in range(n):
                    phi = 0
                    for j in range(p):
                        phi += lamb[j] * 2 *(1 - gaussian(view[k,j], v[i,j], sigma[j]))
                    J +=  ((u[i,k])**m) * phi
            
            print(J)
            if (J_prev - J) < e:
                if J_prev < J:
                    print("ERROR!")
                break
            
        if J < best_J:
            best_u = u
            best_lamb = lamb
            best_v = v
            best_J = J
    
    # 
    best_J = np.array([best_J])
    np.savetxt(name + "/best_u.csv", best_u, delimiter=",")
    np.savetxt(name + "/best_lamb.csv", best_lamb, delimiter=",")
    np.savetxt(name + "/best_v.csv", best_v, delimiter=",")
    np.savetxt(name + "/best_J.csv", best_J, delimiter=",")
    



        
        





        







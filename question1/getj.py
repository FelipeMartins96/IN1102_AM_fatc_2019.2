from question1.gaussian import gaussian

def getj(c, n, p, m, sigma, view, v, lamb, u):
    # Calculate J 
    J = 0
    for i in range(c):
        for k in range(n):
            phi = 0
            for j in range(p):
                phi += lamb[j] * 2 *(1 - gaussian(view[k,j], v[i,j], sigma[j]))
            J +=  ((u[i,k])**m) * phi
    return J